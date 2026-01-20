from typing import Optional, Dict, List, Sequence, Any
import logging
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
import mlflow

logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def build_feature_map_from_names(
    column_transformer: Any,
    categorical_vars: Sequence[str],
    numeric_vars: Sequence[str],
    free_text_colname: Optional[str] = None,
) -> Dict[str, List[int]]:
    """
    Constructs a mapping from original feature names to their corresponding output indices 
    in the transformed feature matrix.

    This function relies on the `get_feature_names_out()` method of a fitted sklearn `ColumnTransformer`.
    It handles:
    1. Categorical features: Often expanded into multiple columns via OneHotEncoder (e.g., "color" -> "color_red", "color_blue").
       It attempts to find all output columns starting with "{feature_name}_".
    2. Numeric features: Usually map 1-to-1 to an output column.
    3. Binary/Ordinal features: May map 1-to-1 without a prefix change.

    Args:
        column_transformer (Any): A fitted sklearn ColumnTransformer. Typed as Any to avoid 
                                  hard dependency on scikit-learn in this module.
        categorical_vars (Sequence[str]): List of categorical feature names.
        numeric_vars (Sequence[str]): List of numeric feature names.
        free_text_colname (Optional[str]): Name of a text column to exclude from the map 
                                           (e.g., if it's handled separately by a BERT tokenizer).

    Returns:
        Dict[str, List[int]]: A dictionary where keys are original feature names and values 
                              are lists of indices in the transformed output vector.
    
    Raises:
        ValueError: If the column_transformer is not fitted.
    """
    if not hasattr(column_transformer, "transformers_"):
        raise ValueError("ColumnTransformer must be fitted before building the feature map.")
 
    # Get all output feature names from the transformer
    # Example: ['age', 'income', 'city_Berlin', 'city_Munich', ...]
    out_names = list(column_transformer.get_feature_names_out())
    name_to_idx = {name: i for i, name in enumerate(out_names)}
 
    feature_map: Dict[str, List[int]] = {}
    
    # Handle Categorical Variables
    for feature in categorical_vars:
        # Strategy A: Look for One-Hot-Encoded prefixes (e.g. "city" -> "city_Berlin")
        prefix = f"{feature}_"
        indices = [i for i, name in enumerate(out_names) if name.startswith(prefix)]
        
        # Strategy B: Fallback to exact match
        if not indices:
             if feature in name_to_idx:
                 indices = [name_to_idx[feature]]
             else:
                 logger.warning(f"Feature '{feature}' not found in ColumnTransformer output.")
                 pass 
        
        if indices:
            feature_map[feature] = indices
 
    # Handle Numeric Variables
    for feature in numeric_vars:
        # Skip the free text column if it is explicitly excluded
        if feature == free_text_colname:
            continue
            
        # Numeric features typically have a direct 1-to-1 mapping
        idx = name_to_idx.get(feature)
        if idx is not None:
            feature_map[feature] = [idx]
        else:
            logger.debug(f"Numeric feature '{feature}' not found in output (possibly dropped or renamed).")
 
    return feature_map


# =============================================================================
# Tokenizers
# =============================================================================

class TabularTokenizer(nn.Module):
    """
    V1 Tokenizer: Project raw tabular features (B, F) into tab tokens (B, T, H).
    Each projection head maps the full feature vector into one H-dim token.
    """
    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_tab_tokens: int = 4,
        dropout: float = 0.1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_tab_tokens = num_tab_tokens
 
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_size),
            )
            for _ in range(num_tab_tokens)
        ])
 
    def forward(self, tabular_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular_input: (B, F)
        Returns:
            (B, T_tab, H)
        """
        tokens = [proj(tabular_input).unsqueeze(1) for proj in self.projections]
        return torch.cat(tokens, dim=1)


class TabularTokenizerV2(nn.Module):
    """
    V2 Tokenizer: Per-feature tokenization.
    Projections are materialized based on feature_map.
    """
    def __init__(
        self,
        hidden_size: int,
        feature_map: Dict[str, List[int]],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
 
        self.feature_map: Dict[str, List[int]] = feature_map
        self._feature_dims: Dict[str, int] = {k: len(v) for k, v in feature_map.items()}
 
        self.projections = nn.ModuleDict()
        for fname, in_dim in self._feature_dims.items():
            self.projections[fname] = nn.Sequential(
                nn.Linear(in_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.LayerNorm(self.hidden_size),
            )
            logger.debug(f"[TabularTokenizerV2] init '{fname}' ({in_dim}->{self.hidden_size})")
 
    def forward(self, tabular_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tabular_input: (B, F_flat)
        Returns:
            (B, T, H)
        """
        B, device = tabular_input.shape[0], tabular_input.device
        feature_tokens = []
        for fname, idxs in self.feature_map.items():
            x = tabular_input[:, idxs]
            token = self.projections[fname](x)         # (B, H)
            feature_tokens.append(token.unsqueeze(1))  # (B, 1, H)

        if not feature_tokens:
            return torch.empty(B, 0, self.hidden_size, device=device)
        
        return torch.cat(feature_tokens, dim=1)


# =============================================================================
# Attention Blocks
# =============================================================================

class CrossAttentionBlock(nn.Module):
    """
    Standard Cross-attention (CLS -> tab tokens) with Pre-LN + FFN.
    No gating.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(
        self,
        cls_token: torch.Tensor,
        tab_tokens: torch.Tensor,
        tab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = (tab_mask == 0) if tab_mask is not None else None

        # Pre-LN
        q = self.layer_norm_1(cls_token)
        k = self.layer_norm_1(tab_tokens)
        v = k
        
        # Attention
        use_cuda = q.is_cuda
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", enabled=False):
            q32, k32, v32 = q.float(), k.float(), v.float()
            attn_out32, _ = self.attention(
                q32, k32, v32,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
        attn_out = attn_out32.to(q.dtype)
        
        # Residual
        x = cls_token + self.dropout_attn(attn_out)
        
        # FFN
        y = self.feed_forward_network(self.layer_norm_2(x))
        out = x + self.dropout_ffn(y)
        
        return out


class GatedCrossAttentionBlock(nn.Module):
    """
    Cross-attention with Flamingo-style gating.
    Supports enabling/disabling gating via `use_gating` flag.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
        *,
        use_gating: bool = True,
        init_alpha_xattn: float = 0.0,
        init_alpha_dense: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )
        self.layer_norm_1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.layer_norm_2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
 
        self.use_gating = use_gating
        self.alpha_xattn = nn.Parameter(torch.tensor(float(init_alpha_xattn)))
        self.alpha_dense = nn.Parameter(torch.tensor(float(init_alpha_dense)))
        
        logger.debug(f"GatedCrossAttentionBlock initialized (use_gating={use_gating})")
    
    def get_gate_values(self) -> dict[str, float] | None:
        if not self.use_gating:
            return None
        def _to_float(x):
            if isinstance(x, torch.Tensor):
                return float(torch.tanh(x).detach().cpu().item())
            return float(x)
        return {
            "gate_attn_tanh": _to_float(self.alpha_xattn),
            "gate_ffn_tanh":  _to_float(self.alpha_dense),
        }
 
    def forward(
        self,
        cls_token: torch.Tensor,
        tab_tokens: torch.Tensor,
        tab_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding_mask = (tab_mask == 0) if tab_mask is not None else None
 
        # Pre-LN
        q = self.layer_norm_1(cls_token)
        k = self.layer_norm_1(tab_tokens)
        v = k
 
        use_cuda = q.is_cuda
        with torch.autocast(device_type="cuda" if use_cuda else "cpu", enabled=False):
            q32, k32, v32 = q.float(), k.float(), v.float()
            attn_out32, _ = self.attention(
                q32, k32, v32,
                key_padding_mask=key_padding_mask,
                need_weights=False
            )
        attn_out = attn_out32.to(q.dtype)
 
        # Residual + Gating
        if self.use_gating:
            attn_res = torch.tanh(self.alpha_xattn) * self.dropout_attn(attn_out)
        else:
            attn_res = self.dropout_attn(attn_out)
        
        attn_sum = cls_token + attn_res

        # FFN
        ffn_out = self.feed_forward_network(self.layer_norm_2(attn_sum))

        # Residual + Gating
        if self.use_gating:
            ffn_res = torch.tanh(self.alpha_dense) * self.dropout_ffn(ffn_out)
        else:
            ffn_res = self.dropout_ffn(ffn_out)

        out = attn_sum + ffn_res
        return out


# =============================================================================
# CrossBert Model
# =============================================================================

class CrossBert(BertForSequenceClassification):
    """
    Unified CrossBert model supporting:
    - Tokenizer V1 (Projection) or V2 (Per-Feature)
    - CrossAttentionBlock (Standard) or GatedCrossAttentionBlock (Flamingo)
    - Early / Late / Both / None cross-attention positions
    """
    def __init__(
        self,
        cfg: BertConfig,
        extra_data_dim: int,
        cross_attention_positions: Dict[str, bool],
        # Tokenizer V1 specific
        num_tab_tokens: int = 4,
        # Tokenizer V2 specific
        feature_map: Optional[Dict[str, List[int]]] = None,
        use_tabular_tokenizer_v2: bool = False,
        # Gating specific
        use_gating: bool = False,
        init_alpha_xattn: float = 0.0,
        init_alpha_dense: float = 0.0,
        log_gates_mlflow: bool = True,
        # General Cross-Attention settings
        cross_attention_heads: Optional[int] = None,
        cross_attention_intermediate_size: Optional[int] = None,
        **kwargs
    ):
        super().__init__(cfg)
        self.config: BertConfig
        self.hidden_size = cfg.hidden_size
        self.num_tab_features = extra_data_dim
        self.cross_attn_heads = cross_attention_heads or cfg.num_attention_heads
        self.intermediate_size = cross_attention_intermediate_size or cfg.intermediate_size
        self.hidden_dropout_prob = cfg.hidden_dropout_prob
        self.log_gates_mlflow = log_gates_mlflow
        self._gate_log_step = 1

        # 1) Tabular Tokenizer Selection
        self.tabular_tokenizer = None
        if extra_data_dim > 0:
            if use_tabular_tokenizer_v2:
                if feature_map is None:
                    raise ValueError("feature_map must be provided when use_tabular_tokenizer_v2=True")
                self.tabular_tokenizer = TabularTokenizerV2(
                    hidden_size=self.hidden_size,
                    feature_map=feature_map,
                    dropout=self.hidden_dropout_prob
                )
            else:
                self.tabular_tokenizer = TabularTokenizer(
                    num_features=self.num_tab_features,
                    hidden_size=self.hidden_size,
                    num_tab_tokens=num_tab_tokens,
                    dropout=self.hidden_dropout_prob
                )

        # 2) Cross Attention Block Selection
        if use_gating:
            BlockClass = GatedCrossAttentionBlock
            # Pass gating-specific args
            block_kwargs = {
                "use_gating": True,
                "init_alpha_xattn": init_alpha_xattn,
                "init_alpha_dense": init_alpha_dense
            }
        else:
            BlockClass = CrossAttentionBlock
            block_kwargs = {}

        # 3) Instantiate Early/Late Blocks
        self.cross_attn_early = None
        self.cross_attn_late = None
        
        common_kwargs = {
            "embed_dim": self.hidden_size,
            "num_heads": self.cross_attn_heads,
            "dim_feedforward": self.intermediate_size,
            "dropout": self.hidden_dropout_prob,
            "batch_first": True,
        }
        common_kwargs.update(block_kwargs)

        if cross_attention_positions.get("early", False):
            self.cross_attn_early = BlockClass(**common_kwargs)

        if cross_attention_positions.get("late", False):
            self.cross_attn_late = BlockClass(**common_kwargs)

        # 4) Post-fusion
        self.post_fusion_ln = nn.LayerNorm(self.hidden_size)
        self.post_fusion_dropout = nn.Dropout(self.hidden_dropout_prob)

        logger.info(
            "CrossBert initialized: use_v2_tokenizer=%s, use_gating=%s, early=%s, late=%s",
            use_tabular_tokenizer_v2, use_gating,
            bool(self.cross_attn_early), bool(self.cross_attn_late)
        )

        # Initialize custom weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes the weights of the custom modules."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
        # Gating params init is handled in GatedCrossAttentionBlock.__init__ if parameters are created there
        # but we can re-enforce if needed.
        elif isinstance(module, GatedCrossAttentionBlock):
             pass # Already initialized in __init__

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        extra_data: Optional[torch.Tensor] = None,
        tab_mask: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        if return_dict is None:
            return_dict = True

        # 1) Build tabular tokens
        tab_tokens = None
        if extra_data is not None and self.tabular_tokenizer is not None:
            tab_tokens = self.tabular_tokenizer(extra_data)  # (B, T_tab, H)

        # 2) Text Embeddings
        if inputs_embeds is None:
            embeddings = self.bert.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=None,
                past_key_values_length=0,
            )
        else:
            embeddings = inputs_embeds

        # 3) EARLY Cross-Attention
        if (self.cross_attn_early is not None) and (tab_tokens is not None):
            cls_early = embeddings[:, :1, :]
            cls_early = self.cross_attn_early(cls_early, tab_tokens, tab_mask=tab_mask)
            embeddings = torch.cat([cls_early, embeddings[:, 1:, :]], dim=1)
            
            # Log gating values if applicable
            if self.log_gates_mlflow and self.training and hasattr(self.cross_attn_early, "get_gate_values"):
                vals = self.cross_attn_early.get_gate_values()
                if vals:
                    mlflow.log_metric("gate_attn_tanh_early", vals["gate_attn_tanh"], step=self._gate_log_step)
                    mlflow.log_metric("gate_ffn_tanh_early",  vals["gate_ffn_tanh"],  step=self._gate_log_step)

        # 4) BERT Encoder
        encoder_outputs = self.bert.encoder(
            hidden_states=embeddings,
            attention_mask=self.bert.get_extended_attention_mask(attention_mask, attention_mask.shape, attention_mask.device) 
            if attention_mask is not None else None,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        cls_token = sequence_output[:, :1, :]

        # 5) LATE Cross-Attention
        if (self.cross_attn_late is not None) and (tab_tokens is not None):
            cls_token = self.cross_attn_late(cls_token, tab_tokens, tab_mask=tab_mask)

            if self.log_gates_mlflow and self.training and hasattr(self.cross_attn_late, "get_gate_values"):
                vals = self.cross_attn_late.get_gate_values()
                if vals:
                    mlflow.log_metric("gate_attn_tanh_late", vals["gate_attn_tanh"], step=self._gate_log_step)
                    mlflow.log_metric("gate_ffn_tanh_late",  vals["gate_ffn_tanh"],  step=self._gate_log_step)

        if self.log_gates_mlflow and self.training:
            self._gate_log_step += 1

        # 6) Classification
        fused_cls = self.post_fusion_dropout(self.post_fusion_ln(cls_token)).squeeze(1)
        logits = self.classifier(fused_cls)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
            attentions=encoder_outputs.attentions if output_attentions else None,
        )
