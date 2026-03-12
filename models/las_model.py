"""
LAS Model Implementation
Lightweight Affordance Segmentor for Open-Vocabulary 3D Object Affordance Grounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
import os
from .pointmae_official import MaskTransformer, Group, PointNetFeaturePropagation
from utils.checkpoint import load_model_from_ckpt


class LASModel(nn.Module):
    """
    LAS (Lightweight Affordance Segmentor) Model
    
    Architecture:
    1. Foundation Model Encoders (DINOv2-Base/RoBERTa + Point-MAE)
    2. Unified Sequence Concatenation
    3. Co-attentional Transformer
    4. Dense Segmentation Head
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.model_config = config['model']
        
        # Determine prompt type: 'visual' or 'text'
        self.prompt_type = config['model'].get('prompt_type', 'visual')
        
        # Initialize point encoder (always needed)
        self.point_encoder = PointMAEEncoder(config=config['model']['point_encoder'])
        self.point_feature_dim = self.point_encoder.trans_dim
        
        # Initialize prompt encoder based on type
        if self.prompt_type == 'visual':
            # Visual prompt encoder (DINOv2/v3)
            self.prompt_encoder = DinoImageEncoder(
                model_name=config['model']['image_encoder']['model_name'],
                frozen=config['model']['image_encoder']['frozen'],
                finetune_layers=config['model']['image_encoder'].get('finetune_layers', 2),
                dino_local_repo=config['model']['image_encoder'].get('dino_local_repo', None),
                dino_local_weights_name=config['model']['image_encoder'].get('dino_local_weights_name', None),
                apply_post_layernorm=config['model']['image_encoder'].get('post_layernorm', True)
            )
            self.prompt_feature_dim = self.prompt_encoder.feature_dim
        elif self.prompt_type == 'text':
            # Text prompt encoder (RoBERTa)
            self.prompt_encoder = RobertaTextEncoder(
                model_name=config['model']['text_encoder']['model_name'],
                frozen=config['model']['text_encoder']['frozen'],
                finetune_layers=config['model']['text_encoder'].get('finetune_layers', 2),
                max_length=config['model']['text_encoder'].get('max_length', 128),
                local_model_path=config['model']['text_encoder'].get('local_model_path', None)
            )
            self.prompt_feature_dim = self.prompt_encoder.feature_dim
        else:
            raise ValueError(f"Unsupported prompt_type: {self.prompt_type}. Must be 'visual' or 'text'.")
        
        # Unified sequence parameters
        self.unified_dim = config['model']['unified_sequence']['unified_dim']
        
        # Feature projection layers to unified dimension
        self.prompt_projection = nn.Linear(self.prompt_feature_dim, self.unified_dim)
        self.point_projection = nn.Linear(self.point_feature_dim, self.unified_dim)
        
        # Modality type embeddings
        self.point_type_embedding = nn.Parameter(torch.randn(1, 1, self.unified_dim))
        self.prompt_type_embedding = nn.Parameter(torch.randn(1, 1, self.unified_dim))
        
        # Feature propagation
        self.feature_propagation = PointNetFeaturePropagation(
            in_channel=self.unified_dim,
            mlp=[self.unified_dim, self.unified_dim]
        )
        
        # Co-attentional Transformer
        self.co_attention_transformer = CoAttentionTransformer(
            d_model=self.unified_dim,
            nhead=config['model']['co_attention']['num_heads'],
            num_layers=config['model']['co_attention']['num_layers'],
            dim_feedforward=config['model']['co_attention']['dim_feedforward'],
            dropout=config['model']['co_attention']['dropout']
        )
        
        # Dense segmentation head
        self.segmentation_head = DenseSegmentationHead(
            input_dim=self.unified_dim,
            hidden_dim=config['model']['segmentation_head']['hidden_dim'],
            num_layers=config['model']['segmentation_head']['num_layers'],
            dropout=config['model']['segmentation_head']['dropout']
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        # Initialize type embeddings
        nn.init.normal_(self.point_type_embedding, std=0.02)
        nn.init.normal_(self.prompt_type_embedding, std=0.02)
        
        # Initialize projection layers
        nn.init.xavier_uniform_(self.prompt_projection.weight)
        nn.init.xavier_uniform_(self.point_projection.weight)
        nn.init.constant_(self.prompt_projection.bias, 0)
        nn.init.constant_(self.point_projection.bias, 0)
    
    def forward(self, batch):
        """
        Forward pass of LAS model
        
        Args:
            batch: Dictionary containing:
                - 'points': (B, N, 3) point clouds
                - 'image': (B, 3, H, W) images (for visual prompt)
                - 'text': List of strings (for text prompt)
                - 'gt_mask': (B, N, 1) ground truth masks
        
        Returns:
            outputs: Dictionary containing:
                - 'segmentation_logits': (B, N, 1) segmentation predictions
                - 'point_features': (B, N, unified_dim) fused point features
                - 'prompt_features': (B, seq_len, unified_dim) prompt features
        """
        # Extract inputs
        points = batch['points']  # (B, N, 3)
        B, N, _ = points.shape
        
        # Validate input dimensions
        if N == 0:
            raise ValueError("Empty point cloud detected in batch")
        if points.shape[2] != 3:
            raise ValueError(f"Invalid point cloud format: expected 3 coordinates, got {points.shape[2]}")
        
        # Ensure all samples in batch have same number of points
        expected_points = points.shape[1]
        if 'gt_mask' in batch:
            gt_mask = batch['gt_mask']
            if gt_mask.shape[1] != expected_points:
                raise ValueError(f"Point-mask dimension mismatch: {expected_points} points vs {gt_mask.shape[1]} mask entries")
        
        # Encode point clouds
        point_group_features, point_group_centers = self.point_encoder(points)
        
        # Encode prompts based on type
        prompt_attention_mask = None
        if self.prompt_type == 'visual':
            images = batch['image']  # (B, 3, H, W)
            prompt_features = self.prompt_encoder(images)  # (B, H*W, prompt_feature_dim)
            # For visual prompts, all tokens are valid (no padding)
            prompt_attention_mask = torch.ones(B, prompt_features.size(1), device=prompt_features.device)
        elif self.prompt_type == 'text':
            texts = batch['text']  # List of strings
            prompt_features, prompt_attention_mask = self.prompt_encoder(texts)  # (B, seq_len, prompt_feature_dim), (B, seq_len)
        
        # Project to unified dimension
        prompt_features_proj = self.prompt_projection(prompt_features)
        point_features_proj = self.point_projection(point_group_features)
        
        # Upsample point features
        points_transposed = points.transpose(1, 2).contiguous()
        point_group_centers_transposed = point_group_centers.transpose(1, 2).contiguous()
        point_features_proj_transposed = point_features_proj.transpose(1, 2).contiguous()

        # --- Debug validations before feature propagation ---
        def _validate_tensor(t: torch.Tensor, name: str):
            if t is None:
                raise RuntimeError(f"{name} is None")
            if t.numel() == 0:
                raise RuntimeError(f"{name} has zero elements, shape={tuple(t.shape)}")
            if not torch.isfinite(t).all():
                bad = (~torch.isfinite(t)).nonzero(as_tuple=False)
                sample = bad[:5].tolist() if bad.numel() > 0 else []
                raise RuntimeError(f"{name} contains NaN/Inf, shape={tuple(t.shape)}, samples_idx={sample}")
            if not t.is_contiguous():
                # Keep contiguous for cuDNN conv1d
                t = t.contiguous()
            if t.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                raise RuntimeError(f"{name} dtype must be floating, got {t.dtype}")
            return t

        points_transposed = _validate_tensor(points_transposed, "points_transposed")
        point_group_centers_transposed = _validate_tensor(point_group_centers_transposed, "point_group_centers_transposed")
        point_features_proj_transposed = _validate_tensor(point_features_proj_transposed, "point_features_proj_transposed")
        
        # Validate dimensions before feature propagation
        if points_transposed.shape[2] != N:
            raise ValueError(f"Points dimension mismatch: expected {N}, got {points_transposed.shape[2]}")
        if point_group_centers_transposed.shape[2] != point_group_features.shape[1]:
            raise ValueError(f"Group centers dimension mismatch: {point_group_centers_transposed.shape[2]} vs {point_group_features.shape[1]}")
        
        try:
            # Ensure fp inputs are float32 to avoid odd backend issues
            fp_points = points_transposed.to(dtype=torch.float32)
            fp_centers = point_group_centers_transposed.to(dtype=torch.float32)
            fp_feats = point_features_proj_transposed.to(dtype=torch.float32)

            try:
                upsampled_point_features = self.feature_propagation(
                    fp_points,
                    fp_centers,
                    None,
                    fp_feats
                )
            except RuntimeError as inner_e:
                # Retry without cuDNN as a fallback for mapping errors on some drivers
                if 'cudnn' in str(inner_e).lower() or 'cudnn_status_mapping_error' in str(inner_e).lower():
                    import torch.backends.cudnn as cudnn
                    prev = cudnn.enabled
                    try:
                        cudnn.enabled = False
                        upsampled_point_features = self.feature_propagation(
                            fp_points,
                            fp_centers,
                            None,
                            fp_feats
                        )
                    finally:
                        cudnn.enabled = prev
                else:
                    raise

            # Back to (B, N, C)
            upsampled_point_features = upsampled_point_features.transpose(1, 2).contiguous()

            # Validate output dimensions
            if upsampled_point_features.shape[1] != N:
                raise ValueError(
                    f"Feature propagation output mismatch: expected {N} points, got {upsampled_point_features.shape[1]}"
                )

        except RuntimeError as e:
            # Enrich diagnostics for cuDNN mapping errors
            msg = [
                "[LASModel] FeaturePropagation RuntimeError",
                f"points_transposed: shape={tuple(points_transposed.shape)}, dtype={points_transposed.dtype}, device={points_transposed.device}",
                f"centers_transposed: shape={tuple(point_group_centers_transposed.shape)}, dtype={point_group_centers_transposed.dtype}, device={point_group_centers_transposed.device}",
                f"feats_transposed: shape={tuple(point_features_proj_transposed.shape)}, dtype={point_features_proj_transposed.dtype}, device={point_features_proj_transposed.device}",
            ]
            try:
                mins = (
                    torch.nan_to_num(points_transposed).min().item(),
                    torch.nan_to_num(point_group_centers_transposed).min().item(),
                    torch.nan_to_num(point_features_proj_transposed).min().item(),
                )
                maxs = (
                    torch.nan_to_num(points_transposed).max().item(),
                    torch.nan_to_num(point_group_centers_transposed).max().item(),
                    torch.nan_to_num(point_features_proj_transposed).max().item(),
                )
                msg.append(f"mins={mins}, maxs={maxs}")
            except Exception:
                pass
            print("\n".join(msg))
            raise
        
        # Add modality type embeddings
        point_features_with_type = upsampled_point_features + self.point_type_embedding.expand(B, N, -1)
        prompt_features_with_type = prompt_features_proj + self.prompt_type_embedding.expand(B, prompt_features_proj.size(1), -1)
        
        # Create unified sequence
        unified_sequence = torch.cat([point_features_with_type, prompt_features_with_type], dim=1)
        
        # Create unified attention mask
        # Point cloud tokens are always valid (no padding)
        point_attention_mask = torch.ones(B, N, device=points.device)
        unified_attention_mask = torch.cat([point_attention_mask, prompt_attention_mask], dim=1)
        
        # Apply co-attentional transformer with attention mask
        fused_sequence = self.co_attention_transformer(unified_sequence, src_key_padding_mask=(unified_attention_mask == 0))
        
        # Split back to point and prompt features
        fused_point_features = fused_sequence[:, :N, :]
        fused_prompt_features = fused_sequence[:, N:, :]
        
        # Generate segmentation predictions
        segmentation_logits = self.segmentation_head(fused_point_features)
        
        outputs = {
            'segmentation_logits': segmentation_logits,
            'point_features': fused_point_features,
            'prompt_features': fused_prompt_features
        }
        
        return outputs

class DinoImageEncoder(nn.Module):
    """
    Hybrid DINOv2/v3 image encoder.
    - DINOv2 models are loaded via HuggingFace Transformers (`AutoModel`).
    - DINOv3 models are loaded via `torch.hub` from a local repository.
    """
    
    def __init__(self, model_name, frozen=False, finetune_layers=2, 
                 dino_local_repo=None, dino_local_weights_name=None,
                 apply_post_layernorm: bool = True):
        super().__init__()
        self.model_name = model_name
        self.is_dinov3 = 'dinov3' in model_name
        self.apply_post_layernorm = False  # only enable for DINOv3 by default

        if self.is_dinov3:
            # --- DINOv3 loading via torch.hub ---
            if dino_local_repo is None or dino_local_weights_name is None:
                raise ValueError("`dino_local_repo` and `dino_local_weights_name` must be provided for DINOv3.")
            
            weights_path = os.path.join(dino_local_repo, dino_local_weights_name)
            self.dino_model = torch.hub.load(dino_local_repo, model_name, source='local', weights=weights_path)
            self.feature_dim = self.dino_model.embed_dim
            print(f"Loaded DINOv3 model '{model_name}' from local path: {dino_local_repo}")
            # Enable post LayerNorm only for DINOv3 if requested
            self.apply_post_layernorm = bool(apply_post_layernorm)

        else:
            # --- DINOv2 loading via transformers ---
            self.dino_model = AutoModel.from_pretrained(model_name)
            self.feature_dim = self.dino_model.config.hidden_size
            print(f"Loaded DINOv2 model '{model_name}' from HuggingFace transformers.")
            # Keep post LayerNorm disabled for DINOv2 by default
            self.apply_post_layernorm = False

        # Post feature normalization (only constructed if used)
        if self.apply_post_layernorm:
            self.post_norm = nn.LayerNorm(self.feature_dim)

        # --- Fine-tuning strategy ---
        if frozen:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            if self.is_dinov3:
                self.dino_model.eval() # Hub models are nn.Modules, can be put in eval.
            print(f"Dino encoder ({model_name}) completely frozen.")
        else:
            # Freeze all parameters first
            for param in self.dino_model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last 'finetune_layers' based on model type
            if self.is_dinov3 and hasattr(self.dino_model, 'blocks'):
                # DINOv3 from torch.hub
                total_layers = len(self.dino_model.blocks)
                freeze_layers = max(0, total_layers - finetune_layers)
                for i in range(freeze_layers, total_layers):
                    for param in self.dino_model.blocks[i].parameters():
                        param.requires_grad = True
                
                # Unfreeze registers if they exist
                if hasattr(self.dino_model, 'reg_tokens'):
                    self.dino_model.reg_tokens.requires_grad = True

            elif not self.is_dinov3 and hasattr(self.dino_model, 'encoder') and hasattr(self.dino_model.encoder, 'layer'):
                # DINOv2 from transformers
                total_layers = len(self.dino_model.encoder.layer)
                freeze_layers = max(0, total_layers - finetune_layers)
                for i in range(freeze_layers, total_layers):
                    for param in self.dino_model.encoder.layer[i].parameters():
                        param.requires_grad = True
            else:
                # Fallback for models with unexpected structure
                print(f"Warning: Could not find standard layer structure for {model_name}. Fine-tuning all parameters.")
                for param in self.dino_model.parameters():
                    param.requires_grad = True

            trainable_params = sum(p.numel() for p in self.dino_model.parameters() if p.requires_grad)
            print(f"Dino encoder ({model_name}): fine-tuning with {trainable_params:,} trainable parameters.")

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) tensor of images
        
        Returns:
            features: (B, num_patches, feature_dim) patch features
        """
        if self.is_dinov3:
            # DINOv3 torch.hub API
            output = self.dino_model.forward_features(images)
            if isinstance(output, dict) and 'x_norm_patchtokens' in output:
                features = output['x_norm_patchtokens']
            else:
                features = output[:, 1:]  # Fallback, assumes CLS token is first
        else:
            # DINOv2 transformers API
            outputs = self.dino_model(images)
            features = outputs.last_hidden_state
            # Exclude CLS token for ViT-based models
            if hasattr(self.dino_model, 'embeddings') and hasattr(self.dino_model.embeddings, 'cls_token') and self.dino_model.embeddings.cls_token is not None:
                features = features[:, 1:]

        # Optional post LayerNorm for DINOv3 features
        if self.apply_post_layernorm:
            features = self.post_norm(features)

        return features

class RobertaTextEncoder(nn.Module):
    """
    RoBERTa text encoder for text prompt encoding
    """
    
    def __init__(self, model_name='roberta-base', frozen=False, finetune_layers=2, max_length=128, 
                 local_model_path=None):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.local_model_path = local_model_path
        
        # Load RoBERTa model and tokenizer
        if local_model_path is not None:
            # Load from local path
            import os
            if not os.path.isabs(local_model_path):
                # Convert relative path to absolute path from project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_model_path = os.path.join(project_root, local_model_path)
            
            print(f"Loading RoBERTa model from local path: {local_model_path}")
            self.roberta_model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            print(f"Loaded RoBERTa model '{model_name}' from local path: {local_model_path}")
        else:
            # Load from HuggingFace Hub
            self.roberta_model = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"Loaded RoBERTa model '{model_name}' from HuggingFace transformers.")
        
        # Get feature dimension
        self.feature_dim = self.roberta_model.config.hidden_size
        
        # Fine-tuning strategy
        if frozen:
            for param in self.roberta_model.parameters():
                param.requires_grad = False
            print(f"RoBERTa encoder ({model_name}) completely frozen.")
        else:
            # Freeze all parameters first
            for param in self.roberta_model.parameters():
                param.requires_grad = False
            
            # Unfreeze the last 'finetune_layers' layers
            if hasattr(self.roberta_model, 'encoder') and hasattr(self.roberta_model.encoder, 'layer'):
                total_layers = len(self.roberta_model.encoder.layer)
                freeze_layers = max(0, total_layers - finetune_layers)
                for i in range(freeze_layers, total_layers):
                    for param in self.roberta_model.encoder.layer[i].parameters():
                        param.requires_grad = True
                        
                # Also unfreeze the pooler if it exists
                if hasattr(self.roberta_model, 'pooler') and self.roberta_model.pooler is not None:
                    for param in self.roberta_model.pooler.parameters():
                        param.requires_grad = True
            else:
                # Fallback for models with unexpected structure
                print(f"Warning: Could not find standard layer structure for {model_name}. Fine-tuning all parameters.")
                for param in self.roberta_model.parameters():
                    param.requires_grad = True
            
            trainable_params = sum(p.numel() for p in self.roberta_model.parameters() if p.requires_grad)
            print(f"RoBERTa encoder ({model_name}): fine-tuning with {trainable_params:,} trainable parameters.")
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings (batch of text prompts)
        
        Returns:
            features: (B, seq_len, feature_dim) text features
            attention_mask: (B, seq_len) attention mask for padding tokens
        """
        # Tokenize texts
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.roberta_model.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Forward pass through RoBERTa
        outputs = self.roberta_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get sequence output (all token representations)
        features = outputs.last_hidden_state  # (B, seq_len, feature_dim)
        
        return features, attention_mask

class PointMAEEncoder(nn.Module):
    """
    Point-MAE encoder for point cloud feature extraction
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.trans_dim = config['embed_dim']
        self.group_size = config['group_size']
        self.num_group = config['num_group']
        
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # Create transformer_config object that MaskTransformer expects
        transformer_config = type('obj', (object,), {
            'mask_ratio': config.get('mask_ratio', 0.6),
            'trans_dim': config['embed_dim'],
            'depth': config['depth'],
            'drop_path_rate': config.get('drop_path_rate', 0.1),
            'num_heads': config['num_heads'],
            'decoder_depth': config.get('decoder_depth', 4),
            'decoder_num_heads': config.get('decoder_num_heads', 6),
            'mask_type': 'rand',
            'encoder_dims': config['embed_dim']
        })
        
        mae_config = type('obj', (object,), {
            'transformer_config': transformer_config,
            'group_size': self.group_size,
            'num_group': self.num_group,
            'loss': 'cdl1'
        })
        
        self.MAE_encoder = MaskTransformer(mae_config)
        
        if config.get('pretrain', False):
            print(f"Loading Point-MAE pre-trained model from {config['pretrain_path']}")
            load_model_from_ckpt(self.MAE_encoder, config['pretrain_path'])

        # Fine-tuning strategy
        frozen = config.get('frozen', False)
        if frozen:
            for param in self.MAE_encoder.parameters():
                param.requires_grad = False
            print("Point-MAE encoder completely frozen")
        else:
            for param in self.MAE_encoder.parameters():
                param.requires_grad = True
            # finetune_layers = config.get('finetune_layers', 2)
            
            # # Freeze all parameters first
            # for param in self.MAE_encoder.parameters():
            #     param.requires_grad = False
            
            # # Unfreeze the last 'finetune_layers' of the encoder
            # if hasattr(self.MAE_encoder, 'encoder') and hasattr(self.MAE_encoder.encoder, 'blocks'):
            #     total_layers = len(self.MAE_encoder.encoder.blocks)
            #     freeze_layers = max(0, total_layers - finetune_layers)
                
            #     for i in range(freeze_layers, total_layers):
            #         for param in self.MAE_encoder.encoder.blocks[i].parameters():
            #             param.requires_grad = True

            #     # Also unfreeze the final norm layer
            #     if hasattr(self.MAE_encoder.encoder, 'norm'):
            #          for param in self.MAE_encoder.encoder.norm.parameters():
            #             param.requires_grad = True

            trainable_params = sum(p.numel() for p in self.MAE_encoder.parameters() if p.requires_grad)
            print(f"Point-MAE encoder: fine-tuning all parameters ({trainable_params:,} parameters)")

    def forward(self, points):
        """
        Args:
            points: (B, N, 3) tensor of point clouds
        
        Returns:
            features: (B, num_group, embed_dim) patch features
            center: (B, num_group, 3) center points of the groups
        """
        neighborhood, center = self.group_divider(points)
        features, _, _ = self.MAE_encoder(neighborhood, center)
        return features, center

class CoAttentionTransformer(nn.Module):
    """
    Co-attentional Transformer for multi-modal feature fusion
    """
    
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, unified_sequence, src_key_padding_mask=None):
        """
        Args:
            unified_sequence: (B, seq_len, d_model) unified sequence
            src_key_padding_mask: (B, seq_len) mask for padding tokens (True for padding)
        
        Returns:
            fused_sequence: (B, seq_len, d_model) fused sequence
        """
        # Apply transformer with attention mask
        x = self.transformer(unified_sequence, src_key_padding_mask=src_key_padding_mask)
        
        # Layer norm
        x = self.norm(x)
        
        return x

class DenseSegmentationHead(nn.Module):
    """
    Dense segmentation head for point-wise predictions
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.1):
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, features):
        """
        Args:
            features: (B, N, input_dim) point features
        
        Returns:
            logits: (B, N, 1) segmentation logits
        """
        return self.mlp(features)

class LASLoss(nn.Module):
    """
    LAS loss combining Focal Loss and Dice Loss
    """
    
    def __init__(self, focal_alpha=0.25, focal_gamma=2.0, focal_weight=1.0, dice_weight=1.0):
        super().__init__()
        
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
    
    def focal_loss(self, predictions, targets, focal_alpha=0.25, focal_gamma=2.0):
        """
        Focal Loss implementation (modified to be similar to HM_Loss's structure)
        
        Args:
            predictions: (B, N, 1) logits (will be sigmoid activated internally)
            targets: (B, N, 1) binary targets (0 or 1)
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
        
        Returns:
            loss: scalar tensor
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten for easier computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # Add a small epsilon to probabilities to prevent log(0)
        epsilon = 1e-6
        probs_flat = torch.clamp(probs_flat, epsilon, 1.0 - epsilon)

        # Calculate loss for positive samples (target=1)
        # Term for true positives: -alpha * (1-p)^gamma * log(p) * target
        term_pos = -focal_alpha * ((1 - probs_flat) ** focal_gamma) * \
                torch.log(probs_flat) * targets_flat

        # Calculate loss for negative samples (target=0)
        # Term for true negatives: -(1-alpha) * p^gamma * log(1-p) * (1-target)
        term_neg = -(1 - focal_alpha) * (probs_flat ** focal_gamma) * \
                torch.log(1 - probs_flat) * (1 - targets_flat)

        # Sum the terms for all pixels and take the mean
        focal_loss = torch.mean(term_pos + term_neg)
        
        return focal_loss
    
    def dice_loss(self, predictions, targets, dice_smooth=1e-6):
        """
        Dice Loss implementation (modified to be similar to HM_Loss's structure - bi-directional)
        
        Args:
            predictions: (B, N, 1) logits (will be sigmoid activated internally)
            targets: (B, N, 1) binary targets (0 or 1)
            dice_smooth: Smoothing factor to prevent division by zero
        
        Returns:
            loss: scalar tensor
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(predictions)
        
        # Flatten for easier computation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        # --- Positive (Foreground) Dice Calculation ---
        intersection_positive = (probs_flat * targets_flat).sum()
        # Note: torch.abs is redundant here if probs_flat and targets_flat are 0-1
        cardinality_positive = (probs_flat + targets_flat).sum() 
        
        # Compute dice coefficient for positive class
        dice_positive = (2.0 * intersection_positive + dice_smooth) / \
                        (cardinality_positive + dice_smooth)

        # --- Negative (Background) Dice Calculation ---
        # Intersection for background: (1-pred) * (1-target)
        intersection_negative = ((1.0 - probs_flat) * (1.0 - targets_flat)).sum()
        # Cardinality for background: (1-pred) + (1-target)
        cardinality_negative = ((1.0 - probs_flat) + (1.0 - targets_flat)).sum()

        # Compute dice coefficient for negative class
        dice_negative = (2.0 * intersection_negative + dice_smooth) / \
                        (cardinality_negative + dice_smooth)

        # --- Combine Dice coefficients ---
        # HM_Loss uses 1.5 - dice_positive - dice_negative.
        # This is an unusual combination. A more common approach for bi-directional
        # Dice loss is (1 - dice_positive) + (1 - dice_negative) or their average.
        # I'm implementing the 1.5 - (...) form as you requested to match the HM_Loss structure.
        # Ensure this is what you intend, as it can lead to negative loss values if both dice_positive and dice_negative are high.
        dice_loss = 1.5 - dice_positive - dice_negative
        
        # HM_Loss also sums over some dimension here. For a single (B, N, 1) output,
        # the sum might not change the scalar result.
        # If this was for multi-class (B, N, C), this would sum across classes.
        # For a direct scalar output (like your original functions), we'll just return the scalar.
        return dice_loss
    
    def forward(self, logits, targets):
        """
        Compute total loss
        
        Args:
            logits: (B, N, 1) prediction logits
            targets: (B, N, 1) ground truth masks
        
        Returns:
            total_loss: scalar loss value
            loss_dict: dictionary of individual losses
        """
        # Compute individual losses
        focal_loss = self.focal_loss(logits, targets)
        dice_loss = self.dice_loss(logits, targets)
        
        # Total loss
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'focal_loss': focal_loss,
            'dice_loss': dice_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict

def create_las_model(config):
    """
    Create LAS model instance
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model: LAS model instance
    """
    return LASModel(config)