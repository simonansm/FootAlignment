import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class EfficientNetWithAttentionFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → EfficientNet
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Identity()
        image_emb_dim = 1280  # for efficientnet_b0

        # Step 2: Features → MLP
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        #    nn.Dropout(0.4)
        )
        feature_emb_dim = 128

        # Step 3: Attention module
        self.attention = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(256, 2),  # weights for image and feature
            nn.Softmax(dim=1)
        )

        # Step 4: Final classification
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        image_emb = self.backbone(image)             # [B, 1280]
        feature_emb = self.feature_net(features)     # [B, 128]

        combined = torch.cat([image_emb, feature_emb], dim=1)  # [B, 1408]
        attn_weights = self.attention(combined)                # [B, 2]

        # Apply attention to each modality
        image_weighted = image_emb * attn_weights[:, 0:1]
        feature_weighted = feature_emb * attn_weights[:, 1:2]

        fused = torch.cat([image_weighted, feature_weighted], dim=1)  # [B, 1408]
        output = self.classifier(fused)
        return output
    


class EfficientNetb2WithAttentionFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → EfficientNet-B2
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        image_emb_dim = 1408  # for efficientnet_b2

        # Step 2: Features → MLP
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.5)
        )
        feature_emb_dim = 128

        # Step 3: Attention module
        self.attention = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, image_emb_dim + feature_emb_dim),
            nn.Sigmoid()
        )


        # Step 4: Final classification
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        image_emb = self.backbone(image)             # [B, 1408]
        feature_emb = self.feature_net(features)     # [B, 128]
        combined = torch.cat([image_emb, feature_emb], dim=1)  # [B, 1536]

        attn_weights = self.attention(combined)      # [B, 1536]
        fused = combined * attn_weights              # [B, 1536]

        output = self.classifier(fused)
        return output

class EfficientNetB2HybridFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: EfficientNet backbone (image encoder)
        self.backbone = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Identity()
        image_emb_dim = 1408

        # Step 2: Tabular feature encoder
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU()
        )
        feature_emb_dim = 128

        # Step 3: Transformer-style cross-modal fusion
        fusion_dim = image_emb_dim + feature_emb_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.3,
            batch_first=True,
            activation="gelu"
        )
        self.transformer_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Optional: Add residual shortcut from original combined embedding
        self.residual_proj = nn.Identity()

        # Step 4: Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, features):
        B = image.size(0)
        image_emb = self.backbone(image)  # [B, 1408]
        feature_emb = self.feature_net(features)  # [B, 128]
        combined = torch.cat([image_emb, feature_emb], dim=1).unsqueeze(1)  # [B, 1, 1536]

        # Transformer-based attention fusion
        fused = self.transformer_fusion(combined).squeeze(1)  # [B, 1536]

        # (Optional) Add residual connection
        fused = fused + self.residual_proj(torch.cat([image_emb, feature_emb], dim=1))

        output = self.classifier(fused)
        return output