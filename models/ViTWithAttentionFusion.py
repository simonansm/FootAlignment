import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ViTWithAttentionFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → ViT (Vision Transformer)
        self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.backbone.heads = nn.Identity()  # remove classification head
        image_emb_dim = 768  # ViT-B/16 output dim

        # Step 2: Tabular features → MLP
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.6)
        )
        feature_emb_dim = 128

        # Step 3: Attention fusion
        self.attention = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 2),  # only two weights needed
            nn.Softmax(dim=1)
        )

        # Step 4: Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        image_emb = self.backbone(image)             # [B, 768]
        feature_emb = self.feature_net(features)     # [B, 128]

        combined = torch.cat([image_emb, feature_emb], dim=1)  # [B, 896]
        attn_weights = self.attention(combined)                # [B, 2]

        # Apply learned weights
        image_weighted = image_emb * attn_weights[:, 0:1]
        feature_weighted = feature_emb * attn_weights[:, 1:2]

        fused = torch.cat([image_weighted, feature_weighted], dim=1)  # [B, 896]
        output = self.classifier(fused)
        return output
