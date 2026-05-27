import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNetWithAttentionFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → ResNet-50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity()  # Remove final classification head
        image_emb_dim = 2048  # ResNet-50 output dimension

        # Step 2: Features → MLP
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.8)
        )
        feature_emb_dim = 128

        # Step 3: Attention module
        self.attention = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, 2),  # weights for image and feature
            nn.Softmax(dim=1)
        )

        # Step 4: Final classification
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        image_emb = self.backbone(image)             # [B, 2048]
        feature_emb = self.feature_net(features)     # [B, 128]

        combined = torch.cat([image_emb, feature_emb], dim=1)  # [B, 2176]
        attn_weights = self.attention(combined)                # [B, 2]

        # Apply learned attention
        image_weighted = image_emb * attn_weights[:, 0:1]
        feature_weighted = feature_emb * attn_weights[:, 1:2]

        fused = torch.cat([image_weighted, feature_weighted], dim=1)  # [B, 2176]
        output = self.classifier(fused)
        return output