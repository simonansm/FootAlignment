import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionWithAttentionFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → InceptionV3
        self.backbone = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True  # must be True for pretrained
        )
        self.backbone.aux_logits = False  # disable aux logits at inference
        self.backbone.AuxLogits = nn.Identity()  # remove AuxLogits module
        self.backbone.fc = nn.Identity()  # remove classification head

        image_emb_dim = 2048  # InceptionV3 output features


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
        image_emb = self.backbone(image)  # [B, 2048]
        feature_emb = self.feature_net(features)  # [B, 128]
        combined = torch.cat([image_emb, feature_emb], dim=1)  # [B, 2176]

        attn_weights = self.attention(combined)  # [B, 2176]
        fused = combined * attn_weights  # [B, 2176]

        output = self.classifier(fused)
        return output


class GatedFusion(nn.Module):
    def __init__(self, image_dim, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(image_dim + feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, img_feat, tab_feat):
        gate = self.gate(torch.cat([img_feat, tab_feat], dim=1))  # [B, feature_dim]
        gated_tab = gate * tab_feat
        fused = torch.cat([img_feat, gated_tab], dim=1)
        return fused

class InceptionWithGatedFusion(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()

        # Step 1: Image → InceptionV3
        self.backbone = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True  # must be True for pretrained weights
        )
        self.backbone.aux_logits = False  # disable aux logits at inference
        self.backbone.AuxLogits = nn.Identity()  # remove AuxLogits module
        self.backbone.fc = nn.Identity()  # remove final classifier

        image_emb_dim = 2048  # InceptionV3 output features

        # Step 2: Tabular features → MLP
        self.feature_net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.5)
        )
        feature_emb_dim = 128

        # Step 3: Gated fusion
        self.gated_fusion = GatedFusion(image_emb_dim, feature_emb_dim)

        # Step 4: Final classification
        self.classifier = nn.Sequential(
            nn.Linear(image_emb_dim + feature_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, features):
        image_emb = self.backbone(image)             # [B, 2048]
        feature_emb = self.feature_net(features)     # [B, 128]
        fused = self.gated_fusion(image_emb, feature_emb)  # [B, 2176]
        output = self.classifier(fused)
        return output