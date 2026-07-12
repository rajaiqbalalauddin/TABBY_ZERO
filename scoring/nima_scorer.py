"""
NIMA (Neural Image Assessment) scorer for Tabby Zero.

Architecture:
    MobileNetV2 backbone (pretrained on ImageNet)
    → Dropout(0.75)
    → Linear(1280, 10)
    → Softmax

Output:
    A 10-dimensional probability distribution over scores 1-10.
    The weighted mean of the distribution is the aesthetic score.

Reference:
    Talebi & Milanfar, "NIMA: Neural Image Assessment", IEEE TIP 2018.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AestheticScore:
    """Result of NIMA scoring on a single frame."""
    mean_score: float          # weighted mean of the distribution (1.0 - 10.0)
    std_score: float           # standard deviation — measures confidence
    distribution: list[float]  # 10 probabilities (bins 1-10)

    @property
    def normalized(self) -> float:
        """Score normalized to 0.0 - 1.0 range for compatibility with framing_quality."""
        return (self.mean_score - 1.0) / 9.0

    @property
    def rating(self) -> str:
        """Human-readable rating category."""
        if self.mean_score >= 7.0:
            return "excellent"
        elif self.mean_score >= 5.5:
            return "good"
        elif self.mean_score >= 4.0:
            return "average"
        else:
            return "poor"


class NimaModel(nn.Module):
    """
    NIMA model using MobileNetV2 as the feature extractor.

    MobileNetV2 is chosen over VGG16/InceptionV2 because:
    - Much faster inference (important for real-time pipeline)
    - Small memory footprint
    - Good enough accuracy for aesthetic scoring
    """

    def __init__(self) -> None:
        super().__init__()

        # Load MobileNetV2 pretrained on ImageNet
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Use all layers except the final classifier
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # NIMA head: dropout + linear → 10 score bins
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NimaScorer:
    """
    High-level scorer that handles preprocessing, inference, and score computation.

    Usage:
        scorer = NimaScorer(weights_path="weights/nima_mobilenetv2.pth")
        score = scorer.score_frame(frame)
        print(f"Aesthetic score: {score.mean_score:.2f}/10 ({score.rating})")
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        input_size: int = 224,
        device: Optional[str] = None,
    ) -> None:
        """
        weights_path: Path to pretrained NIMA weights (.pth file).
                     If None, uses ImageNet-pretrained backbone only (baseline scores).
        input_size: Image resize dimension (NIMA paper uses 224).
        device: "cuda", "cpu", or None for auto-detect.
        """
        self.input_size = input_size

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Build model
        self.model = NimaModel()

        # Load NIMA-specific weights if available
        if weights_path and os.path.exists(weights_path):
            logger.info(f"Loading NIMA weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self._weights_loaded = True
        else:
            if weights_path:
                logger.warning(
                    f"NIMA weights not found at '{weights_path}'. "
                    f"Using ImageNet backbone only — scores will be approximate. "
                    f"Run 'python scoring/download_weights.py' to fetch pretrained weights."
                )
            else:
                logger.info("No NIMA weights specified. Using ImageNet backbone baseline.")
            self._weights_loaded = False

        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization (MobileNetV2 expects this)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        logger.info(
            f"NIMA scorer ready on {self.device}. "
            f"Weights loaded: {self._weights_loaded}"
        )

    def score_frame(self, frame: cv2.typing.MatLike) -> AestheticScore:
        """
        Compute the NIMA aesthetic score for a single BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array).

        Returns:
            AestheticScore with mean, std, distribution, and rating.
        """
        # Convert BGR (OpenCV) → RGB (torchvision expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        # Inference (no gradient computation needed)
        with torch.no_grad():
            distribution = self.model(input_tensor)

        # Convert to numpy for score computation
        dist_np = distribution.cpu().numpy()[0]

        # Weighted mean: sum(p_i * i) for i in 1..10
        scores = np.arange(1, 11, dtype=np.float64)
        mean_score = float(np.sum(dist_np * scores))

        # Standard deviation of the distribution
        std_score = float(np.sqrt(np.sum(dist_np * (scores - mean_score) ** 2)))

        return AestheticScore(
            mean_score=mean_score,
            std_score=std_score,
            distribution=dist_np.tolist()
        )

    @property
    def has_trained_weights(self) -> bool:
        """Whether proper NIMA-trained weights were loaded."""
        return self._weights_loaded
