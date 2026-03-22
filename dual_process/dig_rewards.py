"""Differentiable reward models for image-generation quality evaluation.

Each reward model accepts a raw image tensor and (optionally) a text prompt,
and returns a **(loss, meta)** pair where ``loss`` is the *negative* reward
(minimise loss → maximise reward) and ``meta`` is a plain dict of diagnostic
scalars.

Supported models
----------------
* ``CLIPReward``      — text–image alignment via CLIP cosine similarity.
* ``PickScoreReward`` — human-preference score (PickScore, NeurIPS 2023).
* ``AestheticReward`` — LAION aesthetic predictor V2 (CLIP + MLP head).

Factory
-------
Use :func:`load_reward_model` to instantiate a model by type name.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ImageNet-style CLIP normalization constants
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _normalize_for_clip(image_tensor: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Rescale *image_tensor* to [0, 1] then apply CLIP normalisation.

    Args:
        image_tensor: ``[B, C, H, W]`` tensor in any numeric range.
        size: Target spatial dimension after bilinear resize.

    Returns:
        CLIP-normalised tensor of shape ``[B, C, size, size]``.
    """
    # Rescale to [0, 1] per-image
    flat = image_tensor.flatten(2)  # [B, C, H*W]
    min_v = flat.min(dim=-1).values[..., None, None]
    max_v = flat.max(dim=-1).values[..., None, None]
    image_tensor = (image_tensor - min_v) / (max_v - min_v + 1e-8)
    # Resize
    image_tensor = F.interpolate(
        image_tensor, size=(size, size), mode="bilinear", align_corners=False
    )
    # CLIP normalisation
    mean = torch.tensor(_CLIP_MEAN, device=image_tensor.device, dtype=image_tensor.dtype)
    std = torch.tensor(_CLIP_STD, device=image_tensor.device, dtype=image_tensor.dtype)
    image_tensor = (image_tensor - mean[None, :, None, None]) / std[None, :, None, None]
    return image_tensor


class CLIPReward(nn.Module):
    """Text–image alignment reward using CLIP cosine similarity.

    Uses ``openai/clip-vit-large-patch14`` to measure how well a generated
    image matches the target prompt.  Loss = −cosine_similarity.

    Args:
        model_id: HuggingFace model identifier for the CLIP model.
        device: Torch device string.
    """

    DEFAULT_MODEL_ID = "openai/clip-vit-large-patch14"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cuda"):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.clip = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.device = device
        self.clip = self.clip.to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
        logger.info("Loaded CLIPReward from %s on %s", model_id, device)

    def forward(
        self, image_tensor: torch.Tensor, text: str
    ) -> tuple[torch.Tensor, dict]:
        """Compute CLIP alignment loss.

        Args:
            image_tensor: ``[B, C, H, W]`` tensor in any range.
            text: Generation prompt (or list of prompts with length B).

        Returns:
            Tuple of ``(loss, meta)`` where ``loss`` is the negative mean
            cosine similarity and ``meta["clip_score"]`` is its detached value.
        """
        pixel_values = _normalize_for_clip(image_tensor).to(
            self.device, self.clip.dtype
        )
        texts = [text] if isinstance(text, str) else text
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        txt_feat = self.clip.get_text_features(**text_inputs)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        similarity = (img_feat * txt_feat).sum(dim=-1)

        loss = -similarity.mean()
        return loss, {"clip_score": similarity.mean().detach().item()}


class PickScoreReward(nn.Module):
    """Human-preference reward using PickScore.

    PickScore is a CLIP-based model fine-tuned on the PickaPic dataset to
    predict human preference between generated images.

    Reference:
        Kirstain et al., "Pick-a-Pic: An Open Dataset of User Preferences
        for Text-to-Image Generation", NeurIPS 2023.

    Args:
        device: Torch device string.
    """

    _MODEL_ID = "yuvalkirstain/PickScore_v1"
    _PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self._PROCESSOR_ID)
        self.model = AutoModel.from_pretrained(self._MODEL_ID)
        self.device = device
        self.model = self.model.to(device)
        for p in self.model.parameters():
            p.requires_grad = False
        logger.info("Loaded PickScoreReward from %s on %s", self._MODEL_ID, device)

    def forward(
        self, image_tensor: torch.Tensor, text: str
    ) -> tuple[torch.Tensor, dict]:
        """Compute PickScore alignment loss.

        Args:
            image_tensor: ``[B, C, H, W]`` tensor in any range.
            text: Generation prompt (or list of prompts with length B).

        Returns:
            Tuple of ``(loss, meta)`` where ``loss`` is the negative mean
            PickScore and ``meta["pick_score"]`` is its detached value.
        """
        pixel_values = _normalize_for_clip(image_tensor, size=224).to(
            self.device, self.model.dtype
        )
        texts = [text] if isinstance(text, str) else text
        text_inputs = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        img_feat = self.model.get_image_features(pixel_values=pixel_values)
        txt_feat = self.model.get_text_features(**text_inputs)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        similarity = (img_feat * txt_feat).sum(dim=-1)

        loss = -similarity.mean()
        return loss, {"pick_score": similarity.mean().detach().item()}


class _AestheticMLP(nn.Module):
    """MLP head matching the LAION improved aesthetic predictor architecture."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AestheticReward(nn.Module):
    """Aesthetic quality reward using the LAION Aesthetics Predictor V2.

    Computes CLIP ViT-L/14 image features, then passes them through a
    pre-trained MLP that predicts an aesthetic quality score.

    Reference:
        Schuhmann et al., improved-aesthetic-predictor (2022).
        Weights from ``shunk031/aesthetics-predictor`` on HuggingFace Hub.

    Args:
        device: Torch device string.
    """

    _CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
    _WEIGHTS_REPO = "shunk031/aesthetics-predictor"
    _WEIGHTS_FILE = "sac+logos+ava1-l14-linearMSE.pth"

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from huggingface_hub import hf_hub_download
        from transformers import CLIPModel, CLIPProcessor

        self.clip = CLIPModel.from_pretrained(self._CLIP_MODEL_ID)
        self.processor = CLIPProcessor.from_pretrained(self._CLIP_MODEL_ID)
        self.predictor = _AestheticMLP(input_size=768)
        weights_path = hf_hub_download(
            repo_id=self._WEIGHTS_REPO, filename=self._WEIGHTS_FILE
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        self.predictor.load_state_dict(state_dict)
        self.device = device
        self.clip = self.clip.to(device)
        self.predictor = self.predictor.to(device)
        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.predictor.parameters():
            p.requires_grad = False
        logger.info(
            "Loaded AestheticReward (%s / %s) on %s",
            self._WEIGHTS_REPO,
            self._WEIGHTS_FILE,
            device,
        )

    def forward(
        self, image_tensor: torch.Tensor, text: str = ""
    ) -> tuple[torch.Tensor, dict]:
        """Compute aesthetic quality loss.

        Args:
            image_tensor: ``[B, C, H, W]`` tensor in any range.
            text: Unused; kept for API consistency with other reward models.

        Returns:
            Tuple of ``(loss, meta)`` where ``loss`` is the negative mean
            aesthetic score and ``meta["aesthetic_score"]`` is its detached value.
        """
        pixel_values = _normalize_for_clip(image_tensor).to(
            self.device, self.clip.dtype
        )
        img_feat = self.clip.get_image_features(pixel_values=pixel_values)
        img_feat = F.normalize(img_feat, dim=-1).float()
        score = self.predictor(img_feat).squeeze(-1)

        loss = -score.mean()
        return loss, {"aesthetic_score": score.mean().detach().item()}


_REWARD_REGISTRY: dict[str, type] = {
    "clip": CLIPReward,
    "pickscore": PickScoreReward,
    "aesthetic": AestheticReward,
}


def load_reward_model(model_type: str, device: str = "cuda", **kwargs) -> nn.Module:
    """Instantiate a reward model by type name.

    Args:
        model_type: One of ``"clip"``, ``"pickscore"``, ``"aesthetic"``.
        device: Target device string (e.g. ``"cuda:0"``).
        **kwargs: Additional keyword arguments forwarded to the model constructor.

    Returns:
        Reward model instance with all parameters frozen.

    Raises:
        ValueError: If *model_type* is not in the registry.
    """
    if model_type not in _REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward model type '{model_type}'. "
            f"Available: {sorted(_REWARD_REGISTRY)}"
        )
    return _REWARD_REGISTRY[model_type](device=device, **kwargs)
