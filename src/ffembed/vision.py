"""Local image embeddings via DINOv3 backbones from Hugging Face Transformers.

This is kept in its own module so the rest of ffembed can stay free of the
transformers/torch dependency. Image support is only triggered when an image
file is indexed or an image is used as a query.
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path

from .paths import CACHE_DIR, ensure_root

VISION_GARDEN = {
    "dinov3-tiny": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",     # ~29M params, fastest
    "dinov3-small": "facebook/dinov3-convnext-small-pretrain-lvd1689m",   # ~50M params
    "dinov3-base": "facebook/dinov3-convnext-base-pretrain-lvd1689m",     # ~89M params
    "dinov3-vits": "facebook/dinov3-vits16-pretrain-lvd1689m",            # ~21M ViT-S
    "dinov3-vitb": "facebook/dinov3-vitb16-pretrain-lvd1689m",            # ~86M ViT-B
    "dinov3-vitl": "facebook/dinov3-vitl16-pretrain-lvd1689m",            # ~300M ViT-L
}

DEFAULT_VISION_MODEL = "dinov3-tiny"

_IMAGE_DEPS_MISSING = (
    "Image embeddings require 'transformers' and 'torch'. "
    "Install with: uv pip install 'ffembed[image]'"
)


def resolve_vision_model_name(alias: str) -> str:
    return VISION_GARDEN.get(alias, alias)


def _check_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(_IMAGE_DEPS_MISSING) from exc


@lru_cache(maxsize=2)
def _loaded_vision(model_name: str):
    _check_deps()
    from transformers import AutoImageProcessor, AutoModel

    ensure_root()
    hf_name = resolve_vision_model_name(model_name)
    processor = AutoImageProcessor.from_pretrained(hf_name, cache_dir=str(CACHE_DIR))
    model = AutoModel.from_pretrained(hf_name, cache_dir=str(CACHE_DIR))
    return processor, model


def _embed_raw(processor, model, images: list):
    import torch

    inputs = processor(images=images, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**inputs)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        pooled = outputs.last_hidden_state.mean(dim=1)
    return pooled


def embed_images(model_name: str, paths: list[str | Path]) -> list[list[float]]:
    """Embed one or more images with the given DINOv3 model."""
    from PIL import Image

    if not paths:
        return []

    processor, model = _loaded_vision(model_name)
    images = []
    for path in paths:
        data = Path(path).read_bytes()
        images.append(Image.open(BytesIO(data)).convert("RGB"))

    pooled = _embed_raw(processor, model, images)
    vectors = []
    for i in range(pooled.shape[0]):
        vec = pooled[i]
        if vec.dim() > 1:
            vec = vec.squeeze()
        vectors.append(vec.tolist())
    return vectors


def embed_image(model_name: str, path: str | Path) -> list[float]:
    """Embed a single image."""
    return embed_images(model_name, [path])[0]


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    }
