from einops import rearrange
from torch.nn import Sigmoid
from typing import Tuple, Union
import numpy as np
from PIL import Image
import torch

def mask_similarity_topk(
    image: Image.Image,
    similarity_map: torch.Tensor,            # shape: (n_patches_x, n_patches_y)
    topk: Union[int, float],                 # int: number of patches; float in (0,1]: fraction of patches
    fill: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """
    Mask pixels belonging to the top-k highest-similarity patches and return the masked RGBA image.

    Args:
        image: PIL Image.
        similarity_map: Tensor with shape (n_patches_x, n_patches_y) of similarity scores.
        topk: If int >= 1, the number of highest-similarity patches to mask.
              If float in (0, 1], the fraction of patches to mask.
        fill: RGBA tuple to fill masked pixels (default: transparent).

    Returns:
        PIL.Image in RGBA mode with top-k patches masked out.
    """
    # Validate fill
    if len(fill) != 4:
        raise ValueError("`fill` must be an RGBA 4-tuple, e.g., (0, 0, 0, 0).")

    # Convert image to RGBA array
    img_rgba = image.convert("RGBA")
    img_array = np.array(img_rgba)  # (H, W, 4)

    # Get similarity as numpy
    sim = similarity_map.detach().to(torch.float32).cpu().numpy()  # (px, py)
    px, py = sim.shape
    total_patches = px * py

    # Determine k
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError("Float `topk` must be in (0, 1].")
        k = max(1, int(round(topk * total_patches)))
    else:
        k = int(topk)
        if k < 1:
            raise ValueError("Integer `topk` must be >= 1.")
    k = min(k, total_patches)

    # Build boolean patch mask for the top-k highest values
    flat = sim.ravel()
    # Indices of top-k largest elements (argpartition is O(n))
    topk_idx = np.argpartition(flat, -k)[-k:]
    patch_mask = np.ones_like(flat, dtype=bool)
    patch_mask[topk_idx] = False
    patch_mask = patch_mask.reshape(sim.shape)  # (px, py)

    # Reorient to PIL's (width, height) convention used earlier
    patch_mask_pil = rearrange(patch_mask, "h w -> w h")  # (py, px) -> (width, height in patch space)

    # Upscale patch mask to image size using NEAREST to preserve patch blocks
    mask_img = Image.fromarray((patch_mask_pil.astype(np.uint8) * 255), mode="L").resize(
        img_rgba.size, Image.Resampling.NEAREST
    )
    pixel_mask = np.array(mask_img) > 0  # (H, W) True where we want to mask

    # Apply mask
    out = img_array.copy()
    out[pixel_mask] = np.array(fill, dtype=out.dtype)

    return Image.fromarray(out, mode="RGBA")

def denoise_screenshot(rerank_processor, rerank_model, q, img, k_tokens):
    sigmoid_fn = Sigmoid()
    rerank_inputs = rerank_processor.process_texts_and_images([q], [img]).to(rerank_model.device)
    patch_weights = sigmoid_fn(rerank_model(**rerank_inputs))
    rerank_mask = rerank_inputs['input_ids'][0] == rerank_processor.image_token_id
    patch_weights = patch_weights[0][rerank_mask]
    n_patches = rerank_processor.get_n_patches(image_size=img.size, spatial_merge_size=rerank_model.spatial_merge_size)
    chunk_distribution = rearrange(
        patch_weights,
        "(h w) c -> w h c",
        w=n_patches[0],
        h=n_patches[1],
    )
    denoised_img = mask_similarity_topk(img, chunk_distribution.squeeze(-1), k_tokens, fill=(255,255,255,255))
    return denoised_img