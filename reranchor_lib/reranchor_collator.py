from typing import Any, Dict, List
from PIL.Image import Image
from .processing_reranchor import ReranchorProcessor
import torch
import numpy as np

def get_selected_tokens(image, bbox, n_patches):
    """
    Get binary token selection mask for bounding boxes.
    Returns tensor with 1 for selected tokens, 0 for others.
    """
    # Create token index matrix (n_patches_y, n_patches_x)
    total_patches = n_patches[0] * n_patches[1]
    token_indices = np.arange(0, total_patches).reshape(n_patches[1], n_patches[0])
    
    # Calculate patch dimensions
    patch_height = image.size[1] / n_patches[1]
    patch_width = image.size[0] / n_patches[0]
    
    # Create coordinate grids
    y_coords = np.arange(image.size[1])
    x_coords = np.arange(image.size[0])
    Y, X = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Find patch indices for each pixel
    patch_y_indices = np.clip((Y / patch_height).astype(int), 0, n_patches[1] - 1)
    patch_x_indices = np.clip((X / patch_width).astype(int), 0, n_patches[0] - 1)
    
    # Map to token indices
    pixel_matrix = token_indices[patch_y_indices, patch_x_indices]
    
    # Transpose to match PIL convention (width, height)
    pixel_matrix_pil = pixel_matrix.T
    
    selected_tokens = set()
    for i in range(len(bbox)):
        min_x, min_y = min(bbox[i][0], bbox[i][2]), min(bbox[i][1], bbox[i][3])
        max_x, max_y = max(bbox[i][0], bbox[i][2]), max(bbox[i][1], bbox[i][3])
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if x < pixel_matrix_pil.shape[0] and y < pixel_matrix_pil.shape[1]:
                    selected_tokens.add(pixel_matrix_pil[x][y])
    
    # Create binary mask for all tokens (including text tokens)
    ret = [0 for _ in range(n_patches[0]*n_patches[1])] 
    for i in range(len(ret)):
        if i in selected_tokens:
            ret[i] = 1
    
    return ret

def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """
    Prefix all keys in a dictionary with the given prefix.
    """
    return {f"{prefix}{k}": v for k, v in data.items()}

class RerAnchorCollator:
    """
    Collator for training vision retrieval models.
    """

    def __init__(
        self,
        processor: ReranchorProcessor,
        max_length: int = 2048,
        spatial_merge_size: int = 16,
        add_negative: bool = False
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None
        self.spatial_merge_size = spatial_merge_size
        self.add_negative = add_negative

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = []
        neg_texts = []
        images = []
        selected_tokens = []
        # Parse the examples.
        for example in examples:
            if not self.add_negative:
                texts.append(example["query"])
                images.append(example["image"])
                n_patches = self.processor.get_n_patches(
                    image_size=example['image'].size, 
                    spatial_merge_size=self.spatial_merge_size
                )
                selected = get_selected_tokens(example['image'], example['bbox'], n_patches)
                selected_tokens.append(selected)
            else:
                for neg_query in example["negative_queries"][:3]:
                    texts.append(example["query"])
                    neg_texts.append(neg_query)
                    images.append(example["image"])
                    n_patches = self.processor.get_n_patches(
                        image_size=example['image'].size, 
                        spatial_merge_size=self.spatial_merge_size
                    )
                    selected = get_selected_tokens(example['image'], example['bbox'], n_patches)
                    selected_tokens.append(selected)
                
        batch_inputs = self.auto_collate(batch_texts=texts, batch_images=images, prefix="pos_")
        batch_neg_inputs = self.auto_collate(batch_texts=neg_texts, batch_images=images, prefix="neg_") if self.add_negative else {}

        expand = []
        for i in range(len(batch_inputs['pos_input_ids'])):
            toks = batch_inputs['pos_input_ids'][i].tolist()
            vision_start_idx = toks.index(151652)
            vision_end_idx = toks.index(151653)
            expand.append([vision_start_idx, vision_end_idx])

        padded_selected_tokens = []
        for i, tokens in enumerate(selected_tokens):
            padded = [-100 for _ in range(expand[i][0]+1)] + tokens + [-100 for _ in range(expand[i][1], len(batch_inputs['pos_input_ids'][i]))]
            padded_selected_tokens.append(padded)
            assert len(padded_selected_tokens[-1]) == int(batch_inputs['pos_input_ids'].shape[-1])
        batch_inputs["selected_tokens"] = torch.tensor(padded_selected_tokens, dtype=torch.float)
        if self.add_negative:
            expand = []
            for i in range(len(batch_neg_inputs['neg_input_ids'])):
                toks = batch_neg_inputs['neg_input_ids'][i].tolist()
                vision_start_idx = toks.index(151652)
                vision_end_idx = toks.index(151653)
                expand.append([vision_start_idx, vision_end_idx])

            padded_selected_tokens = []
            for i, tokens in enumerate(selected_tokens):
                padded = [-100 for _ in range(expand[i][0]+1)] + tokens + [-100 for _ in range(expand[i][1], len(batch_neg_inputs['neg_input_ids'][i]))]
                padded_selected_tokens.append(padded)
                assert len(padded_selected_tokens[-1]) == int(batch_neg_inputs['neg_input_ids'].shape[-1])
            batch_neg_inputs["neg_selected_tokens"] = torch.tensor(padded_selected_tokens, dtype=torch.float)

        return {
            **batch_inputs,
            **batch_neg_inputs
        }

    def auto_collate(self, batch_texts: List[Image], batch_images: List[Image], prefix="") -> Dict[str, Any]:
        """Automatically collate a batch of documents."""
        proc_batch = self.processor.process_texts_and_images(texts=batch_texts, images=batch_images)
        return prefix_keys(proc_batch, prefix)
