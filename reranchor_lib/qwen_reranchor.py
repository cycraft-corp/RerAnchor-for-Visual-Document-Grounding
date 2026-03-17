import torch
from torch import nn
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLModel
from transformers.models.qwen2_vl import Qwen2VLConfig, Qwen2VLModel 
import os

class Qwen2_5_RerAnchor(Qwen2_5_VLModel):

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config=config)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.padding_side = "left"
        self.post_init()

    def save_pretrained(self, save_directory: str, **kwargs):
        """Override save_pretrained to include custom layers"""
        # Save the base model
        super().save_pretrained(save_directory, **kwargs)
        
        # Save custom layers separately
        custom_state = {
            'classifier': self.classifier.state_dict(),
            'dropout': self.dropout.state_dict()
        }
        torch.save(custom_state, os.path.join(save_directory, 'classifier_head.pt'))

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Override to load custom layers"""
        # Load base model
        model = super().from_pretrained(model_path, **kwargs)
        
        # Load custom layers if they exist
        classifier_path = os.path.join(model_path, 'classifier_head.pt')
        if os.path.exists(classifier_path):
            custom_state = torch.load(classifier_path, map_location='cpu')
            model.classifier.load_state_dict(custom_state['classifier'])
            model.dropout.load_state_dict(custom_state['dropout'])
    
        return model
    
    @classmethod
    def from_pretrained_with_lora(cls, model_path: str, lora_config: dict = None, **kwargs):
        """Load model and apply LoRA"""
        # Load base model
        model = cls.from_pretrained(model_path, **kwargs)
        
        # Default LoRA config
        default_config = {
            "r": 256,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05
        }
        if lora_config:
            default_config.update(lora_config)
        
        peft_config = LoraConfig(**default_config)
        
        # Wrap with PEFT
        model = get_peft_model(model, peft_config)
        
        # Ensure classifier remains trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.dropout.parameters():
            param.requires_grad = True
            
        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        ) 
        # Pools only the image embeddings
        # image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
        # last_hidden_states = last_hidden_states * image_mask
        sequence_output = self.dropout(last_hidden_states)
        logits = self.classifier(sequence_output)  # (num_unmasked_tokens, num_labels)
        return logits

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size


from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn


class Qwen2_RerAnchor(Qwen2VLModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config=config)
        self.classifier = nn.Linear(self.config.hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.padding_side = "left"
        self.post_init()

    @classmethod
    def from_pretrained(cls, *args, key_mapping=None, **kwargs):
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping
        return super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
    
    @classmethod
    def from_pretrained_with_lora(cls, model_path: str, lora_config: dict = None, **kwargs):
        """Load model and apply LoRA"""
        # Load base model
        model = cls.from_pretrained(model_path, **kwargs)
        
        # Default LoRA config
        default_config = {
            "r": 256,
            "lora_alpha": 32,
            "target_modules": "all-linear",
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": TaskType.TOKEN_CLS,
        }
        if lora_config:
            default_config.update(lora_config)
        
        peft_config = LoraConfig(**default_config)
        
        # Wrap with PEFT
        model = get_peft_model(model, peft_config)
        
        # Ensure classifier remains trainable
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.dropout.parameters():
            param.requires_grad = True
            
        return model

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        kwargs.pop("return_dict", True)
        kwargs.pop("output_hidden_states", None)
        kwargs.pop("use_cache", None)
        last_hidden_states = (
            super()
            .forward(*args, **kwargs, use_cache=False, output_hidden_states=True, return_dict=True)
            .last_hidden_state
        ) 
        # Pools only the image embeddings
        image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
        last_hidden_states = last_hidden_states * image_mask
        sequence_output = self.dropout(last_hidden_states)
        logits = self.classifier(sequence_output)
        return logits

    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size