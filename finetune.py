from pathlib import Path
import os
from typing import cast, List, Dict
from datasets import Dataset, DatasetDict
from torch import nn
from transformers import TrainingArguments

from reranchor_lib import RerAnchorCollator, Qwen2_5_RerAnchor, RerAnchorTrainer, ReranchorProcessor
import json
from tqdm import tqdm
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_data_from_json(json_path: str, max_sample: int = -1, file_cnt: int = -1) -> List[Dict]:
    """
    Load data from JSON file in the format similar to your training script.
    
    Expected format:
    [
        {
            "image_path": "path/to/image.jpg",
            "query": ["query text"],
            "positive_region": [[x1, y1, x2, y2], ...] or [[poly coords], ...]
        },
        ...
    ]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if file_cnt > 0:
        data = data[:file_cnt]
    processed_data = []
    for item in tqdm(data):
        # Load image
        img_path = 'image_set/' + item["positive_page"]
        with Image.open(img_path) as image:   
            for q in item['query']:
                data = {
                    "path": img_path,
                    "image": image.copy(),
                    "query": q,
                    "negative_queries": item['negative_query'] if 'negative_query' in item else [],
                    "bbox": item['positive_region']  # List of bounding boxes or polygons
                }
                processed_data.append(data)
        if max_sample > 0 and len(processed_data) >= max_sample:
            break
    
    return processed_data

def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            print("requires_grad: ", name)

    trainable_percentage = 100 * trainable_params / all_param
    print(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {trainable_percentage}")


# ==========================     USER INPUT     ==========================

QUANTIZATION_STRATEGY = None # "4bit"

# ========================================================================

# Automatically set the device
device = "cuda:0"

# Pre-trained model name (with LoRA adapter)
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_RerAnchor.from_pretrained(
        model_name,
        device_map=device
    )
for name, param in model.named_parameters():
    param.requires_grad = True

print_trainable_parameters(model)

processor = cast(
    ReranchorProcessor,
    ReranchorProcessor.from_pretrained(model_name),
)
collator = RerAnchorCollator(processor=processor, spatial_merge_size=model.spatial_merge_size)

data_size = 50000

ds = load_data_from_json(f"training_set/train.json", max_sample=data_size, file_cnt=-1)
full_dataset = Dataset.from_list(ds)
# Step 2: Split into train and test
split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42, shuffle=False)

# Step 3: Wrap in DatasetDict
ds = DatasetDict({
    "train": split_dataset["train"],
    "test": split_dataset["test"]
})

print("num. of train data: ", len(ds['train']))
print("num. of test data: ", len(ds['test']))


checkpoints_dir = Path(f"reranchor_checkpoints")
checkpoints_dir.mkdir(exist_ok=True, parents=True)

train_ds = ds["train"]
eval_ds = ds["test"]

training_args = TrainingArguments(
    output_dir=str(checkpoints_dir),
    hub_model_id=None,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    eval_strategy="steps" if eval_ds is not None else "no",
    save_steps=5000,
    eval_steps=5000,
    warmup_steps=20,
    learning_rate=2e-5,
    save_total_limit=3,
    report_to=[],
    ddp_find_unused_parameters=False,
    logging_strategy="steps",
    logging_steps=1,
)


trainer = RerAnchorTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=training_args,
    data_collator=collator,
    processor=processor
)

trainer.args.remove_unused_columns = False

train_results = trainer.train()
trainer.save_model("reranchor")

if eval_ds is not None:
    eval_results = trainer.evaluate()
    print(f"eval_results: {eval_results}")
