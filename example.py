import torch
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from reranchor_lib import Qwen2_5_RerAnchor, ReranchorProcessor, denoise_screenshot


model = ColQwen2_5.from_pretrained(
    "vidore/colqwen2.5-v0.2",
    device_map="cuda:0"
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    "vidore/colqwen2.5-v0.2"
)

rerank_model = Qwen2_5_RerAnchor.from_pretrained(
    "ricky42613/reranchor-qwen2.5-3b",
    device_map="cuda:0",
    torch_dtype=torch.bfloat16
).eval()

rerank_processor = ReranchorProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct"
)


query = "<your query here>"  # Replace with your actual query
image = Image.open("<path to your image>")  # Replace with the path to your actual image


query_inputs = processor.process_queries([query]).to(model.device)
with torch.no_grad():
    query_embed = model(**query_inputs)

denoised_image = denoise_screenshot(
    rerank_processor,
    rerank_model,
    query,
    image,
    k_tokens=200
)


image_inputs = processor.process_images([denoised_image]).to(model.device)
with torch.no_grad():
    image_embed = model(**image_inputs)

# ----------------------
# Step 4: similarity score
# ----------------------
score = processor.score_multi_vector(query_embed, image_embed)

print("Similarity score:", score.item())


