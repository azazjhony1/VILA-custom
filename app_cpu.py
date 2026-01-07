import os
os.environ['SKIP_PS3'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from PIL import Image

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

model_path = "Efficient-Large-Model/VILA1.5-3b"
image_file = "/home/azaz/vlmProfiling/VILA/traffic.jpg"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cpu",
    device_map={"": "cpu"},
    torch_dtype=torch.float32,
    attn_implementation="eager"
)

print(f"Model device: {next(model.parameters()).device}")
print(f"Model dtype: {next(model.parameters()).dtype}")

for param in model.parameters():
    if param.dtype == torch.float16:
        param.data = param.data.cpu().float()

for name, buffer in model.named_buffers():
    if buffer.dtype == torch.float16:
        buffer.data = buffer.data.cpu().float()

image = Image.open(image_file).convert('RGB')
image_tensor = process_images([image], image_processor, model.config).to('cpu', dtype=torch.float32)

prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cpu')

media_dict = {"image": [image_tensor[0]]}
media_config_dict = {"image": {"block_sizes": None}}

print("Starting inference on CPU (this will be slow)...")
print(f"Input device: {input_ids.device}")
print(f"Image tensor device: {image_tensor.device}")

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        media=media_dict,
        media_config=media_config_dict,
        max_new_tokens=512,
        do_sample=False
    )

output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print(output_text)
