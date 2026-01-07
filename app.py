import os
os.environ['SKIP_PS3'] = '1'

import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images
from PIL import Image

model_path = "Efficient-Large-Model/VILA1.5-3b"
image_file = "/home/azaz/vlmProfiling/VILA/traffic.jpg"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device_map="cuda:0"
)

image = Image.open(image_file).convert('RGB')
image_tensor = process_images([image], image_processor, model.config).to('cuda:0', dtype=torch.float16)

prompt = "USER: <image>\nDescribe this image in detail. ASSISTANT:"
input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda:0')

media_dict = {"image": [image_tensor[0]]}
media_config_dict = {"image": {"block_sizes": None}}

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
