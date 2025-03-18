from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import torch
import os
import base64


prompt = (
    "The following frames show a classic NES-style platformer."
    "Describe the scene's visual environment style, as well as what is happening."
    "Look at all frames and explain the level design - what platforms, hazards, and collectibles are visible and how they're arranged"
    "Describe what is happening across these frames how things move or change (e.g., the hero's actions and any moving elements)."
    "Also mention interactions between the hero and any enemies, items, and the environment."
)

folder_path = "./output_test_smb/test_00002"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256 * 28 * 28
# max_pixels = 1280 * 28 * 28
# processor = AutoProcessor.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
# )

message_content = [{"type": "text", "text": prompt}]

# Iterate over each image file in the folder and append an image block for each.
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png")):
        image_path = os.path.join(folder_path, filename)
        base64_image = encode_image(image_path)
        message_content.append(
            {"type": "image", "image": f"data:image/png;base64,{base64_image}"}
        )

messages = [{"role": "user", "content": message_content}]
# print(messages)

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=500)
generated_ids_trimmed = [
    out_ids[len(in_ids) :]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
