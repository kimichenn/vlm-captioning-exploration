from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
import os
import base64

basic = "The following images are frames from a 2D NES-styled side-scrolling platformer. Describe the overall frames in detail, noting the environment style, environment layout, environment dynamics, and any character-NPC interactions and character-environment object interactions. Write your response in a detailed but succinct paragraph."
cot = "The following images are frames from a 2D NES-styled side-scrolling platformer. Analyze the frames step by step, (1) Describe the background and art style, (2) list key objects and terrain and their spatial location and layout on screen, (3) note any motion or changes (or the lack of), (4) describe how the character interacts with the enemies and environment objects. What action is occurring? Finally, compile this information into a descriptive paragraph without removing any detail or information."
structured = "The following images are frames from a 2D NES-styled side-scrolling platformer. Describe the over frames in detail, specifically on the following aspects: (a) Environment style (e.g. art style, atmosphere), (b) Environment layout (key platforms, terrain, objects visible, hero, etc.) and their spatial locations, (c) Dynamics (any movement or action ongoing), and (d) Character-environment interactions (how characters or NPCs are interacting with objects or each other). Answer in complete sentences."
role_system = "You are a game analyst describing scenes for a walkthrough. You should give a detailed, enumerative description covering all aspects of the scene. Base your explanation on truth and not predictive ideas."

folder_path = "./output_test_smb/test_00001"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "remyxai/SpaceQwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(
    "remyxai/SpaceQwen2.5-VL-3B-Instruct"
)

message_content = [{"type": "text", "text": cot}]

# Iterate over each image file in the folder and append an image block for each.
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png")):
        image_path = os.path.join(folder_path, filename)
        base64_image = encode_image(image_path)
        message_content.append(
            {"type": "image", "image": f"data:image/png;base64,{base64_image}"}
        )

messages = [{"role": "user", "content": message_content}]

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

# Inference: Generation of the output
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
print(output_text[0])
