from openai import OpenAI
from pydantic import BaseModel
import os
import base64

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

folder_path = "./output_test_smb/test_set_new"

basic = "The following images are frames from a Mario platformer. Describe the overall frames in detail, noting the environment style, environment layout, environment dynamics, and any character-NPC interactions and character-environment object interactions. Write your response in a detailed but succinct paragraph."
cot = "The following images are frames from a Mario platformer. Analyze the frames step by step, (1) Describe the background and art style, (2) list key objects and terrain and their spatial location and layout on screen, (3) note any motion or changes (or the lack of), (4) describe how the character interacts with the enemies and environment objects. What action is occurring? Finally, compile this information into a descriptive paragraph without removing any detail or information."
structured = "The following images are frames from a Mario platformer. Describe the over frames in detail, specifically on the following aspects: (a) Environment style (e.g. art style, atmosphere), (b) Environment layout (key platforms, terrain, objects visible, hero, etc.) and their spatial locations, (c) Dynamics (any movement or action ongoing), and (d) Character-environment interactions (how characters or NPCs are interacting with objects or each other). Answer in complete sentences."
role_system = "You are a game analyst describing scenes for a walkthrough. You should give a detailed, enumerative description covering all aspects of the scene. Base your explanation on truth and not predictive ideas."


class StructuredOutput(BaseModel):
    environment_style: str
    environment_layout: str
    environment_dynamics: str
    character_environment_interactions: str


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Create the base content with the text prompt.
message_content = [{"type": "text", "text": cot}]

# Iterate over each image file in the folder and append an image block for each.
for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png")):
        image_path = os.path.join(folder_path, filename)
        base64_image = encode_image(image_path)
        message_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "low",
                },
            }
        )

# response = client.chat.completions.create(
#     model="gpt-4o-2024-11-20",
#     max_tokens=1000,
#     messages=[{"role": "user", "content": message_content}],
# )

# For Structured
# response = client.beta.chat.completions.parse(
#     model="gpt-4o-2024-11-20",
#     max_tokens=1000,
#     messages=[{"role": "user", "content": message_content}],
#     response_format=StructuredOutput,
# )

# For Roles
response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    max_tokens=1000,
    messages=[
        {"role": "developer", "content": role_system},
        {"role": "user", "content": message_content},
    ],
)


output = response.choices[0].message.content
tokens_used = response.usage.total_tokens
finish_reason = response.choices[0].finish_reason

print("Response:")
print(output)


def get_unique_filename(filename):
    """Generate a unique filename by appending _1, _2, etc., if the file already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    return new_filename


filename = get_unique_filename("responses/GPT_response.txt")


with open(filename, "w") as f:
    f.write(
        f"Output:\n{output}\n{'=' * 50}\nTokens_used: {tokens_used}\n{'=' * 50}\nFinish reason: {finish_reason}"
    )
    print(f"Output file: {filename}")
