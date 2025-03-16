from openai import OpenAI
import os
import base64

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt = (
    "what's happening through the frames of this gameplay. "
    "Highlight the environment style, the environment layout - What elements are included in the environment. "
    "The environment dynamics - Overall environmental movements and movements of specific elements. "
    "Character-environment interactions - Includes character-NPC/enemy interactions and character-environment object interactions."
)
folder_path = "./output_test_smb/test_00001"


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# Create the base content with the text prompt.
message_content = [{"type": "text", "text": prompt}]

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

response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    max_tokens=1000,
    messages=[{"role": "user", "content": message_content}],
)

output = response.choices[0].message.content
tokens_used = response.usage.total_tokens
finish_reason = response.choices[0].finish_reason

print("Response:")
print(output)

# Optionally, write the response to a file.
with open("response.txt", "w") as f:
    f.write(
        f"Output:\n{output}\n{'=' * 50}\nTokens_used: {tokens_used}\n{'=' * 50}\nFinish reason: {finish_reason}"
    )
