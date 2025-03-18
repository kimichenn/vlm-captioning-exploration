from openai import OpenAI
import os
import base64

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

prompt = (
    "The following frames show a classic NES-style platformer."
    "Describe the scene's visual environment style, as well as what is happening."
    "Look at all frames and explain the level design - what platforms, hazards, and collectibles are visible and how they're arranged"
    "Describe what is happening across these frames how things move or change (e.g., Mario's actions and any moving elements)."
    "Also mention interactions between the hero and any enemies, items, and the environment."
    "Summarize the following in a succinct, precise, and detailed paragraph."
)
folder_path = "./output_test_smb/test_00002"


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
    max_tokens=500,
    messages=[{"role": "user", "content": message_content}],
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


# Example usage:
filename = get_unique_filename("responses/response.txt")


with open(filename, "w") as f:
    f.write(
        f"Output:\n{output}\n{'=' * 50}\nTokens_used: {tokens_used}\n{'=' * 50}\nFinish reason: {finish_reason}"
    )
    print(f"Output file: {filename}")
