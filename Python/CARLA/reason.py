import base64
from openai import OpenAI

client = OpenAI(api_key="")

# Load and encode the image
def encode_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_path = "image.png"
base64_image = encode_image_base64(image_path)

# Send multimodal message
response = client.chat.completions.create(
    model="gpt-4.1-2025-04-14",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Can I safely turn left based on this scene?"},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }}
            ]
        }
    ],
    max_tokens=500,
)

print("Response:")
print(response.choices[0].message.content)
