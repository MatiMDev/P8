import base64
import json
import time
from datetime import datetime
from openai import OpenAI

# === CONFIGURATION ===
API_KEY = ""
IMAGE_PATH = "image.png"
INSTRUCTION = "Find a safe parking spot and stop when possible."
OUTPUT_LOG_FILE = "llm_logs.jsonl"
OUTPUT_JSON_FILE = "llm_output.json"

# === CLIENT ===
client = OpenAI(api_key=API_KEY)

# === IMAGE ENCODING ===
def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# === GENERATE PROMPT ===
def build_prompt(instruction):
    return f"""
<system>You are an assistant for an autonomous driving agent. Given a front-view camera image and a user instruction, return a structured navigation plan.</system>

<user>User command: <instruction>{instruction}</instruction></user>

<output>Return response as JSON with:
- "message": brief natural language explanation of the result
- "waypoints": list of 2D waypoints (x, y) in meters relative to the car (optional if no movement)
- "actions": list of high-level symbolic steps (e.g., ["slow_down", "turn_left", "stop"])</output>
""".strip()

# === CALL LLM ===
def query_llm(image_b64, instruction):
    prompt_text = build_prompt(instruction)

    response = client.chat.completions.create(
        model="gpt-4.1-2025-04-14",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    }}
                ]
            }
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return response.choices[0].message.content

# === PARSE JSON OUTPUT ===
def extract_json_from_response(raw_response):
    try:
        json_start = raw_response.find("{")
        json_str = raw_response[json_start:]
        parsed = json.loads(json_str)
        return parsed
    except Exception as e:
        print("[ERROR] Failed to parse JSON:", e)
        print("Raw output was:\n", raw_response)
        return None

# === LOGGING ===
def log_output(timestamp, instruction, response, parsed_json):
    with open(OUTPUT_LOG_FILE, "a") as log:
        log.write(json.dumps({
            "timestamp": timestamp,
            "instruction": instruction,
            "raw_response": response,
            "parsed_output": parsed_json
        }) + "\n")

# === MAIN PIPELINE ===
def main():
    print("[INFO] Encoding image...")
    image_b64 = encode_image_base64(IMAGE_PATH)

    print("[INFO] Sending prompt to GPT-4V...")
    raw_response = query_llm(image_b64, INSTRUCTION)
    print("[INFO] Raw response received.")

    print("\n=== Raw Response ===")
    print(raw_response)

    print("\n[INFO] Parsing structured JSON...")
    parsed_output = extract_json_from_response(raw_response)
    if not parsed_output:
        print("[ERROR] Could not extract structured response.")
        return

    print("\n=== Parsed JSON ===")
    print(json.dumps(parsed_output, indent=2))

    # Save last output
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump(parsed_output, f, indent=2)

    # Log to JSONL history
    timestamp = datetime.now().isoformat()
    log_output(timestamp, INSTRUCTION, raw_response, parsed_output)

    print(f"\n[INFO] Output saved to '{OUTPUT_JSON_FILE}' and logged to '{OUTPUT_LOG_FILE}'.")

# === ENTRY POINT ===
if __name__ == "__main__":
    main()
