import base64
import json
import time
from datetime import datetime
from openai import OpenAI
from llm_config import AVAILABLE_COMMANDS, WAYPOINT_CONSTRAINTS, SYSTEM_PROMPT

class LLMService:
    def __init__(self, api_key, output_log_file="llm_logs.jsonl", output_json_file="llm_output.json"):
        self.api_key = ""
        self.output_log_file = output_log_file
        self.output_json_file = output_json_file
        self.client = OpenAI(api_key="")
        self.available_commands = AVAILABLE_COMMANDS
        self.waypoint_constraints = WAYPOINT_CONSTRAINTS

    def encode_image_base64(self, image_array):
        """Convert numpy array to base64 string"""
        import cv2
        import numpy as np
        success, buffer = cv2.imencode('.png', image_array)
        if not success:
            raise Exception("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")

    def build_prompt(self, instruction):
        system_prompt = SYSTEM_PROMPT.format(
            commands=", ".join(self.available_commands),
            **self.waypoint_constraints
        )
        
        return f"""
<system>{system_prompt}</system>

<user>User command: <instruction>{instruction}</instruction></user>

<output>Return response as JSON with:
- "message": brief natural language explanation of the result
- "waypoints": list of 2D waypoints (x, y) in meters relative to the car (optional if no movement)
- "actions": list of high-level symbolic steps (must be from the available commands list)</output>
""".strip()

    def query_llm(self, image_array, instruction):
        """Query the LLM with an image array and instruction"""
        image_b64 = self.encode_image_base64(image_array)
        prompt_text = self.build_prompt(instruction)

        response = self.client.chat.completions.create(
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

    def extract_json_from_response(self, raw_response):
        try:
            json_start = raw_response.find("{")
            json_str = raw_response[json_start:]
            parsed = json.loads(json_str)
            return parsed
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            print("Raw output was:\n", raw_response)
            return None

    def log_output(self, timestamp, instruction, response, parsed_json):
        with open(self.output_log_file, "a") as log:
            log.write(json.dumps({
                "timestamp": timestamp,
                "instruction": instruction,
                "raw_response": response,
                "parsed_output": parsed_json
            }) + "\n")

    def process_image(self, image_array, instruction):
        """Main processing pipeline for a single image"""
        try:
            raw_response = self.query_llm(image_array, instruction)
            parsed_output = self.extract_json_from_response(raw_response)
            
            if parsed_output:
                # Save last output
                with open(self.output_json_file, "w") as f:
                    json.dump(parsed_output, f, indent=2)

                # Log to JSONL history
                timestamp = datetime.now().isoformat()
                self.log_output(timestamp, instruction, raw_response, parsed_output)
                
                return parsed_output
            return None
        except Exception as e:
            print(f"[ERROR] Failed to process image: {e}")
            return None 