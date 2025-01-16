from modelscope import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat-Int4",
    trust_remote_code=True
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-VL-Chat-Int4",
    device_map="auto",
    trust_remote_code=True,
    fp16=True  # Add this since we trained with fp16
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model,
    "output_qlora/checkpoint-6",  # Let's try loading a specific checkpoint
    device_map="auto",
    torch_dtype=torch.float16
)

# Don't merge weights for Qwen
model.eval()

print("Model loaded. Testing...")
image_path = '/content/Zeus/training_images/Image_1.png'

# Exactly match the training format
prompt = f'''You are a GUI automation assistant. You will receive a screenshot with a coordinate grid overlay and instructions. 
Return JSON following this schema:
{{
    "actions": [
        {{"action": "mouse_move", "x": 100, "y": 200}},
        {{"action": "mouse_click"}},
        {{"action": "type_text", "text": "Hello, World!"}},
        {{"action": "keydown", "key": "ctrl"}},
        {{"action": "press_key", "key": "c"}},
        {{"action": "keyup", "key": "ctrl"}}
    ]
}}
Reference the screenshot grid for coordinates.
Be precise with coordinates based on the grid overlay.
Refer to the screenshot for exact coordinates, and make judgements based off of the grid reference.
The coordinates and relative lines are black and white for clear visibility, look at them and identify what you are trying to locate, and use the lines to determine the coordinates.
The user is on a MacOS computer with a single screen.

Picture 1: <img>{image_path}</img>
User Instruction: Open my notion app in the bottom panel'''

print("\nGenerating response...")
response, history = model.chat(tokenizer, query=prompt, history=None)
print("\nResponse:", response)