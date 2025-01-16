from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch

print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen-VL-Chat-Int4",
    trust_remote_code=True
)

print("Loading fine-tuned model...")
model = AutoModelForCausalLM.from_pretrained(
    "/content/Zeus/output_qlora",  # Path to your fine-tuned adapter
    device_map="auto",
    trust_remote_code=True
).eval()

# Test the model
print("Testing model...")
image_path = '/content/Zeus/training_images/Image_1.png'
response, history = model.chat(
    tokenizer, 
    query=f'''<img>{image_path}</img>
    You are a GUI automation assistant. You will receive a screenshot with a coordinate grid overlay and instructions. 
                Return JSON following the schema you have been trained on. 
                Reference the screenshot grid for coordinates.
                Be precise with coordinates based on the grid overlay.
                Refer to the screenshot for exact coordinates, and make judgements based off of the grid reference.
                The coordinates and relative lines are black and white for clear visibility, look at them and identify what you are trying to locate, and use the lines to determine the coordinates. 
                The user is on a MacOS computer with a single screen.

                User Instruction: 'Open my notion app in the bottom panel'
    ''', 
    history=None
)
print("Response:", response)