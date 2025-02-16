import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import gradio as gr
from transformers import GPT2Tokenizer
from model import SmolLM2Config, SmolLM2ForCausalLM

tokenizer = GPT2Tokenizer.from_pretrained(
    "HuggingFaceTB/cosmo2-tokenizer"
)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


# Load the model
def load_model():
    config = SmolLM2Config()
    model = SmolLM2ForCausalLM(config)  # Create base model instead of Lightning model

    # Load just the model weights
    state_dict = torch.load("model_weights.pth", map_location="cpu")['model_state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    return model

# Load the model globally
model = load_model()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("smollm2_135.log"),  # Logs will be saved to 'app2.log'
        logging.StreamHandler()          # Logs will also appear in the console
    ],
)

app = FastAPI()

history = []

@app.post("/generate_text")
async def generate_text(prompt, max_tokens, temperature=0.8, top_k=40):
    """Generate text based on the prompt"""
    try:
        # Encode the prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Move to device if needed
        device = next(model.parameters()).device
        prompt_ids = prompt_ids.to(device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(  # Call generate directly on base model
                prompt_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0].tolist())
        history.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_k": top_k, "generated_text": generated_text})
        logging.info(f"prompt: {prompt}, max_tokens: {max_tokens}, temperature: {temperature}, top_k: {top_k}, generated_text: {generated_text}")

        return generated_text

    except Exception as e:
        history.append({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_k": top_k, "error": str(e)})
        logging.info(f"prompt: {prompt}, max_tokens: {max_tokens}, temperature: {temperature}, top_k: {top_k}, error: {str(e)}")

        return -1

@app.get("/history")
async def get_history():
    logging.info(f"History requested. Total entries: {len(history)}")
    return {"history": history}
