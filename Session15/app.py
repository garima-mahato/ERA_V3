import torch
import gradio as gr
from transformers import GPT2Tokenizer
from model import SmolLM2Config, plSmolLM2

tokenizer = GPT2Tokenizer.from_pretrained(
    "HuggingFaceTB/cosmo2-tokenizer"
)
tokenizer.pad_token = tokenizer.eos_token
vocab_size = tokenizer.vocab_size


# Load the model
def load_model():
    config = SmolLM2Config()
    model = plSmolLM2(config, None, None, None)  # Create base model instead of Lightning model

    # Load just the model weights
    state_dict = torch.load("model_weights.pth", map_location="cpu")['model_state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    return model


def generate_text(prompt, max_tokens, temperature=0.8, top_k=40):
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

        return generated_text

    except Exception as e:
        return f"An error occurred: {str(e)}"


# Load the model globally
model = load_model()

# Create the Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            label="Enter your prompt", placeholder="Hello there!", lines=3
        ),
        gr.Slider(
            minimum=50,
            maximum=500,
            value=100,
            step=10,
            label="Maximum number of tokens",
        ),
    ],
    outputs=gr.Textbox(label="Generated Text", lines=10),
    title="Custom SmolLM2 Text Generator",
    description="Enter a prompt and a custom SmolLM2 model will continue.",
    examples=[
        ["Hello there! Today, we are going to talk about", 100],
        ["Logical implication is a fundamental", 200],
        ["o find the second derivative of", 150],
    ],
)

if __name__ == "__main__":
    demo.launch()