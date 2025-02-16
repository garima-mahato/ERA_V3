from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get_inputs():
    html_content = """
    <form action="/generate" method="post">
        <label for="prompt">Enter Prompt:</label>
        <input type="string" id="prompt" name="prompt" required>
        <br>
        <label for="max_tokens">Enter Max Tokens:</label>
        <input type="number" id="max_tokens" name="max_tokens" required>
        <br>
        <button type="submit">Submit</button>
    </form>
    """
    return html_content

@app.post("/generate")
async def generate(prompt: str = Form(...), max_tokens: int = Form(...)):
    response = requests.post("http://smollm2_135_service:8001/generate_text", json={"prompt": prompt, "max_tokens": max_tokens})
    return response.text
