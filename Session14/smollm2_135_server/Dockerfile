FROM python:3.9-slim
WORKDIR /app
COPY model_weights.pth .
COPY model.py .
COPY main.py .
RUN pip install fastapi uvicorn torch transformers pytorch-lightning datasets wandb lightning
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
