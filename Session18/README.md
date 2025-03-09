# QLoRA Fine-Tuning of Microsoft phi2 with OASST1 Data



This repository contains the code and configuration used to fine-tune the Microsoft phi2 model using the QLoRA method on the OASST1 dataset. The fine-tuning process was inspired by techniques demonstrated in [this notebook](https://github.com/mshumer/gpt-llm-trainer/blob/main/One_Prompt___Fine_Tuned_LLaMA_2.ipynb) and adapted to work with phi2.

## Overview

QLoRA (Quantized Low Rank Adaptation) is an efficient approach for fine-tuning large language models. By quantizing the model weights to lower precision and applying low-rank adaptations, QLoRA allows us to train large-scale models on affordable hardware while retaining much of the original model’s performance.

In this, we:
- Fine-tune Microsoft’s phi2 model using QLoRA.
- Use the OASST1 (Open Assistant) dataset to adapt the model’s responses.
- Leverage techniques from the LLaMA 2 fine-tuning notebook for guidance and best practices.


## Fine-Tuning Process

### 1. Data Preparation

The OASST1 dataset is preprocessed to suit the fine-tuning requirements. Use the `prepare_data.py` script to:
- Clean and tokenize the raw text data.
- Format the data into a structure compatible with the training script.
- Optionally filter or augment the data as necessary.

### 2. Model and QLoRA Configuration

The QLoRA configuration in `config/qlora_config.json` defines:
- Quantization parameters (e.g., 4-bit precision).
- Rank adaptation parameters.
- Other hyperparameters specific to the phi2 model.

The training configuration file (`config/training_config.yaml`) contains:
- Learning rate and optimizer settings.
- Batch sizes.
- Checkpoint saving intervals.
- Number of training epochs.

### 3. Fine-Tuning Execution

Run the main training script using:
```bash
python scripts/train.py --config config/training_config.yaml --qlora_config config/qlora_config.json
```

This command launches the training process, where:
- The phi2 model is loaded from the Hugging Face Model Hub.
- QLoRA applies quantization and low-rank adapters to reduce memory consumption.
- The model is then fine-tuned on the preprocessed OASST1 data.

### 4. Evaluation and Testing

After training, evaluate the model’s performance using:
```bash
python scripts/evaluate.py --model_path models/phi2_finetuned --data_path data/validation.json
```
This evaluation script computes relevant metrics (e.g., perplexity, response quality) to validate the fine-tuning process.

### 5. Notebook Walkthrough

For a detailed, step-by-step demonstration, see the notebook in `notebooks/One_Prompt_FineTuned_phi2.ipynb`. This notebook was adapted from the [LLaMA 2 fine-tuning example](https://github.com/mshumer/gpt-llm-trainer/blob/main/One_Prompt___Fine_Tuned_LLaMA_2.ipynb) and includes:
- Data inspection and preprocessing.
- Configuration details.
- Training progress visualizations.
- Evaluation and inference examples.

## Results

The fine-tuned Microsoft phi2 model shows:
- Improved alignment with the OASST1 conversational data.
- Enhanced generation quality on various prompt tasks.
- Efficient resource utilization thanks to QLoRA's quantization methods.


## Acknowledgements

- The fine-tuning approach was inspired by the methods outlined in [this fine-tuning notebook](https://github.com/mshumer/gpt-llm-trainer/blob/main/One_Prompt___Fine_Tuned_LLaMA_2.ipynb).
- Thanks to the developers of QLoRA and Hugging Face for their continuous contributions to the open-source community.

