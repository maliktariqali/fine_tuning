# Domain-Specific AI Assistant Fine-Tuning with Unsloth

## Project Title

IT Helpdesk Assistant using Unsloth Fine-Tuning

## Domain Selected

IT Helpdesk Assistant

## Business Problem

A company needs an internal assistant that can answer common IT helpdesk questions in a clear, safe, and policy-specific way. Generic base models often give broad advice, but employees need answers that include company workflow details, security precautions, escalation rules, and ticket information.

## Dataset Details

This project uses synthetic but cleaned IT helpdesk policy data prepared for the assignment:

- `data/non_instruction_data.txt`: 100 raw IT policy paragraphs for non-instruction fine-tuning.
- `data/instruction_dataset.jsonl`: 191 instruction-response examples for SFT.
- `data/preference_dataset.jsonl`: 150 prompt/chosen/rejected examples for DPO alignment.

The dataset covers password resets, MFA, VPN, Wi-Fi, software installation, phishing, email, printer issues, hardware support, cloud storage, secure transfer, access requests, onboarding, offboarding, and incident escalation.

## Base Model Used

`unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit`

This model was selected because it is small enough for student GPU environments and compatible with QLoRA-style training through Unsloth.

## Repository Structure

```text
domain-ai-assistant-finetuning/
├── data/
│   ├── non_instruction_data.txt
│   ├── instruction_dataset.jsonl
│   └── preference_dataset.jsonl
├── notebooks/
│   ├── non_instruction_finetuning.ipynb
│   ├── instruction_finetuning.ipynb
│   └── dpo_alignment.ipynb
├── reports/
│   ├── base_model_evaluation.md
│   ├── sft_model_comparison.md
│   ├── final_evaluation.md
│   └── fine_tuning_explanation.md
├── src/
│   └── inference.py
├── README.md
└── requirements.txt
```

## Non-Instruction Fine-Tuning Approach

Stage 1 uses `data/non_instruction_data.txt` as raw domain text. The notebook loads the text, creates a Hugging Face `Dataset`, loads the base model with Unsloth in 4-bit mode, applies LoRA adapters, and trains on raw policy paragraphs. This helps the model learn IT helpdesk terminology and policy wording before instruction training.

Notebook: `notebooks/non_instruction_finetuning.ipynb`

## Instruction Fine-Tuning Approach

Stage 2 uses `data/instruction_dataset.jsonl`, which contains 191 question-answer examples. Each example is formatted with a system prompt, user question, and assistant answer. The notebook uses supervised fine-tuning so the assistant learns how to answer real helpdesk questions in the correct style.

Notebook: `notebooks/instruction_finetuning.ipynb`

## DPO Alignment Approach

Stage 3 uses `data/preference_dataset.jsonl`, which contains 150 prompts with chosen and rejected responses. The chosen responses are safe, professional, helpful, and domain-specific. The rejected responses are generic, unsafe, incomplete, or not policy-aware. DPO improves the model's preference for better responses.

Notebook: `notebooks/dpo_alignment.ipynb`

## LoRA / QLoRA Configuration

| Stage | Rank | Alpha | Dropout | Learning Rate | Batch Size | Gradient Accumulation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-instruction FT | 16 | 32 | 0.0 | 2e-4 | 2 | 8|
| Instruction SFT | 16 | 32 | 0.0 | 2e-4 | 2 | 8 |
| DPO alignment | 32 | 32 | 0.0 | 5e-6 | 1 | 8 |

The model is loaded in 4-bit mode, which follows the QLoRA idea of reducing memory use while training small LoRA adapters.

## How to Run

1. Open the project in Google Colab or another CUDA GPU environment.
2. Install the libraries from `requirements.txt`.
3. Run `notebooks/non_instruction_finetuning.ipynb`.
4. Run `notebooks/instruction_finetuning.ipynb`.
5. Run `notebooks/dpo_alignment.ipynb`.
6. Test the final assistant with:

```bash
python src/inference.py
```

## Before vs After Output Comparison

See:

- `reports/base_model_evaluation.md`
- `reports/sft_model_comparison.md`
- `reports/final_evaluation.md`

Expected improvement:

- Base model: generic IT answers.
- SFT model: domain-specific answers using internal policy details.
- DPO model: safer, more professional, and more complete answers.

## Training Screenshots or Logs

The notebooks print training progress during `trainer.train()`. For final submission, run the notebooks on GPU and add screenshots of:

- Stage 1 training loss
- Stage 2 SFT training loss
- Stage 3 DPO training loss/reward metrics
- Final inference examples

## Final Observations

This project demonstrates a practical three-stage workflow:

1. Non-instruction fine-tuning teaches domain language.
2. Instruction fine-tuning teaches the assistant how to answer employee questions.
3. DPO alignment improves answer preference toward safe, helpful, and professional responses.

## Challenges Faced

- Small student GPUs limit model size and batch size.
- Synthetic datasets must be reviewed carefully to avoid unsafe or unrealistic policy answers.
- DPO requires high-quality rejected responses so the model learns meaningful preferences.

## Future Improvements

- Replace synthetic policies with real approved internal IT knowledge base articles.
- Add more multi-turn troubleshooting conversations.
- Evaluate with a larger test set and human review.
- Export a merged model for easier deployment.

## Public References

- Unsloth fine-tuning documentation: https://docs.unsloth.ai/get-started/fine-tuning-llms-guide
- Unsloth reinforcement learning / DPO documentation: https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide
- Base model page: https://huggingface.co/unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit
