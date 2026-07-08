# Fine-Tuning Explanation

## Why full fine-tuning is expensive

Full fine-tuning updates all model weights. This requires more GPU memory, storage, and training time because gradients and optimizer states must be stored for the entire model. For a student or small team, full fine-tuning is often impractical on limited hardware.

## What LoRA does

LoRA, or Low-Rank Adaptation, freezes the original model weights and trains small low-rank adapter matrices inside selected layers. The model learns the task-specific behavior through these adapters, which are much smaller than the full model.

## What QLoRA does

QLoRA combines LoRA with quantized model loading. The base model is loaded in 4-bit precision while the LoRA adapters are trained. This reduces GPU memory requirements and makes fine-tuning possible on smaller GPUs.

## Why QLoRA is useful on limited GPU

QLoRA is useful because it keeps the large base model compressed during training. A 0.5B or 1.5B model can be fine-tuned more cheaply, leaving GPU memory for activations, batches, and the optimizer.

## Non-instruction fine-tuning

Non-instruction fine-tuning uses raw domain text instead of question-answer pairs. In this project, raw IT policy paragraphs help the model learn helpdesk terminology, process language, and background knowledge.

## Instruction fine-tuning

Instruction fine-tuning trains the model on user questions and ideal assistant responses. It teaches the model how to follow instructions, structure answers, and respond in the expected domain tone.

## DPO

DPO, or Direct Preference Optimization, aligns the model with preferred answers by training on chosen and rejected response pairs. In this project, chosen answers are safe, professional, and policy-specific, while rejected answers are vague, unsafe, or incomplete.

## Difference between SFT and DPO

SFT teaches the model what a good answer looks like. DPO teaches the model to prefer better answers over weaker alternatives. SFT builds the basic task behavior, while DPO improves quality, safety, and preference alignment.

## Hyperparameters Used

| Stage | Rank | Alpha | Dropout | Learning Rate | Batch Size | Gradient Accumulation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Non-instruction FT | 16 | 32 | 0.0 | 2e-4 | 2 | 8|
| Instruction SFT | 16 | 32 | 0.0 | 2e-4 | 2 | 8 |
| DPO alignment | 32 | 32 | 0.0 | 5e-6 | 1 | 8 |

These values were selected to keep memory usage low while still allowing the adapter to learn domain behavior. Dropout is set to zero because Unsloth commonly recommends this for optimized LoRA training.
