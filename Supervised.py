import torch
import torch.nn as nn
import re
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling




from huggingface_hub import login
login()

from datasets import load_dataset
dataset_id = 'AI-MO/NuminaMath-TIR'
train_ds, test_ds = load_dataset(dataset_id, split=['train[:5%]', 'test[:5%]'])


def wrap_think_answer(ex):
     solution = ex['solution']
     match = re.findall(r'\\boxed\{([^}]+)\}', solution)
     if match:
      last_match = match[-1]
      thinking = solution.split(f"\\boxed{{{last_match}}}")[0].strip()
      ex['solution'] = f"<think>{thinking}</think><answer>{last_match}</answer>"
     else:
      raise ValueError("No \\boxed{…} in solution")

     return ex

train_ds = train_ds.map(wrap_think_answer)
test_ds  = test_ds.map(wrap_think_answer)


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags."
)

def make_conversation(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["problem"]},
            {"role": "assistant", "content": ex["solution"]},
        ],
        "solution": ex.get("solution", "")

    }

train_ds = train_ds.map(make_conversation)
test_ds  = test_ds.map(make_conversation)
train_ds = train_ds.remove_columns(['messages', 'problem'])
test_ds  = test_ds.remove_columns(['messages', 'problem'])

model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

def tokenize_fn(examples):
    batch_input_ids, batch_attention_mask, batch_labels = [], [], []
    for prompt, sol in zip(examples["prompt"], examples["solution"]):
        p = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False) 
        full = p 
        tok = tokenizer(full, truncation=True, padding="max_length", max_length=200)
        ids = tok["input_ids"]
        batch_input_ids.append(ids)
        batch_attention_mask.append(tok["attention_mask"])
        batch_labels.append(ids.copy())
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels
    }

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["prompt","solution"])
test_ds  = test_ds.map(tokenize_fn, batched=True, remove_columns=["prompt","solution"])

sft_args = TrainingArguments(
    output_dir="Qwen2-0.5B-SFT-full",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    learning_rate=5e-5,
    bf16=True,
    logging_steps=50,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=4500,
    remove_unused_columns=False,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)

trainer = Trainer(
    model=model,
    args=sft_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)


trainer.train()
