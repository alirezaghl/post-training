import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from math_verify import parse, verify
from torch.optim import Adam
from tqdm import tqdm
import os
import re
import time
from datasets import load_dataset
from Supervised import tokenizer, make_conversation
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think> </think> and "
    "<answer> </answer> tags."
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"]  
    }
dataset_id = 'AI-MO/NuminaMath-TIR'
train_dataset, test_dataset = load_dataset(dataset_id, split=['train[:5%]', 'test[:20%]'])  
train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(['messages', 'problem'])
test_dataset = test_dataset.remove_columns(['messages', 'problem'])

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

def format_reward(completions, **kwargs):
    rewards_list = []
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"

    for completion in completions:
        completion_content = completion[0]["content"]
        if re.search(pattern, completion_content, re.DOTALL):
            rewards_list.append(1.25)
        else:
            rewards_list.append(-1)

    return rewards_list

def accuracy_reward(completions, **kwargs):

    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, solution in zip(completion_contents, solutions):
        try:
            parsed_pred = parse(content)
            parsed_solution = parse(solution)
            correct = verify(parsed_pred, parsed_solution)
            rewards.append(1 if correct else 0)
        except Exception as e:
            print(f"NO REWARD CAN BE CALCULATED!")
            rewards.append(0)           
    
    return rewards

def generate_with_reasoning(model, tokenizer, prompt):

    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False)

    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    input_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    generated_text = full_text[len(input_text):].strip()

    inference_duration = end_time - start_time

    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens

class REINFORCETrainer:
    def __init__(self, model, tokenizer, reward_funcs, train_dataset, lr=1e-5, batch_size=8, num_epochs=10):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.train_dataset = train_dataset
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.optimizer = Adam(model.parameters(), lr=self.lr)

    def compute_rewards(self, batch):
        prompt = batch["prompt"]  
        solution = batch["solution"]  
        
        prompts = [prompt]  
        solutions = [solution] 
        
        prompt_texts = [self.tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]

        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True) 
        completions = [[{"content": text}] for text in decoded_outputs]
        kwargs = {"solution": solutions}

        all_rewards = []
        for func in self.reward_funcs:
            func_rewards = func(completions, **kwargs)
            all_rewards.append(func_rewards)

        rewards = list(zip(*all_rewards))
        
        return torch.tensor(rewards, dtype=torch.float32).to(self.model.device)

    def update_model(self, batch, training_rewards):
        prompt = batch["prompt"]  
        prompts = [prompt]  
        
        prompt_texts = [self.tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
          
        output_ids = output_ids.clone().detach()
        logits = self.model(output_ids).logits
        logits = logits[:, :-1, :]
        output_ids_shifted = output_ids[:, 1:]
        program_length = inputs.input_ids.shape[1]

        logprobs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(logprobs, dim=2, index=output_ids_shifted.unsqueeze(2)).squeeze(2)

        generated_log_probs = token_log_probs[:, program_length-1:] 
        generated_token_ids = output_ids_shifted[:, program_length-1:]

        mask = (generated_token_ids != self.tokenizer.pad_token_id).float()
        sequence_log_probs = (generated_log_probs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

        loss = -(sequence_log_probs * training_rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0
            accuracy_successes = []  
            
            for batch in tqdm(self.train_dataset, desc=f"Epoch {epoch+1}"):

                rewards = self.compute_rewards(batch)

                training_rewards = rewards.sum(dim=1)

                loss = self.update_model(batch, training_rewards)
                epoch_loss += loss
                    
                avg_accuracy = rewards[:, 1].mean().item()
                accuracy_successes.append(1 if avg_accuracy > 0 else 0)
                    
                num_batches += 1
                    
                if num_batches % 5 == 0:
                    avg_format = rewards[:, 0].mean().item()
                    recent_10 = accuracy_successes[-10:]
                    accuracy_rate = sum(recent_10) / len(recent_10) * 100
                        
                    print(f"Batch {num_batches}: Format: {avg_format:.2f}, Accuracy: {avg_accuracy:.2f}")
                    print(f"  Last 10 batches accuracy: {accuracy_rate:.1f}%")

                    print(f"\nReasoning Generation")
                    test_batch = test_dataset[num_batches % len(test_dataset)]
                    generated_text, inference_time, num_tokens = generate_with_reasoning(
                        self.model, self.tokenizer, test_batch["prompt"]
                    )
                    print(f"Prompt: {test_batch['prompt']}")
                    print(f"Solution: {test_batch['solution']}")
                    print(f"Model Answer: {generated_text}")
                    print(f"Time: {inference_time:.2f}s, Tokens: {num_tokens}")
                        
                    torch.cuda.empty_cache()
                        
            total_accuracy = sum(accuracy_successes) / len(accuracy_successes) * 100 if accuracy_successes else 0
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}, Overall Accuracy: {total_accuracy:.1f}%")
        
        return self.model

if __name__ == "__main__":
    model_path = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/Qwen2-0.5B-SFT-full"
    print("Loading fine-tuned model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  
        device_map="auto",          
        trust_remote_code=True       
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = model.to("cuda")
    model = get_peft_model(model, lora_config)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = REINFORCETrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        train_dataset=train_dataset,  
        lr=1e-5,
        batch_size=1,  
        num_epochs=1
    )

    trained_model = trainer.train()
    
    output_dir = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/Qwen2-0.5B-REINFORCE-trained"
    print(f"Saving trained model to {output_dir}...")
    
    trained_model.save_pretrained(output_dir)
    
    tokenizer.save_pretrained(output_dir)
