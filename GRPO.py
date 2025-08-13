import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import os
import re
import time
import numpy as np

from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from math_verify import parse, verify

class TrainingConfig:
    DATASET_ID = 'AI-MO/NuminaMath-TIR'
    TRAIN_SPLIT = 'train[:20%]'
    TEST_SPLIT = 'test[:]'
    
    MODEL_PATH = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/Qwen2-0.5B-SFT-full"
    OUTPUT_DIR = "/home/neuroali/pytorch_projects/pytorch_cuda_env/RL-LLM/Qwen2-0.5B-GRPO-trained"
    
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 1
    NUM_EPOCHS = 1
    MAX_NEW_TOKENS = 128
    TEMPERATURE = 0.7
    KL_COEFF = 0.01      
    GENERATIONS_PER_SAMPLE = 2
    
    LORA_CONFIG = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
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
            {"role": "system", "content": TrainingConfig.SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
        "solution": example["solution"]  
    }

def collate_fn(batch):

    prompts = [item['prompt'] for item in batch]
    solutions = [item['solution'] for item in batch]
    return {
        'prompt': prompts,
        'solution': solutions
    }

def load_and_process_dataset():
    train_dataset, test_dataset = load_dataset(
        TrainingConfig.DATASET_ID, 
        split=[TrainingConfig.TRAIN_SPLIT, TrainingConfig.TEST_SPLIT]
    )
    
    train_dataset = train_dataset.map(make_conversation)
    test_dataset = test_dataset.map(make_conversation)
    
    train_dataset = train_dataset.remove_columns(['messages', 'problem'])
    test_dataset = test_dataset.remove_columns(['messages', 'problem'])
    
    return train_dataset, test_dataset

class RolloutGenerator:
    def __init__(self, model, tokenizer, generations_per_sample=4):
        self.model = model
        self.tokenizer = tokenizer
        self.GENERATIONS_PER_SAMPLE = generations_per_sample
        self.tokenizer.padding_side = "left"
        
    def generate_rollouts_batch(self, samples):

        all_prompt_texts = []
        for sample in samples:
            prompt_text = self.tokenizer.apply_chat_template(
                sample["prompt"], 
                tokenize=False, 
                add_generation_prompt=True
            )
            all_prompt_texts.extend([prompt_text] * self.GENERATIONS_PER_SAMPLE)
        
        inputs = self.tokenizer(
            all_prompt_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        generated_sequences = outputs
        all_generations = []
        all_finish_reasons = []
        
        for sequence in generated_sequences:
            generated_tokens = sequence[input_length:].tolist()
            
            if self.tokenizer.pad_token_id in generated_tokens:
                try:
                    pad_idx = generated_tokens.index(self.tokenizer.pad_token_id)
                    generated_tokens = generated_tokens[:pad_idx]
                except ValueError:
                    pass
            
            if self.tokenizer.eos_token_id in generated_tokens:
                finish_reason = "stop"
                if generated_tokens and generated_tokens[-1] == self.tokenizer.eos_token_id:
                    generated_tokens = generated_tokens[:-1]
            else:
                finish_reason = "length"
            
            all_generations.append(generated_tokens)
            all_finish_reasons.append(finish_reason)
        
        return all_generations, all_finish_reasons, generated_sequences
    
    def create_training_episodes(self, samples, all_generations, all_finish_reasons, rewards):

        assert len(all_generations) == len(samples) * self.GENERATIONS_PER_SAMPLE
        assert len(rewards) == len(all_generations)
        
        groups = [
            list(range(i, i + self.GENERATIONS_PER_SAMPLE))
            for i in range(0, len(all_generations), self.GENERATIONS_PER_SAMPLE)
        ]
        
        all_query_token_ids = []
        all_response_token_ids = []
        all_advantages = []
        
        stats = {
            "response_lengths": [],
            "rewards": [],
            "non_stop_rate": [],
        }
        
        for sample, group_indices in zip(samples, groups):
            response_token_ids = [all_generations[i] for i in group_indices]
            finish_reasons = [all_finish_reasons[i] for i in group_indices]
            group_rewards = np.array([rewards[i] for i in group_indices])
            
            if len(group_rewards) > 1 and group_rewards.std() > 1e-8:
                advantages = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-4)
            else:
                advantages = np.zeros_like(group_rewards)
            
            for resp_tokens, advantage in zip(response_token_ids, advantages):
                input_tokens = self.tokenizer.apply_chat_template(
                    sample["prompt"], 
                    tokenize=True, 
                    add_generation_prompt=True
                )
                
                all_query_token_ids.append(input_tokens)
                all_response_token_ids.append(resp_tokens)
                
                token_advantages = [float(advantage)] * len(resp_tokens)
                all_advantages.append(token_advantages)
            
            stats["rewards"].extend(group_rewards.tolist())
            stats["response_lengths"].extend([len(tokens) for tokens in response_token_ids])
            stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
        
        episodes = {
            "all_query_token_ids": all_query_token_ids,
            "all_response_token_ids": all_response_token_ids,  
            "all_advantages": all_advantages,
        }
        
        return episodes, stats

def format_reward(response_text):
    
    pattern = r"<think>.*?</think>.*?<answer>.*?</answer>"
    return 1.25 if re.search(pattern, response_text, re.DOTALL) else -1.0

def accuracy_reward(response_text, solution):
    try:
        parsed_pred = parse(response_text)
        parsed_solution = parse(solution)
        correct = verify(parsed_pred, parsed_solution)
        return 1.0 if correct else 0.0
    except Exception as e:
        return 0.0

def compute_rewards(generated_texts, solutions):
    rewards = []
    repeated_solutions = []
    
    # Repeat solutions to match generations
    for solution in solutions:
        repeated_solutions.extend([solution] * TrainingConfig.GENERATIONS_PER_SAMPLE)
    
    for text, solution in zip(generated_texts, repeated_solutions):
        format_r = format_reward(text)
        accuracy_r = accuracy_reward(text, solution)
        total_reward = format_r + accuracy_r
        rewards.append(total_reward)
    
    return rewards

def setup_model_and_tokenizer():
    ref_model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.MODEL_PATH,
        torch_dtype=torch.bfloat16,  
        device_map="auto",          
        trust_remote_code=True       
    )

    policy_model = AutoModelForCausalLM.from_pretrained(
        TrainingConfig.MODEL_PATH,
        torch_dtype=torch.bfloat16,  
        device_map="auto",          
        trust_remote_code=True       
    )

    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_PATH)
    
    policy_model = policy_model.to("cuda")
    policy_model = get_peft_model(policy_model, TrainingConfig.LORA_CONFIG)
    
    ref_model = ref_model.to("cuda")

    for param in ref_model.parameters():
        param.requires_grad = False
    
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    return ref_model, policy_model, tokenizer

class GRPOTrainer:
    def __init__(self, ref_model, policy_model, tokenizer, train_dataloader, test_dataset, config=None):
        self.ref_model = ref_model
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.test_dataset = test_dataset
        self.config = config or TrainingConfig()
        
        self.optimizer = Adam(policy_model.parameters(), lr=self.config.LEARNING_RATE)
        
        self.rollout_generator = RolloutGenerator(
            policy_model, 
            tokenizer, 
            generations_per_sample=self.config.GENERATIONS_PER_SAMPLE
        )
        
    def compute_logprobs_for_episodes(self, model, episodes):

        all_logprobs = []
        
        batch_size = 4
        query_ids = episodes["all_query_token_ids"]
        response_ids = episodes["all_response_token_ids"]
        
        for i in range(0, len(query_ids), batch_size):
            batch_queries = query_ids[i:i+batch_size]
            batch_responses = response_ids[i:i+batch_size]
            
            combined_sequences = []
            response_lengths = []
            
            for query, response in zip(batch_queries, batch_responses):
                combined = query + response
                combined_sequences.append(combined)
                response_lengths.append(len(response))
            
            max_len = max(len(seq) for seq in combined_sequences)
            padded_sequences = []
            
            for seq in combined_sequences:
                padded = seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
                padded_sequences.append(padded)
            
            sequences_tensor = torch.tensor(padded_sequences, device=model.device)
            
            model.eval()
            with torch.no_grad() if model == self.ref_model else torch.enable_grad():
                outputs = model(sequences_tensor)
                
            logits = outputs.logits / self.config.TEMPERATURE
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = sequences_tensor[:, 1:].contiguous()
            
            log_probs = F.log_softmax(shift_logits, dim=-1)
            gathered_log_probs = torch.gather(log_probs, 2, shift_labels.unsqueeze(2)).squeeze(2)
            
            for j, resp_len in enumerate(response_lengths):
                query_len = len(batch_queries[j])
                start_idx = query_len - 1
                end_idx = start_idx + resp_len
                
                response_logprobs = gathered_log_probs[j, start_idx:end_idx]
                
                response_tokens = torch.tensor(batch_responses[j], device=model.device)
                mask = (response_tokens != self.tokenizer.pad_token_id).float()
                
                
                avg_logprob = (response_logprobs * mask).sum() / mask.sum()
                all_logprobs.append(avg_logprob)
        
        return torch.stack(all_logprobs)

    def update_policy_with_episodes(self, episodes):
        
        advantages = []
        for adv_list in episodes["all_advantages"]:
            avg_advantage = sum(adv_list) / len(adv_list) if adv_list else 0.0
            advantages.append(avg_advantage)
        
        advantages_tensor = torch.tensor(advantages, device=self.policy_model.device)
        
        policy_log_probs = self.compute_logprobs_for_episodes(self.policy_model, episodes)
        ref_log_probs = self.compute_logprobs_for_episodes(self.ref_model, episodes)
        
        num_samples = len(episodes["all_query_token_ids"]) // self.config.GENERATIONS_PER_SAMPLE
        grouped_advantages = advantages_tensor.view(num_samples, self.config.GENERATIONS_PER_SAMPLE)
        grouped_policy_logprobs = policy_log_probs.view(num_samples, self.config.GENERATIONS_PER_SAMPLE)
        grouped_ref_logprobs = ref_log_probs.view(num_samples, self.config.GENERATIONS_PER_SAMPLE)
        
        kl_penalty = torch.exp(grouped_ref_logprobs - grouped_policy_logprobs) - (grouped_ref_logprobs - grouped_policy_logprobs) - 1
        
        pg_loss = -(grouped_policy_logprobs * grouped_advantages).mean()
        kl_loss = self.config.KL_COEFF * kl_penalty.mean()
        
        total_loss = pg_loss + kl_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item(), grouped_advantages

    def train(self):

        for epoch in range(self.config.NUM_EPOCHS):
            epoch_loss = 0
            num_batches = 0
            
            for batch in tqdm(self.train_dataloader, desc=f"GRPO Epoch {epoch+1}"):
                
                samples = []
                for i in range(len(batch["prompt"])):
                    samples.append({
                        "prompt": batch["prompt"][i],
                        "solution": batch["solution"][i]
                    })
                
                generations, finish_reasons, full_sequences = self.rollout_generator.generate_rollouts_batch(samples)
                
                generated_texts = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
                
                rewards = compute_rewards(generated_texts, batch["solution"])
                
                episodes, stats = self.rollout_generator.create_training_episodes(
                    samples, generations, finish_reasons, rewards
                )
                
                loss, advantages = self.update_policy_with_episodes(episodes)
                
                epoch_loss += loss
                num_batches += 1
                
                if num_batches % 1 == 0:
                    avg_reward = np.mean(stats["rewards"])
                    
                    print(f"\nBatch {num_batches}:")
                    print(f"  Loss: {loss:.4f}")
                    print(f"  Avg Reward: {avg_reward:.3f}")
                    print(f"  Avg response length: {np.mean(stats['response_lengths']):.1f}")
                    
                    if generated_texts:
                        print(f" generation: {generated_texts[0][:-150]}...")
                    
                    torch.cuda.empty_cache()
                
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}, Avg Loss: {avg_loss:.4f}")
        
        return self.policy_model

def save_trained_model(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    train_dataset, test_dataset = load_and_process_dataset()
    ref_model, policy_model, tokenizer = setup_model_and_tokenizer()
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=TrainingConfig.BATCH_SIZE, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    trainer = GRPOTrainer(
        ref_model=ref_model,
        policy_model=policy_model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        test_dataset=test_dataset,
        config=TrainingConfig()
    )

    trained_model = trainer.train()
    save_trained_model(trained_model, tokenizer, TrainingConfig.OUTPUT_DIR)

if __name__ == "__main__":
    main()