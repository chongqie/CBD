# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, '/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/joyce/llada/LLaDA-8B-Instruct')
#from generate import generate
import argparse
import json
import random
from pathlib import Path
import gc
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType
import re

from modeling_llada import LLaDAModelLM, LLaDAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from transformers import AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model
from modeling_llada import LLaDAModelLM
from datasets import Dataset
from tqdm import tqdm 
from accelerate import Accelerator

from generate_parallel import generate_parallel

import torch
import flash_attn
import numpy as np




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./finetuned_llada")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--block_length", type=int, default=64)
    parser.add_argument("--grad_acc_steps", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=1024)
    return parser.parse_args()
    
    
from glob import glob

def load_json_folder(file_path, max_samples=None):
    if os.path.isfile(file_path):
        files = [file_path]
    else:
        raise FileNotFoundError(f"Dataset path does not exist or is not a file: {file_path}")

    if len(files) == 0:
        raise FileNotFoundError(f"No valid files found in {file_path}")
    
    logger.info(f"Found {len(files)} file in {file_path}")
    all_records = []

    for fp in files:
        logger.info(f"Processing file: {fp}")
        try:
            if fp.endswith('.jsonl'):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line: 
                            try:
                                all_records.append(json.loads(line))
                            except json.JSONDecodeError:
                                logger.warning(f"Skipping invalid JSON line in {fp}")
            else:
                raise ValueError(f"Unsupported file format for {fp}, only '.jsonl' is supported.")
        except Exception as e:
            logger.error(f"Error processing file {fp}: {str(e)}")

    if not all_records:
        raise ValueError(f"No valid data found in the file: {file_path}")

    if max_samples:
        all_records = random.sample(all_records, min(max_samples, len(all_records)))

    logger.info(f"Total samples: {len(all_records)}")
    return Dataset.from_list(all_records)


def prepare_examples(raw, tokenizer, mask_len_per_step=128, mask_id=126336, split_token_id=126085, device="cpu"):
    examples = []
    for idex, ex in enumerate(tqdm(raw, desc="Preparing examples")):
        input_ids, step_list, semantic_block_lengths, prompt_len, detailed_range = _tokenize_with_mask(
            ex, tokenizer, mask_id, split_token_id, mask_len_per_step 
        )
        
        num_step_count = len(step_list)
        mask_pos = []
        block_ranges = []

        offset = 0
        for L in semantic_block_lengths:
            start = offset
            end = L + offset 
            block_ranges.append((start, end))
            offset = end
        
        for (s, e) in block_ranges:
            split_offset = 1
            if(e == len(input_ids)): 
                mask_start = e - 2 * mask_len_per_step
                split_offset = 0
            else:
                mask_start = e - mask_len_per_step - split_offset
            mask_pos.append((mask_start, e - split_offset))

        mask_count = int((input_ids == mask_id).sum().item())
        if mask_count == 0:
            print(f"Warning:example {ex[0]} has no mask tokens, skipping.")
            continue
        gen_actual = mask_count
        

        
        examples.append({
            "input_ids": input_ids,
            "steps": step_list,
            "semantic_block_lengths": semantic_block_lengths,
            "num_step_count": num_step_count,
            "gen_actual": gen_actual,
            "mask_pos": mask_pos,
            "prompt_len": prompt_len,
            "detailed_process": ex.get("detailed_process", ""),  
            "detailed_range" : detailed_range

        })

    return examples


def _tokenize_with_mask(entry, tokenizer, mask_token_id,  split_token_id, mask_len_per_step=128):
        
    prompt_text = (
      "- Your job is to EXPAND the draft into a full reasoning."
      "- ONLY output the reasoning in the required format; do NOT repeat the Question, Summary, or Steps."
      "- Each paragraph should explain how to apply that step to the given Question and help reach the solution."
      "- The number of paragraphs MUST exactly match the number of steps."
      "- Keep reasoning concise, logically consistent, and focused strictly on solving the question."
      "- At the end of the reasoning, clearly state the final numeric answer in the format: '#### <number>'"
    )



    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids[0]


    input_ids = prompt_ids.clone()
    semantic_block_length = []
    
    

    if isinstance(entry, dict):
        question = entry.get("question", "")
        summary = entry.get("summary", "")
        steps_list = entry.get("steps", [])
        detailed_process = entry.get("detailed_process", "")
    elif isinstance(entry, str):
        summary = entry
        steps_list = re.findall(r"(Step \d+ - .*?:)", entry)

    else:
        raise ValueError(f"Unsupported entry type: {type(entry)}")
    
    if question.strip():
        question_text = f"Question:{question}\n"
        question_ids = tokenizer(question_text, return_tensors="pt").input_ids[0]
        input_ids = torch.cat([input_ids, question_ids], dim=0) if input_ids.numel() > 0 else question_ids

    if summary.strip():
        summary_text = f"Summary:{summary}\n"
        summary_ids = tokenizer(summary_text, return_tensors="pt").input_ids[0]
        input_ids = torch.cat([input_ids, summary_ids], dim=0) if input_ids.numel() > 0 else summary_ids
        prompt_len = len(input_ids)
    
    detailed_range = []

    if detailed_process.strip():
        #prefix_text = "Detailed Explanation:\n\n"
        #prefix_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
        detailed_text = detailed_process


        for i in range(len(steps_list)):
            step_number = i + 1
            pattern = fr"Step\s*{step_number}\s*-\s*.*?:"
            start_match = re.search(pattern, detailed_text)
            if not start_match:
                print(f"Warning: Step not found: {step}")
                continue
            start_idx = start_match.end()

            if i + 1 < len(steps_list):
                next_pattern = fr"Step\s*{step_number+1}\s*-\s*.*?:"
                next_match = re.search(next_pattern, detailed_text)
                end_idx = next_match.start() if next_match else len(detailed_text)
            else:
                end_idx = len(detailed_text)

            detailed_range.append((start_idx, end_idx))

    
    def convert_char_to_token_ranges(detailed_text, detailed_range, tokenizer):
        token_ranges = []
        for (start_c, end_c) in detailed_range:
            prefix = detailed_text[:start_c]
            seg = detailed_text[start_c:end_c]
            start_t = len(tokenizer(prefix, add_special_tokens=False).input_ids)
            seg_len = len(tokenizer(seg, add_special_tokens=False).input_ids)
            token_ranges.append((start_t, start_t + seg_len))
        return token_ranges

    token_ranges = convert_char_to_token_ranges(detailed_text, detailed_range, tokenizer)
    detailed_range = token_ranges

    for i, step_text in enumerate(steps_list):
        step_ids = tokenizer(step_text, return_tensors="pt").input_ids[0]
        mask_len = max(mask_len_per_step, len(step_ids))

        if i == len(steps_list) - 1:
            mask_len += mask_len_per_step 

        mask_ids = torch.full((mask_len,), mask_token_id, dtype=torch.long)
        input_ids = torch.cat([input_ids, step_ids, mask_ids], dim=0)
        
        add_split = 0
        if split_token_id is not None and i != len(steps_list) - 1:
            input_ids = torch.cat([input_ids, torch.tensor([int(split_token_id)], dtype=torch.long)], dim=0)
            add_split = 1

        if not semantic_block_length:
            semantic_block_length.append(len(input_ids))
            continue
        semantic_block_length.append(len(step_ids) + mask_len + add_split)
    
    
    return input_ids, steps_list, semantic_block_length, prompt_len, detailed_range



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 确保每次运行时使用相同的计算顺序
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




def main():
    args = parse_args()
    accelerator = Accelerator()
    seed = 42  # 可以选择任意整数
    set_seed(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. tokenizer
    accelerator.print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    pad_token_id = tokenizer.convert_tokens_to_ids("<|reserved_token_0|>")
    print(f"PAD token id: {pad_token_id}")
    split_token_id = tokenizer.convert_tokens_to_ids("<|reserved_token_1|>")
    print(f"split token id:{split_token_id}")
    

    print("Tokenizer loaded.")
    mask_token_id = tokenizer.convert_tokens_to_ids("<|mdm_mask|>")
    print(f"Mask token id: {mask_token_id}")


    # 2. load & prepare dataset
    accelerator.print(" Loading dataset...")
    raw = load_json_folder(args.data_path, max_samples=args.max_samples)
    print(f"The raw dataset has {len(raw)} examples.")
    dataset = prepare_examples(raw, tokenizer, args.block_length, mask_token_id, split_token_id, device)
    
    split = int(0.9 * len(dataset))
    train_dataset = dataset[:split]

    val_dataset =dataset[split:]
    accelerator.print(f"Train: {len(train_dataset)} examples, Val: {len(val_dataset)} examples.")


    # 3. load base model
    accelerator.print(" Loading model...")
    base_model = LLaDAModelLM.from_pretrained(args.model_path)
    base_model.resize_token_embeddings(len(tokenizer))
    #print("PAD id:", tokenizer.pad_token_id)
    '''
    base_model.flash_attn_func = None
        
    
    print("LLaDA flash_attn_func is None:", base_model.flash_attn_func is None)
'''


    # 4. LoRA 
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    model.to(device)


        
    def collate_fn(batch, pad_token_id, ignore_index=-100):
        input_ids_list = [b["input_ids"] for b in batch]
        max_len = max(x.size(0) for x in input_ids_list)

        input_ids_padded = torch.stack([
            F.pad(x, (0, max_len - x.size(0)), value=pad_token_id) for x in input_ids_list
        ])
        mask_pos_list = [b["mask_pos"] for b in batch]
        return input_ids_padded, mask_pos_list, batch  
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id)
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01, betas=(0.9, 0.95))
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)



    model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    
    
    prefix_text = "Detailed Explanation:\n\n"
    prefix_len = len(tokenizer(prefix_text, add_special_tokens=False).input_ids)
    

    per_steps = args.steps
    


    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for step, (input_ids, mask_pos_list, batch_raw) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            #print(input_ids)

            # forward
            outputs = model(input_ids)
            logits = outputs.logits  # [B, L, V]
            

            losses = []
            total_tokens = 0
            for i, ex in enumerate(batch_raw):
                mask_ranges = mask_pos_list[i]  # [(start, end), ...]
                gen_actual = ex.get("gen_actual",0)
                prompt_len = ex.get("prompt_len", 0)        
                detailed_text = ex.get("detailed_process", "")
                detailed_range = ex.get("detailed_range", [])
                
                sample_tokens = sum(end - start for (start, end) in detailed_range)
                total_tokens += sample_tokens
                
                if not detailed_text:
                    continue
    
                target_tokens = tokenizer(
                    detailed_text,
                    return_tensors="pt",
                    add_special_tokens=False
                ).input_ids[0].to(device)
    
                for j, ((mask_start, mask_end), (gold_start, gold_end)) in enumerate(zip(mask_ranges, detailed_range)):
                    '''
                    print(f"for {step} in {j} turn:\nGold range:{gold_start}, {gold_end}")
                    print(f"Mask range:{mask_start}, {mask_end}") 
                    print("Mask range len:", mask_end - mask_start)
                    print("Gold range len:", gold_end - gold_start)
                    '''

                    gold = target_tokens[gold_start: gold_end]
                    #print("Gold segment (token):", tokenizer.decode(gold))

                    pred = logits[i, mask_start: mask_end, :] # [seg_len, V]
                    
                    #input_segment = input_ids[i, start:end]
                    #print("Input segment (token):", tokenizer.decode(input_segment))
                    
                    
                    pred_ids = pred.argmax(dim=-1)
                    #print("Pred segment (decoded):", tokenizer.decode(pred_ids))
                    
                    '''                
                    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id  
                    if gold.size(0) < pred.size(0):
                        pad_len = pred.size(0) - gold.size(0)
                        gold = torch.cat([gold, torch.full((pad_len,), pad_token_id, device=device, dtype=torch.long)])
                    elif gold.size(0) > pred.size(0):
                        gold = gold[:pred.size(0)]


    
                    if gold.numel() == 0 or pred.size(0) == 0:
                        continue
                    '''
                    gold_size = gold.size(0)
                    pred_size = pred.size(0)
                                        
                    if gold_size < pred_size:
                        # gold 比 pred 短 → 在 gold 末尾填 pad_token_id
                        pad_size = pred_size - gold_size
                        pad_tensor = torch.full((pad_size,), pad_token_id, device=device, dtype=torch.long)
                        gold = torch.cat([gold, pad_tensor])
                    elif gold_size > pred_size:
                        # gold 比 pred 长 → 截断
                        gold = gold[:pred_size]

                        
                    loss_seg = F.cross_entropy(pred, gold, reduction="sum", ignore_index=pad_token_id)
                    losses.append(loss_seg)
    
    
            if not losses:
                continue
    
            #loss = torch.stack(losses).sum() / gen_actual
            loss = torch.stack(losses).sum() / total_tokens
            
            
    
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
    
            total_loss += loss.item()
            if step % 50 == 0:
                accelerator.print(f"Epoch {epoch} Step {step} Loss {loss.item():.5f}")
    
        avg_loss = total_loss / (step + 1)
        accelerator.print(f"Epoch {epoch} AvgLoss {avg_loss:.5f}")


           # ---- optional validation ----

        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for input_ids, mask_pos_list, batch_raw in val_loader:
                input_ids = input_ids.to(device)
                outputs = model(input_ids)
                logits = outputs.logits  # [B, L, V]
        
                losses = []
                total_tokens = 0
                for i, ex in enumerate(batch_raw):
                    mask_ranges = mask_pos_list[i]
                    prompt_len = ex.get("prompt_len", 0)    
                    detailed_text = ex.get("detailed_process", "")
                    detailed_range = ex.get("detailed_range", [])   
                    semantic_block_lengths = ex.get("semantic_block_lengths", [gen_actual])
                    sample_tokens = sum(end - start for (start, end) in detailed_range)
                    total_tokens += sample_tokens

                    if not detailed_text:
                        continue
        
                    target_tokens = tokenizer(
                        detailed_text,
                        return_tensors="pt",
                        add_special_tokens=False
                    ).input_ids[0].to(device)
        
    
                    for j, ((mask_start, mask_end), (gold_start, gold_end)) in enumerate(zip(mask_ranges, detailed_range)):

                        gold = target_tokens[gold_start: gold_end]
                        pred = logits[i, mask_start: mask_end, :]
                        '''
                        
                        pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id  
                        if gold.size(0) < pred.size(0):
                            pad_len = pred.size(0) - gold.size(0)
                            gold = torch.cat([gold, torch.full((pad_len,), pad_token_id, device=device, dtype=torch.long)])
                        elif gold.size(0) > pred.size(0):
                            gold = gold[:pred.size(0)]
                            
        
                        if gold.numel() == 0 or pred.size(0) == 0:
                            continue
                        
                        '''


                        gold_size = gold.size(0)
                        pred_size = pred.size(0)
                                            
                        if gold_size < pred_size:
                            
                            pad_size = pred_size - gold_size
                            pad_tensor = torch.full((pad_size,), pad_token_id, device=device, dtype=torch.long)
                            gold = torch.cat([gold, pad_tensor])
                        elif gold_size > pred_size:
                           
                            gold = gold[:pred_size]
                        loss_seg = F.cross_entropy(pred, gold, reduction="sum", ignore_index=pad_token_id)
                        losses.append(loss_seg)
                    
                    '''
                    if i < 3 and epoch == 2:
                        gen_output = generate_parallel(
                            model,
                            prompt=input_ids,
                            steps=args.steps,             
                            gen_length=gen_actual,       
                            block_length=args.block_length,      
                            temperature=0.0,      
                            cfg_scale=0.0,        
                            remasking="low_confidence",
                            mask_id=mask_token_id,
                            semantic_block_mask_lengths=semantic_block_lengths, 
                        )
                    
                        decoded_text = tokenizer.decode(gen_output[0], skip_special_tokens=True)
                        print("Generated text:", decoded_text)
                        print("Target detailed text:", ex.get("detailed_process", ""))                        
        
                        '''

        
                if len(losses) == 0:
                    continue
        
                batch_loss = torch.stack(losses).sum() / total_tokens
                val_loss += batch_loss.item()
                val_steps += 1
        
        
        if val_steps > 0:
            accelerator.print(f"Epoch {epoch} Val Loss {val_loss / val_steps:.4f}")
    




    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
