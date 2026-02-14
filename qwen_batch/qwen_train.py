# -*- coding: utf-8 -*-
import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate structured drafts using a trained model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input jsonl file containing questions")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output jsonl file to save generated drafts")

    parser.add_argument("--num_samples", type=int, default=235,
                        help="Number of samples to generate drafts for")
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)

    return parser.parse_args()

def read_jsonl(file_path, num_samples):
    data = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    print(f"[read_jsonl] Error parsing line: {e}")
    return data[:num_samples]


def parse_generated_output(generated_text: str):
    summary = ""
    steps = []
    
    lines = generated_text.strip().split("\n")
    current_section = None
    current_step = []
    
    for line in lines:
        line = line.strip()
        
        if not line:
            continue
        
        
        if line.lower().startswith("summary:"):
            current_section = "summary"
            summary = line.split(":", 1)[1].strip() if ":" in line else ""
            continue
        
        if line.lower().startswith("steps:"):
            current_section = "steps"
            continue
        
    
        if "[DONE]" in line.upper() or "[END]" in line.upper():
            if current_step:
                steps.append(" ".join(current_step))
            break
        
        
        if current_section == "steps":
            import re
            step_match = re.match(r'^Step\s+\d+:', line, re.IGNORECASE)
            
            if step_match:
                
                if current_step:
                    steps.append(" ".join(current_step))
                    current_step = []
                
                current_step.append(line)
            else:
                
                if current_step:
                    current_step.append(line)
    
    
    if current_step:
        steps.append(" ".join(current_step))
    
    return {
        "summary": summary,
        "steps": steps
    }



def generate_draft(question, model, tokenizer, device, args):
    print(f"\nGenerating draft for question: {question[:100]}...")

    
    messages = [
        {
            "role": "system",
            "content": (
                "You are an assistant that generates structured draft solutions. "
                "Output format:\n"
                "Summary: [overview]\n\n"
                "Steps:\n"
                "Step 1: [detail]\n"
                "Step 2: [detail]\n\n"
                "[DONE]"
            )
        },
        {
            "role": "user",
            "content": f"Question: {question}"
        }
    ]
    
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  
    )
    
    
    if args.num_samples <= 5:  
        print(f"\n[DEBUG] Formatted prompt:")
        print(input_text)
        print()

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
        padding=False,
    ).to(device)
    
    print(f"[DEBUG] Input length: {inputs.input_ids.shape[1]}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True if args.temperature > 0 else False,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    
    print(f"[DEBUG] Generated {outputs.shape[1] - inputs.input_ids.shape[1]} tokens")
    
    
    print("========== RAW GENERATED OUTPUT ==========")
    print(generated_text)
    print("========== END RAW OUTPUT ==========\n")

    
    parsed = parse_generated_output(generated_text)
    
    return {
        "question": question,
        "generated_answer": generated_text,
        "summary": parsed["summary"],
        "steps": parsed["steps"]
    }



def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    print(f"[Info] Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    
    
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("[WARNING] Tokenizer has no chat_template!")
    else:
        print(f"[Info] Using Qwen2.5-Instruct chat format")

    data = read_jsonl(args.input_file, args.num_samples)
    print(f"[Info] Loaded {len(data)} samples")

    drafts = []
    for i, entry in enumerate(data):
        print(f"\n{'=' * 60}")
        print(f"Processing {i + 1}/{len(data)}")
        print(f"{'=' * 60}")

        question = entry.get("question", "")
        if not question:
            continue

        try:
            draft = generate_draft(question, model, tokenizer, device, args)
            drafts.append(draft)

            if (i + 1) % 10 == 0:
                print(f"\n>>> Progress: {i + 1}/{len(data)} completed\n")

        except Exception as e:
            print(f"[Error] Sample {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            drafts.append({
                "question": question,
                "generated_answer": "",
                "summary": "",
                "steps": [],
                "error": str(e),
            })

    with open(args.output_file, "w", encoding="utf-8") as f:
        for d in drafts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Generated {len(drafts)} drafts")
    print(f"Saved to: {args.output_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
