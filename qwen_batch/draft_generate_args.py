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

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)

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


def build_prompt(question: str) -> str:
    """
    IMPORTANT:
    This prompt MUST stay strictly identical to the training prompt.
    """
    return (
        "You are an assistant tasked with generating a STRUCTURED DRAFT solution for a problem. "
        "Your output is NOT the final answer, but a clean, minimal outline consisting of a Summary and numbered Steps.\n\n"

        "Core Objective:\n"
        "- The Summary defines the EXACT solution structure.\n"
        "- The Steps MUST strictly follow and correspond to the Summary.\n\n"

        "Strict Alignment Rules:\n"
        "- The Summary must be written as a numbered list describing each logical step at a high level.\n"
        "- The Steps section must contain the SAME number of steps as the Summary.\n"
        "- Each step in Steps must correspond one-to-one with the step of the same number in Summary.\n"
        "- No extra steps, no missing steps, no merged or split steps are allowed.\n"
        "- The logical content of each Summary item and its corresponding Step must be aligned.\n\n"

        "Step Quality Constraints:\n"
        "- Each step must be necessary, distinct, and non-redundant.\n"
        "- Do NOT repeat information across different steps.\n"
        "- Do NOT add explanations beyond what is required to outline the solution.\n\n"

        "Output Format (STRICT):\n"
        "Summary: Total N steps: Step 1 - ...; Step 2 - ...; ...\n\n"
        "Steps:\n"
        "Step 1: ...\n"
        "Step 2: ...\n"
        "...\n\n"
        "[END]\n\n"

        f"Question: {question}\n\n"
        "Answer:"
    )


def parse_generated_output(generated_text: str):
    summary = ""
    steps = []

    lines = generated_text.strip().split("\n")
    current_section = None
    step_buffer = []

    for line in lines:
        line = line.strip()

        if line.startswith("Summary:"):
            current_section = "summary"
            summary = line.replace("Summary:", "").strip()

        elif line.startswith("Steps:"):
            current_section = "steps"

        elif line.startswith("Step ") and current_section == "steps":
            if step_buffer:
                steps.append(" ".join(step_buffer))
                step_buffer = []
            step_buffer.append(line)

        elif line == "[END]":
            if step_buffer:
                steps.append(" ".join(step_buffer))
            break

        elif current_section == "steps" and line:
            if step_buffer:
                step_buffer.append(line)

    if step_buffer and "[END]" not in step_buffer:
        steps.append(" ".join(step_buffer))

    return {
        "summary": summary,
        "steps": steps
    }


def generate_draft(question, model, tokenizer, device, args):
    print(f"\nGenerating draft for question: {question[:100]}...")

    input_text = build_prompt(question)
 
    
    print("\n[DEBUG] Prompt length:", len(input_text))
    print("[DEBUG] First 200 chars of prompt:")
    print(input_text[:200])    
    print("[DEBUG] Last 100 chars of prompt:")
    print(input_text[-100:])
   
   inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=args.max_input_length,
        truncation=True,
        padding=False,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )


    print("========== RAW GENERATED OUTPUT ==========")
    print(generated_text)
    print("========== END RAW OUTPUT ==========\n")

    
    return {
        "question": question,
        "generated_answer": generated_text
        }

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

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
                print(f">>> Progress: {i + 1}/{len(data)} completed")

        except Exception as e:
            print(f"[Error] Sample {i + 1}: {e}")
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

    print(f"\nGenerated {len(drafts)} drafts")
    print(f"Saved to: {args.output_file}")

if __name__ == "__main__":
    main()


