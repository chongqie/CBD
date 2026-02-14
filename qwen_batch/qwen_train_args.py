import os
import json
import glob
import argparse
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from rouge_score import rouge_scorer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")

    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    return parser.parse_args()



def read_jsonl(path):
    all_data = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.jsonl"))
    else:
        raise ValueError(f"Invalid path: {path}")

    for file in files:
        print(f"[read_jsonl] Loading {file}")
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
    return all_data


def create_dataset(data, tokenizer, max_length=512):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        inputs, labels = [], []

        for q, s, st in zip(
            examples["question"], examples["summary"], examples["steps"]
        ):
            prompt = (
                "You are an assistant tasked with generating a STRUCTURED DRAFT solution for a problem. "
                "Your output is NOT the final answer, but a clean, minimal outline consisting of a Summary and numbered Steps.\n\n"
                "Core Objective:\n"
                "- The Summary defines the EXACT solution structure.\n"
                "- The Steps MUST strictly follow and correspond to the Summary.\n\n"
                "Strict Alignment Rules:\n"
                "- The Summary must be written as a numbered list describing each logical step at a high level.\n"
                "- The Steps section must contain the SAME number of steps as the Summary.\n"
                "- Each step in Steps must correspond one-to-one with the step of the same number in Summary.\n"
                "- No extra steps, no missing steps, no merged or split steps are allowed.\n\n"
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
                f"Question: {q}\n\n"
                "Answer:"
            )

            target = (
                f"Summary: {s}\n\n"
                "Steps:\n"
                + "\n".join(st)
                + "\n\n[END]"
            )

            full_text = prompt + " " + target

            tokenized = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )

            input_ids = tokenized["input_ids"]
            labels_ids = input_ids.copy()

            prompt_len = len(
                tokenizer(prompt + " ", add_special_tokens=True)["input_ids"]
            )
            labels_ids[:prompt_len] = [-100] * prompt_len

            inputs.append(input_ids)
            labels.append(labels_ids)

        return {
            "input_ids": inputs,
            "attention_mask": [
                [1 if t != tokenizer.pad_token_id else 0 for t in ids]
                for ids in inputs
            ],
            "labels": labels,
        }

    dataset = Dataset.from_dict({
        "question": [d["question"] for d in data],
        "summary": [d["summary"] for d in data],
        "steps": [d["steps"] for d in data],
    })

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    dataset.set_format(type="torch")
    return dataset


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_metrics(eval_preds, tokenizer):
    predictions, labels = eval_preds
    preds_ids = np.argmax(predictions, axis=-1)
    label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)

    pred_strs = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    label_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for p, l in zip(pred_strs, label_strs):
        s = scorer.score(p.strip(), l.strip())
        for k in scores:
            scores[k].append(s[k].fmeasure)

    return {k: float(np.mean(v)) for k, v in scores.items()}


class EvalLossTracker(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            with open(os.path.join(args.output_dir, "eval_losses.txt"), "a") as f:
                f.write(f"{state.global_step}\t{metrics['eval_loss']}\n")


def compare_weights(original_model_path, trained_model):
    original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
    w0 = original_model.model.layers[0].self_attn.q_proj.weight.data.cpu()
    w1 = trained_model.model.layers[0].self_attn.q_proj.weight.data.cpu()
    diff = torch.norm(w0 - w1).item()
    print(f"Weight diff: {diff}")
    return diff

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    data = read_jsonl(args.data_path)
    dataset = create_dataset(data, tokenizer)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=args.fp16,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        #compute_metrics=lambda x: compute_metrics(x, tokenizer),
        callbacks=[EvalLossTracker()],
    )

    trainer.train()
    compare_weights(args.model_path, model)
