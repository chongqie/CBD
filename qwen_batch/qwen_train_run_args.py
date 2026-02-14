import subprocess
import os

EXPERIMENTS = [
    {"lr": 3e-6, "epochs": 5, "grad_acc": 1, "seed": 42},
    {"lr": 1e-6, "epochs": 8, "grad_acc": 1, "seed": 42},
    {"lr": 1e-5, "epochs": 3, "grad_acc": 1, "seed": 42},
    {"lr": 3e-6, "epochs": 5, "grad_acc": 2, "seed": 42},
    {"lr": 2e-6, "epochs": 6, "grad_acc": 2, "seed": 42},
]


BASE_DIR = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/joyce/qwen_train_version"
MODEL_PATH = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/joyce/qwen_2.5B"
DATA_PATH = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/joyce/data_latest/qwen_valid_1048.jsonl"
DRAFT_INPUT = "/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/joyce/data_latest/gsm8k_only_answer_train.jsonl"


for cfg in EXPERIMENTS:
    tag = f"lr{cfg['lr']}_ep{cfg['epochs']}"
    out_dir = os.path.join(BASE_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    subprocess.run([
        "python", "train_temp.py",
        "--model_path", MODEL_PATH,
        "--data_path", DATA_PATH,
        "--output_dir", out_dir,
        "--learning_rate", str(cfg["lr"]),
        "--num_train_epochs", str(cfg["epochs"]),
    ], check=True)

    ckpt = sorted(
        [d for d in os.listdir(out_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[-1])
    )[-1]

    subprocess.run([
        "python", "temp.py",
        "--model_path", os.path.join(out_dir, ckpt),
        "--input_file", DRAFT_INPUT,
        "--output_file", os.path.join(out_dir, "drafts.jsonl"),
    ], check=True)
