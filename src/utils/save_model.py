import torch
import torch.nn as nn
from pathlib import Path


def save_model(
    model: nn.Module,
    folder_path: str,
    model_name: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    best_acc: float = None,
):

    target_dir = Path(folder_path)

    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)

    save_path = target_dir / model_name

    if optimizer is not None:
        print(f"[INFO] Saving Full Checkpoint to: {save_path}")
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(obj=checkpoint, f=save_path)

    else:
        print(f"[INFO] Saving Model Weights to: {save_path}")
        torch.save(obj=model.state_dict(), f=save_path)
