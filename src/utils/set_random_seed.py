import random
import os
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    def generate_random_values():
        # A. Python Native
        py_rand = random.randint(0, 100)

        # B. Numpy
        np_rand = np.random.rand(1)[0]

        # C. PyTorch Tensor (CPU/GPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_rand = torch.rand(1).to(device).item()

        layer = nn.Linear(1, 1).to(device)
        weight_init = layer.weight.item()

        return py_rand, np_rand, torch_rand, weight_init

    print("--- Test repoducibility ---\n")

    seed_everything(42)
    res1 = generate_random_values()
    print(f"Take 1 (Seed 42): {res1}")

    seed_everything(42)
    res2 = generate_random_values()
    print(f"Take 2 (Seed 42): {res2}")

    seed_everything(999)
    res3 = generate_random_values()
    print(f"Take 3 (Seed 99): {res3}")

    print("\n--- Result ---")

    is_match = res1 == res2
    if is_match:
        print("✅ PASS: Take 1 and 2 similar")
    else:
        print("❌ FAIL: Take 1 and 3 different")

    is_diff = res1 != res3
    if is_diff:
        print("✅ PASS: Take 1 and 3 different.")
    else:
        print("❌ FAIL: Take 1 and 3 similar")
