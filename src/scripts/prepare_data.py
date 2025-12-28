import shutil
import random
import src.core.config as config

from pathlib import Path

from src.utils.set_random_seed import seed_everything


def prepare_data(source_dir: str, output_dir: str, train_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if output_path.exists():
        print("Processed data was already created, skip splitting data.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    for class_dir in source_path.iterdir():
        if not class_dir.is_dir():
            continue

        images = list(class_dir.glob("*"))
        random.shuffle(images)

        split_len = int(len(images) * train_ratio)
        train_images = images[:split_len]
        test_images = images[split_len:]

        output_train_path = output_path / "train" / class_dir.name
        output_test_path = output_path / "test" / class_dir.name

        output_train_path.mkdir(parents=True, exist_ok=True)
        output_test_path.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy(img, output_train_path / img.name)

        for img in test_images:
            shutil.copy(img, output_test_path / img.name)

        print(f"{class_dir.name}: {len(train_images)} train | {len(test_images)} test")


if __name__ == "__main__":
    source_dir = config.RAW_DATA_PATH
    output_dir = config.PROCESSED_DATA_PATH

    seed_everything(config.RANDOM_SEED)
    prepare_data(source_dir, output_dir)
