import torch
import time
import torch.nn as nn
import src.core.config as config

from src.models.ResNet import ResNet
from src.utils.save_model import save_model
from src.utils.set_random_seed import seed_everything
from src.core.data_loader import ImageDataLoader


def train(model: nn.Module, save_name: str):
    # Set random seed
    print(f"Set random seed={config.RANDOM_SEED}")
    seed_everything(seed=config.RANDOM_SEED)

    # Get data loaders
    print(f"Initialize data loaders")
    train_loader, test_loader = ImageDataLoader(
        train_dir=config.PROCESSED_DATA_PATH + "/train",
        test_dir=config.PROCESSED_DATA_PATH + "/test",
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        num_workers=2,
    ).get_loader()

    # Setup optimizer and loss function
    print("Setup optimizer and loss function")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)
    loss_fn = nn.CrossEntropyLoss()

    train_history = {"loss": [], "acc": []}
    test_history = {"loss": [], "acc": []}

    best_acc = 0.0

    print("Start training...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        # ===================== TRAIN =====================
        model.train()

        train_loss_total = 0.0
        train_correct_total = 0
        train_samples_count = 0

        for X, y in train_loader:
            X = X.to(config.DEVICE)
            y = y.to(config.DEVICE)

            batch_size = X.size(0)

            # Forward
            logits = model(X)
            loss = loss_fn(logits, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            preds = logits.argmax(dim=1)

            train_loss_total += loss.item() * batch_size
            train_correct_total += (preds == y).sum().item()
            train_samples_count += batch_size

        epoch_train_loss = train_loss_total / train_samples_count
        epoch_train_acc = train_correct_total / train_samples_count

        train_history["loss"].append(epoch_train_loss)
        train_history["acc"].append(epoch_train_acc)

        print(f"Epoch [{epoch+1}/{config.EPOCHS}]")
        print(
            f"Train loss: {epoch_train_loss:.5f} | Train acc: {epoch_train_acc*100:.2f}%"
        )
        end_time = time.time()

        # ===================== TEST =====================
        model.eval()

        test_loss_total = 0.0
        test_correct_total = 0
        test_samples_count = 0

        with torch.inference_mode():
            for X, y in test_loader:
                X = X.to(config.DEVICE)
                y = y.to(config.DEVICE)

                batch_size = X.size(0)

                logits = model(X)
                loss = loss_fn(logits, y)

                preds = logits.argmax(dim=1)

                test_loss_total += loss.item() * batch_size
                test_correct_total += (preds == y).sum().item()
                test_samples_count += batch_size

        epoch_test_loss = test_loss_total / test_samples_count
        epoch_test_acc = test_correct_total / test_samples_count

        test_history["loss"].append(epoch_test_loss)
        test_history["acc"].append(epoch_test_acc)

        print(
            f"Test loss: {epoch_test_loss:.5f} | Test acc: {epoch_test_acc*100:.2f}%\n"
        )

        print(f"Training time: {(end_time - start_time):.2f}s")

        # ===================== SAVE BEST =====================
        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            print("*** New best model, saving...")
            save_model(
                model=model,
                folder_path=config.RESULT_PATH,
                model_name=save_name,
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc,
            )

        torch.cuda.empty_cache()


if __name__ == "__main__":
    model = ResNet(input_dim=3, output_dim=64).to(config.DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    untrainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"untrainable parameters: {untrainable_params}")

    train(model=model, save_name="test.pth")
