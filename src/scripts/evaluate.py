import torch
import matplotlib.pyplot as plt
import seaborn as sns
import src.core.config as config

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

from src.utils.set_random_seed import seed_everything
from src.models.ResNet import ResNet

# --- 1. CONFIG  ---
seed_everything(config.RANDOM_SEED)
MODEL_PATH = config.RESULT_PATH + "/best_resnet_model.pth"
TEST_DIR = config.PROCESSED_DATA_PATH + "/train"

# --- 2. PREPARE DATA ---
test_transform = transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

print(f"--> Loading test data from: {TEST_DIR}")
try:
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    class_names = test_dataset.classes
    print(f"--> Found {len(class_names)} classes.")
except FileNotFoundError:
    print(f"Error: cannot find test data folder at {TEST_DIR}")
    exit()


# --- 3. LOAD MODEL ---
def load_model_architecture():
    print("--> Initializing Model Architecture...")
    # Loading checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=config.DEVICE)

    # Loading weight to model
    model = ResNet(input_dim=3, output_dim=config.NUM_CLASSES)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    model.eval()
    return model


model = load_model_architecture().to(config.DEVICE)

# --- 4. EVALUATION LOOP ---
y_true = []
y_pred = []

print("--> Begin evaluation...")
with torch.no_grad():
    for inputs, labels in tqdm(test_loader, desc="Testing"):
        inputs = inputs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# --- 5. METRICS & REPORT ---
print("\n" + "=" * 30)
print("EVALUATION RESULT")
print("=" * 30)

acc = accuracy_score(y_true, y_pred)
print(f"✅ Accuracy Score: {acc:.4f} ({acc*100:.2f}%)")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 12))

    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title(f"Confusion Matrix (Acc: {acc:.2%})")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print(f"\n🖼️  Saved confusion matrix as: confusion_matrix.png")
    plt.show()


plot_confusion_matrix(y_true, y_pred, class_names)
