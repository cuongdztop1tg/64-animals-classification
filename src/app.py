import uvicorn
import torch
import torch.nn as nn
import io
import json
import torch.nn.functional as F

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from torchvision.transforms import transforms

from src.models.ResNet import ResNet

MODEL_NAME = "resnet"
MODEL_PATH = f"./models/best_{MODEL_NAME}_model.pth"
NUM_CLASSES = 64
IMG_SIZE = (128, 128)
DEVICE = "cpu"

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)


def create_model():
    # Loading checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    print(checkpoint["best_acc"])

    # Loading weight to model
    model = ResNet(input_dim=3, output_dim=NUM_CLASSES)
    model.load_state_dict(state_dict=checkpoint["model_state_dict"])
    model.eval()

    return model


model = create_model().to(DEVICE)

preprocess = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(
    title="Animal Classification API",
    description="API to classify animal images using a trained PyTorch model.",
    version="1.0.0",
)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/bmp", "image/gif"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image (JPEG, PNG, BMP, GIF).",
        )

    try:
        image_data = await file.read()

        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image_tensor = preprocess(image)

        image_batch = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image_batch)

        probs = F.softmax(logits[0], dim=0)
        pred_class_id = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_class_id]
        pred_prob = probs[pred_class_id].item()

        return {"predicted_class": pred_class, "probability": round(pred_prob, 2)}
        # probabilities = torch.nn.functional.softmax(logits[0], dim=0)
        # top3_prob, top3_id = torch.topk(probabilities, 3)

        # results = []
        # for i in range(3):
        #     results.append({
        #         "class": CLASS_NAMES[top3_id[i]],
        #         "prob": f"{top3_prob[i].item():.2%}"
        #     })
        # print(results)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {str(e)}"
        )


@app.get("/")
async def health_check():
    return {"status": "Server healthy and ready to run"}


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
