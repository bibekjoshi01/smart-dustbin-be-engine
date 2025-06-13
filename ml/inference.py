from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

from ml.constants import CLASS_GROUP_MAP, CLASS_NAMES


def preprocess_for_densenet(input):
    transform = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if isinstance(input, str):
        image = Image.open(input).convert("RGB")
    elif isinstance(input, np.ndarray):  # image array
        image = input
    else:
        raise ValueError("Unsupported image input type.")

    return transform(image).unsqueeze(0)


def make_inference(densenet_model, _, image_input, threshold=0.5):
    """
    Run inference using DenseNet only.
    Accepts either image path or image array.
    """
    with torch.no_grad():
        image = preprocess_for_densenet(image_input)
        outputs = densenet_model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        class_name = CLASS_NAMES[predicted_class.item()]
        final_group = CLASS_GROUP_MAP.get(class_name, "Unknown")
        final_confidence = confidence.item()

    if final_group != "Unknown" and final_confidence >= threshold:
        return {
            "group": final_group,
            "confidence": final_confidence,
            "details": {
                final_group: final_confidence
            }
        }

    return None
