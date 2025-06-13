from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2 as cv
import os
from datetime import datetime
from ml.constants import CLASS_GROUP_MAP, CLASS_NAMES


CAPTURED_DIR = "captured_images"
os.makedirs(CAPTURED_DIR, exist_ok=True)


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
    Saves image with label and confidence in 'captured/' folder.
    """

    with torch.no_grad():
        image_tensor = preprocess_for_densenet(image_input) 
        outputs = densenet_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        class_name = CLASS_NAMES[predicted_class.item()]
        final_group = CLASS_GROUP_MAP.get(class_name, "Unknown")
        final_confidence = confidence.item()

    if final_group != "Unknown" and final_confidence >= threshold:
        if isinstance(image_input, str):
            image = cv.imread(image_input)
        else:
            image = image_input.copy()

        # Draw label box
        label = f"{final_group}: {final_confidence:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0) 

        text_size, _ = cv.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        margin = 10
        cv.rectangle(image, (margin, margin), (margin + text_w + 10, margin + text_h + 20), (0, 0, 0), -1)
        cv.putText(image, label, (margin + 5, margin + text_h + 10), font, font_scale, color, thickness)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CAPTURED_DIR, f"{final_group}_{timestamp}.jpg")
        cv.imwrite(save_path, image)

        return {
            "group": final_group,
            "confidence": final_confidence,
            "details": {
                final_group: final_confidence
            },
            "saved_path": save_path
        }

    return None
