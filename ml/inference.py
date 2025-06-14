from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2 as cv
import os
from datetime import datetime
from ml.constants import CLASS_GROUP_MAP, CLASS_NAMES, DEVICE


CAPTURED_DIR = "captured_images"
os.makedirs(CAPTURED_DIR, exist_ok=True)


def preprocess_for_densenet(input):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(input, str):
        image = Image.open(input).convert("RGB")
    elif isinstance(input, np.ndarray):  # image array
        image = Image.fromarray(cv.cvtColor(input, cv.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image input type.")

    return transform(image).unsqueeze(0).to(DEVICE)


def make_inference(densenet_model, _, image_input, threshold=0.5):
    """
    Run inference using DenseNet only.
    Accepts either image path or image array.
    Saves image with label and confidence in 'captured/' folder.
    """
    densenet_class_names = ["O", "R"]

    with torch.no_grad():
        image_tensor = preprocess_for_densenet(image_input)
        outputs = densenet_model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class_idx = torch.argmax(
            probs, dim=1
        ).item()  # class index with highest probability
        pred_class = densenet_class_names[pred_class_idx]  # classname from index
        confidence = probs[0, pred_class_idx].item()

        if isinstance(image_input, str):
            image = cv.imread(image_input)
        else:
            image = image_input.copy()

        # Draw label box
        label = f"{pred_class}: {confidence:.2f}"
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0)

        text_size, _ = cv.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size
        margin = 10
        cv.rectangle(
            image,
            (margin, margin),
            (margin + text_w + 10, margin + text_h + 20),
            (0, 0, 0),
            -1,
        )
        cv.putText(
            image,
            label,
            (margin + 5, margin + text_h + 10),
            font,
            font_scale,
            color,
            thickness,
        )

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(CAPTURED_DIR, f"{pred_class}_{timestamp}.jpg")
        cv.imwrite(save_path, image)

        return {
            "group": pred_class,
            "confidence": confidence,
            "details": {pred_class: confidence},
            "saved_path": save_path,
        }

    return None
