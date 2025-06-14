from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2 as cv
import os
from datetime import datetime
from ml.constants import CLASS_GROUP_MAP, CLASS_NAMES, DEVICE
from collections import defaultdict


CAPTURED_DIR = "captured_images"
os.makedirs(CAPTURED_DIR, exist_ok=True)


def preprocess_image(input):
    if isinstance(input, str):
        image = Image.open(input).convert("RGB")
    elif isinstance(input, np.ndarray):  # image array
        image = Image.fromarray(cv.cvtColor(input, cv.COLOR_BGR2RGB))
    else:
        raise ValueError("Unsupported image input type.")

    return image


def preprocess_for_densenet(input):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = preprocess_image(input)
    return transform(image).unsqueeze(0).to(DEVICE)


def weighted_group_vote(
    results, class_names, class_group_map, threshold=0.5
) -> tuple[str, float, list]:
    """
    Returns the group with highest average confidence if above threshold and confidence.
    """

    all_boxes = []
    for r in results:
        all_boxes.extend(r.boxes)

    group_confidence_sum = defaultdict(float)
    group_count = defaultdict(int)

    for box in all_boxes:
        cls = int(box.cls[0])  # class index
        conf = float(box.conf[0])  # confidence score
        class_name = class_names[cls]
        group = class_group_map.get(class_name, "Unknown")  # O or R

        group_confidence_sum[
            group
        ] += conf  # sum of confidence of objects in each group
        group_count[group] += 1  # count of objects in each group

    if not group_confidence_sum:
        return "Unknown", 0.0, []

    # calculate average confidence per group normalized to 0-1
    group_avg_conf = {
        group: group_confidence_sum[group] / group_count[group]
        for group in group_confidence_sum
    }

    final_group = max(group_avg_conf, key=group_avg_conf.get)  # final group : O or R
    final_confidence = group_avg_conf[final_group]  # final confidence

    if final_confidence < threshold:
        return "Unknown", final_confidence

    return final_group, final_confidence, all_boxes


def get_results(densenet_model, yolo, image_input, threshold=0.5):

    results = yolo_inference(yolo, image_input)

    final_group, final_confidence, all_boxes = weighted_group_vote(
        results, CLASS_NAMES, CLASS_GROUP_MAP, threshold=threshold
    )

    if final_group == "Unknown":
        final_group, final_confidence = densenet_inference(densenet_model, image_input)
        all_boxes = None

    return final_group, final_confidence, all_boxes


def densenet_inference(densenet_model, image_input):
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

    return pred_class, confidence


def yolo_inference(yolo, image_input):
    image = preprocess_image(image_input)
    results = yolo.predict(image, imgsz=640)
    return results


def make_inference(densenet_model, yolo, image_input, threshold=0.5):
    """
    Run inference using DenseNet and yolo model.
    Accepts either image path or image array.
    Saves image with label and confidence in 'captured/' folder.
    """

    pred_class, confidence, all_boxes = get_results(
        densenet_model, yolo, image_input, threshold
    )

    image = preprocess_image(image_input)

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    if all_boxes:
        for box in all_boxes:
            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
            cls = int(box.cls[0])  # class index
            conf = box.conf[0]  # confidence score

            label = f"{CLASS_NAMES[cls]} {conf:.2f}"

            # Draw rectangle
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

            # Draw label background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle(
                [xmin, ymin - text_height, xmin + text_width, ymin], fill="red"
            )
            draw.text((xmin, ymin - text_height), label, fill="white", font=font)

    # Draw predicted group label
    label = f"{pred_class}: {confidence:.2f}"

    # Draw blue rectangle as background for label
    bbox = draw.textbbox((10, 10), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.rectangle([10, 10, 10 + text_width, 10 + text_height], fill="blue")
    draw.text((10, 10), label, fill="white", font=font)

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(CAPTURED_DIR, f"{pred_class}_{timestamp}.jpg")
    image.save(save_path)

    return {
        "group": pred_class,
        "confidence": confidence,
        "details": {pred_class: confidence},
        "saved_path": save_path,
    }
