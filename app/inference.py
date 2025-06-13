import torch
import cv2
from torchvision import transforms
from ml.constants import CLASS_NAMES, CLASS_GROUP_MAP, DEVICE


def preprocess_for_densenet(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = transforms.ToPILImage()(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(DEVICE)
    return image_tensor


def make_inference(densenet_model, _, image_path, threshold=0.5):
    """
    Run inference using DenseNet only.
    """
    with torch.no_grad():
        image = preprocess_for_densenet(image_path)
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