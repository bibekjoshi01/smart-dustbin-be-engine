import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet201_Weights
from ultralytics import YOLO


from .constants import DENSENET_MODEL_PATH, DEVICE, YOLO_MODEL_PATH


def load_densenet201_model(checkpoint_path: str, num_classes: int = 2) -> nn.Module:
    """
    Load DenseNet201 model from a checkpoint and return it ready for inference.
    """
    weights = DenseNet201_Weights.DEFAULT
    model = models.densenet201(weights=weights)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),  # 2 classes: O, R
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_yolov8s_model(checkpoint_path: str) -> YOLO:
    """
    Load YOLOv5 model from a checkpoint and return it ready for inference.
    """
    model = YOLO(checkpoint_path)
    return model


def get_models():
    densenet = load_densenet201_model(DENSENET_MODEL_PATH)
    yolo = load_yolov8s_model(YOLO_MODEL_PATH)
    return densenet, yolo
