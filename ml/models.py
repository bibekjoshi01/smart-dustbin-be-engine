import torch
import torch.nn as nn
from torchvision import models

from .constants import DENSENET_MODEL_PATH, DEVICE, YOLO_MODEL_PATH


def load_densenet201_model(checkpoint_path: str, num_classes: int = 2) -> nn.Module:
    """
    Load DenseNet201 model from a checkpoint and return it ready for inference.
    """
    model = models.densenet201(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_yolov5_model(checkpoint_path: str) -> nn.Module:
    """
    Load YOLOv5 model from a checkpoint and return it ready for inference.
    """
    model = None
    return model


def get_models():
    densenet = load_densenet201_model(DENSENET_MODEL_PATH)
    yolo = load_yolov5_model(YOLO_MODEL_PATH) 
    return densenet, yolo
