"""Model evaluation script for few-shot face recognition architectures.

Compares FaceNet (InceptionResnetV1) against traditional classification
architectures (VGG16, ResNet50, MobileNetV2) as feature extractors.

Outputs Sensitivity, Specificity, and Accuracy.
"""

from __future__ import annotations

import os
import argparse
from pathlib import Path
import random
from typing import List, Tuple, Dict
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import accuracy_score, confusion_matrix
from prettytable import PrettyTable

from utils import get_identity_folders, load_face_detector, detect_faces, _normalize_face

# Fix DLL issue for Windows
if os.name == "nt":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BaselineEmbedder:
    """Wrapper for traditional PyTorch classification models as feature extractors."""
    
    def __init__(self, model_name: str, target_size: int = 224):
        self.model_name = model_name
        self.target_size = target_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load requested architecture
        if model_name == "VGG-16":
            base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # Remove classification head, keep features
            self.model = base.features
            
        elif model_name == "ResNet-50":
            base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            # Remove fc layer
            self.model = nn.Sequential(*list(base.children())[:-1])
            
        elif model_name == "MobileNet":
            base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.model = base.features
            
        else:
            raise ValueError(f"Unknown baseline: {model_name}")
            
        self.model = self.model.eval().to(self.device)

    def embed(self, face_rgb: np.ndarray) -> np.ndarray:
        # Resize to standard ImageNet size
        resized = cv2.resize(face_rgb, (self.target_size, self.target_size))
        # ImageNet normalization standard
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        
        with torch.no_grad():
            features = self.model(tensor)
            # Global Average Pooling if needed
            if len(features.shape) > 2:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
            
            embedding = features.cpu().numpy()[0].astype(np.float32)
            
        # L2 Normalize
        embedding /= np.linalg.norm(embedding) + 1e-8
        return embedding


def prepare_dataset_split(dataset_dir: str, train_ratio: float = 0.5) -> Tuple[Dict, Dict]:
    """Splits available data into Train (prototypes) and Test (queries)."""
    folders = get_identity_folders(dataset_dir)
    detector = load_face_detector()
    
    train_data = {}
    test_data = {}
    
    print(f"Loading and splitting dataset from {dataset_dir} (Train Ratio: {train_ratio*100:.0f}%)")
    
    total_train = 0
    total_test = 0
    
    for folder in folders:
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        valid_faces = []
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            dets = detect_faces(img, detector, min_confidence=0.8)
            if dets:
                largest = max(dets, key=lambda d: (d.box[2]-d.box[0])*(d.box[3]-d.box[1]))
                valid_faces.append(largest.face_rgb)
                
        if len(valid_faces) < 2:
            print(f"Skipping {folder.name} (Need at least 2 faces to split train/test)")
            continue
            
        random.seed(42) # Deterministic shuffle
        random.shuffle(valid_faces)
        
        split_idx = max(1, int(len(valid_faces) * train_ratio))
        train_data[folder.name] = valid_faces[:split_idx]
        test_data[folder.name] = valid_faces[split_idx:]
        
        total_train += len(train_data[folder.name])
        total_test += len(test_data[folder.name])
        
    print(f"Found {len(train_data)} usable identities.")
    print(f"Train samples (Prototypes): {total_train}")
    print(f"Test samples (Queries): {total_test}\n")
    
    return train_data, test_data


def evaluate_model(model_name: str, embedder, train_data: Dict, test_data: Dict) -> Dict:
    """Evaluates a single model architecture on the provided test set."""
    print(f"Evaluating {model_name}...")
    start_time = time.time()
    
    # 1. Build Prototypes (Means)
    prototypes = {}
    for name, faces in train_data.items():
        if model_name == "FaceNet (Ours)":
            embeddings = [embedder.embed_face(f) for f in faces]
        else:
            embeddings = [embedder.embed(f) for f in faces]
        prototype = np.mean(embeddings, axis=0)
        prototype /= np.linalg.norm(prototype) + 1e-8
        prototypes[name] = prototype
        
    class_names = list(prototypes.keys())
    proto_matrix = np.array([prototypes[n] for n in class_names])
    
    # 2. Test Set Predictions
    y_true = []
    y_pred = []
    
    for true_name, faces in test_data.items():
        for face in faces:
            if model_name == "FaceNet (Ours)":
                embedding = embedder.embed_face(face)
            else:
                embedding = embedder.embed(face)
            
            # Cosine similarity against all prototypes
            similarities = np.dot(proto_matrix, embedding)
            pred_idx = np.argmax(similarities)
            pred_name = class_names[pred_idx]
            
            y_true.append(true_name)
            y_pred.append(pred_name)
            
    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    
    # Calculate macro Sensitivity and Specificity via Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    
    sensitivities = []
    specificities = []
    
    for i in range(len(class_names)):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        
    avg_sensitivity = np.mean(sensitivities)
    avg_specificity = np.mean(specificities)
    
    elapsed = time.time() - start_time
    print(f"  > Looked at {len(y_true)} queries in {elapsed:.1f}s")
    
    return {
        "Model": model_name,
        "Specificity": avg_specificity * 100,
        "Sensitivity": avg_sensitivity * 100,
        "Accuracy": acc * 100
    }

def main():
    parser = argparse.ArgumentParser("Evaluate Face Recognition Models")
    parser.add_argument("--dataset", default="dataset", help="Path to enrollment images")
    parser.add_argument("--train-ratio", type=float, default=0.4, help="Ratio of images to use for training vs testing")
    args = parser.parse_args()
    
    # 1. Prepare Data
    train_data, test_data = prepare_dataset_split(args.dataset, train_ratio=args.train_ratio)
    
    if not train_data or not test_data:
        print("Not enough data to run evaluation. Need folders with multiple images.")
        return
        
    results = []
    from utils import FaceEmbedder
    
    # 2. Evaluate Models
    # FaceNet (The one we use currently!)
    facenet = FaceEmbedder(backend="facenet")
    res = evaluate_model("FaceNet (Ours)", facenet, train_data, test_data)
    results.append(res)
    
    # Baseline: VGG-16
    vgg16 = BaselineEmbedder("VGG-16")
    res = evaluate_model("VGG-16", vgg16, train_data, test_data)
    results.append(res)
    
    # Baseline: ResNet-50
    resnet = BaselineEmbedder("ResNet-50")
    res = evaluate_model("ResNet-50", resnet, train_data, test_data)
    results.append(res)
    
    # Baseline: MobileNet
    mobilenet = BaselineEmbedder("MobileNet")
    res = evaluate_model("MobileNet", mobilenet, train_data, test_data)
    results.append(res)
    
    # 3. Print Table
    print("\n\n")
    print("Table 1: Comparison of traditional deep learning models for classification (Your Dataset)")
    table = PrettyTable()
    table.field_names = ["Model", "Specificity (%)", "Sensitivity (%)", "Accuracy (%)"]
    
    for r in results:
        table.add_row([
            r["Model"], 
            f"{r['Specificity']:.2f}", 
            f"{r['Sensitivity']:.2f}", 
            f"{r['Accuracy']:.2f}"
        ])
        
    print(table)


if __name__ == "__main__":
    main()
