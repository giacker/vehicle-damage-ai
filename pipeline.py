"""Reusable inference pipeline for the Vehicle Damage AI project."""
import cv2
import numpy as np
import pandas as pd
import torch
import joblib
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms
import torch.nn as nn


class DamagePipeline:
    def __init__(self, det_path, sev_path, cost_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.det_model = YOLO(det_path)

        sev_ckpt = torch.load(sev_path, map_location=self.device)
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 3)
        m.load_state_dict(sev_ckpt["state_dict"])
        self.sev_model = m.to(self.device).eval()
        self.sev_classes = sev_ckpt["classes"]

        bundle = joblib.load(cost_path)
        self.cost_model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]

        self.sev_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _classify_severity(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self.sev_tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.sev_model(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return self.sev_classes[idx], float(probs[idx])

    def _estimate_cost(self, damage_type, severity, area_pct):
        row = {c: 0 for c in self.feature_cols}
        row["area_pct"] = area_pct
        for key in (f"damage_type_{damage_type}", f"severity_{severity}"):
            if key in row: row[key] = 1
        x = pd.DataFrame([row])[self.feature_cols]
        return float(self.cost_model.predict(x)[0])

    def analyze(self, img_bgr, conf_threshold=0.25):
        H, W = img_bgr.shape[:2]
        img_area = H * W
        r = self.det_model.predict(img_bgr, conf=conf_threshold, verbose=False)[0]

        findings = []
        annotated = img_bgr.copy()
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            damage_type = self.det_model.names[int(cls)]
            crop = img_bgr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if crop.size == 0: continue

            severity, sev_conf = self._classify_severity(crop)
            area_pct = ((x2-x1) * (y2-y1)) / img_area
            cost = self._estimate_cost(damage_type, severity, area_pct)

            findings.append({
                "damage_type": damage_type,
                "severity": severity,
                "severity_conf": sev_conf,
                "detection_conf": float(conf),
                "area_pct": area_pct,
                "estimated_cost": cost,
                "bbox": (x1, y1, x2, y2),
            })

            color = {"minor": (0,200,0), "moderate": (0,165,255), "severe": (0,0,255)}[severity]
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            label = f"{damage_type} [{severity}] INR {cost:,.0f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(annotated, label, (x1+2, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        total_cost = sum(f["estimated_cost"] for f in findings)
        return annotated, findings, total_cost
