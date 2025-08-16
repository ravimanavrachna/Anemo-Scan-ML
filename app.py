from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, os, tempfile
from datetime import datetime
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image
from ultralytics import YOLO
import mediapipe as mp
import os
from torchvision.models import mobilenet_v2
from ultralytics import YOLO

# =========================
# App & Device
# =========================
app = Flask(__name__)
CORS(app, origins=["https://anemoscan.healthinnovations.in"], supports_credentials=True, allow_headers=["Content-Type", "Authorization"], methods=["GET", "POST", "OPTIONS"])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Utility
# =========================
def image_to_base64(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def majority_label(labels, positive="anemia", negative="non-anemia"):
    pos = sum(1 for x in labels if x == positive)
    neg = sum(1 for x in labels if x == negative)
    if pos > neg:
        return positive
    if neg > pos:
        return negative
    # tie → prefer non-anemia (or choose any policy you like)
    return negative

# =========================
# 1) EYE: YOLO (seg or bbox) + MobileNetV2 (binary)
# =========================
EYE_YOLO_MODEL_PATH = "./models/best_10aug2025_eye.pt"     # segmentation or bbox
EYE_CLASSIFIER_PATH = "./models/mobilenetv2_conjunctiva.pth"
EYE_CLASS_TARGET = 0  # conjunctiva class id

yolo_eye_model = YOLO(EYE_YOLO_MODEL_PATH)

eye_classifier = models.mobilenet_v2(weights=None)
# Replace the classifier head with a single logit for binary (sigmoid later)
eye_classifier.classifier = nn.Sequential(nn.Linear(eye_classifier.last_channel, 1))
eye_classifier.load_state_dict(torch.load(EYE_CLASSIFIER_PATH, map_location=DEVICE), strict=False)
eye_classifier.to(DEVICE).eval()

eye_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# chek

def process_eye(image_path: str, save_dir: str):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"cropped": None, "confidence": 0.0, "prediction": "error", "success": False}

        h, w = img.shape[:2]
        results = yolo_eye_model(image_path, imgsz=224, verbose=False)

        cropped_image = None
        # Prefer segmentation masks when available; else fall back to bbox
        for r in results:
            # Segmentation path
            if getattr(r, "masks", None) is not None and r.masks is not None and r.masks.data is not None:
                masks = r.masks.data.cpu().numpy()  # (N,H,W) in model space
                # Filter by class if boxes have cls
                for i, mask in enumerate(masks):
                    cls = int(r.boxes.cls[i].cpu().numpy()) if len(r.boxes) > i else None
                    if cls is not None and cls != EYE_CLASS_TARGET:
                        continue
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    nonz = cv2.findNonZero(mask_resized)
                    if nonz is None:
                        continue
                    x, y, bw, bh = cv2.boundingRect(nonz)
                    cropped_masked = cv2.bitwise_and(img, img, mask=mask_resized)
                    cropped_image = cropped_masked[y:y+bh, x:x+bw]
                    break
                if cropped_image is not None:
                    break

            # BBox path
            if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes) > 0 and cropped_image is None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(xyxy))
                # Choose first bbox with target class
                chosen_idx = None
                for i, cls in enumerate(clses):
                    if int(cls) == EYE_CLASS_TARGET:
                        chosen_idx = i
                        break
                # If no target class found, just take the highest-conf box
                if chosen_idx is None:
                    chosen_idx = 0
                x1, y1, x2, y2 = map(int, xyxy[chosen_idx])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    cropped_image = img[y1:y2, x1:x2]

        if cropped_image is None:
            return {"cropped": None, "confidence": 0.0, "prediction": "no_mask_or_box_detected", "success": False}

        cropped_path = os.path.join(save_dir, f"cropped_eye_{os.path.basename(image_path)}")
        cv2.imwrite(cropped_path, cropped_image)

        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        tensor = eye_transform(cropped_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logit = eye_classifier(tensor)
            prob = torch.sigmoid(logit).item()  # probability of class=1 (Anemia)
        pred_class = 1 if prob > 0.5 else 0
        class_names = ["non-anemia", "anemia"]

        return {
            "cropped": cropped_path,
            "confidence": round(prob * 100, 2),
            "prediction": class_names[pred_class],
            "success": True
        }
    except Exception as e:
        return {"cropped": None, "confidence": 0.0, "prediction": "error", "success": False, "error": str(e)}

# =========================
# 2) PALM: MediaPipe crop + MobileNetV2 (binary)
# =========================
PALM_MODEL_PATH = "./models/mobilenetv2_finetuned_palm.pth"

palm_model = models.mobilenet_v2(weights=None)
palm_model.classifier[1] = nn.Linear(palm_model.last_channel, 2)
palm_model.load_state_dict(torch.load(PALM_MODEL_PATH, map_location=DEVICE), strict=False)
palm_model.to(DEVICE).eval()

palm_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_palm(image_path: str, margin=20):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None
        h, w, _ = image.shape
        landmark_ids = [0, 1, 5, 9, 13, 17]
        hand_landmarks = results.multi_hand_landmarks[0]
        coords = [(int(hand_landmarks.landmark[i].x * w),
                   int(hand_landmarks.landmark[i].y * h)) for i in landmark_ids]
        x_vals, y_vals = zip(*coords)
        x_min = max(min(x_vals) - margin, 0)
        x_max = min(max(x_vals) + margin, w)
        y_min = max(min(y_vals) - margin, 0)
        y_max = min(max(y_vals) + margin, h)
        if x_max <= x_min or y_max <= y_min:
            return None
        palm_crop = image[y_min:y_max, x_min:x_max]
        return cv2.cvtColor(palm_crop, cv2.COLOR_BGR2RGB)

def process_palm(image_path: str, save_dir: str):
    try:
        palm_rgb = extract_palm(image_path)
        if palm_rgb is None:
            return {"cropped": None, "confidence": 0.0, "prediction": "no_hand_detected", "success": False}

        cropped_path = os.path.join(save_dir, f"cropped_palm_{os.path.basename(image_path)}")
        cv2.imwrite(cropped_path, cv2.cvtColor(palm_rgb, cv2.COLOR_RGB2BGR))

        palm_pil = Image.fromarray(palm_rgb)
        tensor = palm_transform(palm_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = palm_model(tensor)
            probs = torch.softmax(out, dim=1)
            cls = int(torch.argmax(probs).item())
            conf = float(probs[0][cls].item())
        prediction = "anemia" if cls == 1 else "non-anemia"

        return {
            "cropped": cropped_path,
            "confidence": round(conf * 100, 2),
            "prediction": prediction,
            "success": True
        }
    except Exception as e:
        return {"cropped": None, "confidence": 0.0, "prediction": "error", "success": False, "error": str(e)}

# =========================
# 3) NAILS: YOLO (bbox) + SimpleCNN (binary), per-nail details (up to 10)
# =========================


# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAIL_YOLO_PATH = "./models/yolov8_nail_best.pt"
NAIL_MODEL_PATH = "./models/mobilenetv2_nailbeds.pth"
CLASS_NAMES = ["anemia", "non-anemia"]

# ============================================================
# LOAD MODELS
# ============================================================
yolo_nail_model = YOLO(NAIL_YOLO_PATH)

nail_classifier = mobilenet_v2(weights=None)  # No pretrained weights
nail_classifier.classifier[1] = nn.Linear(nail_classifier.last_channel, len(CLASS_NAMES))
nail_classifier.load_state_dict(torch.load(NAIL_MODEL_PATH, map_location=DEVICE))
nail_classifier = nail_classifier.to(DEVICE).eval()

# ============================================================
# TRANSFORMS
# ============================================================
nail_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# HELPERS
# ============================================================
def safe_float(x):
    try:
        return float(x)
    except:
        return 0.0

def majority_label(labels):
    """Return majority label among predictions."""
    if not labels:
        return "non-anemia"
    return max(set(labels), key=labels.count)

# ============================================================
# MAIN FUNCTION
# ============================================================
def process_nail(image_path: str, save_dir: str):
    """
    Detect nails, classify anemia, and return structured results.
    """
    try:
        image_pil = Image.open(image_path).convert("RGB")
        results = yolo_nail_model(image_pil)

        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return {
                "per_nail": [],
                "overall_prediction": "non-anemia",
                "overall_confidence": 0.0,
                "success": False
            }

        boxes = results[0].boxes.xyxy.cpu().numpy()
        det_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else np.zeros(len(boxes))

        # Take top 10 nails by detection confidence
        order = np.argsort(det_confs)[::-1]
        boxes = boxes[order][:10]
        det_confs = det_confs[order][:10]

        per_nail, anemia_preds, anemia_confs = [], [], []

        for i, (box, det_c) in enumerate(zip(boxes, det_confs), start=1):
            x1, y1, x2, y2 = map(int, box)
            nail_crop = image_pil.crop((x1, y1, x2, y2))

            # Save cropped nail
            cropped_path = os.path.join(save_dir, f"cropped_nail_{i}_{os.path.basename(image_path)}")
            os.makedirs(save_dir, exist_ok=True)
            nail_crop.save(cropped_path)

            # Predict anemia with MobileNetV2
            tensor = nail_transform(nail_crop).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = nail_classifier(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, pred_class = torch.max(probs, 1)

            predicted_label = CLASS_NAMES[pred_class.item()]
            confidence = conf.item()

            per_nail.append({
                "nail_id": i,
                "path": cropped_path,
                "det_conf": round(safe_float(det_c) * 100, 2),
                "anemia_conf": round(confidence * 100, 2),
                "anemia_pred": predicted_label
            })

            anemia_preds.append(predicted_label)
            anemia_confs.append(confidence)

        overall_pred = majority_label(anemia_preds)
        overall_conf = round(np.mean(anemia_confs) * 100, 2) if anemia_confs else 0.0

        return {
            "per_nail": per_nail,
            "overall_prediction": overall_pred,
            "overall_confidence": overall_conf,
            "success": True
        }

    except Exception as e:
        return {
            "per_nail": [],
            "overall_prediction": "non-anemia",
            "overall_confidence": 0.0,
            "success": False,
            "error": str(e)
        }

# =========================
# Flask Endpoint
# =========================
@app.route('/predict_anemia', methods=['POST'])
def predict_anemia():
    """
    Expects form-data files:
      - nail_image, left_palm, right_palm, left_eye, right_eye
    Returns JSON with 14 base64 images (2 eyes + 2 palms + up to 10 nails) and predictions.
    """
    try:
        files = request.files
        required_fields = ['nail_image', 'left_palm', 'right_palm', 'left_eye', 'right_eye']
        if not all(k in files for k in required_fields):
            return jsonify({
                "error": "Missing images",
                "message": "Provide all: nail_image, left_palm, right_palm, left_eye, right_eye",
                "success": False
            }), 400

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploads
            paths = {}
            for key in required_fields:
                f = files[key]
                save_path = os.path.join(tmpdir, f.filename or f"{key}.jpg")
                f.save(save_path)
                paths[key] = save_path

            # Run pipelines
            left_eye_res  = process_eye(paths['left_eye'], tmpdir)
            right_eye_res = process_eye(paths['right_eye'], tmpdir)
            left_palm_res = process_palm(paths['left_palm'], tmpdir)
            right_palm_res= process_palm(paths['right_palm'], tmpdir)
            nails_res     = process_nail(paths['nail_image'], tmpdir)

            # Eye overall
            eye_labels = []
            eye_conf_list = []
            for res in (left_eye_res, right_eye_res):
                if res.get("success"):
                    eye_labels.append("anemia" if res["prediction"] == "anemia" else "non-anemia")
                    eye_conf_list.append(safe_float(res.get("confidence", 0.0)) / 100.0)
            eye_overall_pred = majority_label(eye_labels) if eye_labels else "not_processed"
            eye_overall_conf = round((np.mean(eye_conf_list) * 100) if eye_conf_list else 0.0, 2)

            # Palm overall
            palm_labels = []
            palm_conf_list = []
            for res in (left_palm_res, right_palm_res):
                if res.get("success"):
                    palm_labels.append("anemia" if res["prediction"] == "anemia" else "non-anemia")
                    palm_conf_list.append(safe_float(res.get("confidence", 0.0)) / 100.0)
            palm_overall_pred = majority_label(palm_labels) if palm_labels else "not_processed"
            palm_overall_conf = round((np.mean(palm_conf_list) * 100) if palm_conf_list else 0.0, 2)

            
            # Final combined
            trio_labels, weighted_scores = [], []

            # Assign weights
            eye_weight = 0.50
            palm_weight = 0.25
            nail_weight = 0.25

            if eye_overall_pred in ("anemia", "non-anemia"):
                trio_labels.append(eye_overall_pred)
                weighted_scores.append(eye_overall_conf/100.0 * eye_weight)

            if palm_overall_pred in ("anemia", "non-anemia"):
                trio_labels.append(palm_overall_pred)
                weighted_scores.append(palm_overall_conf/100.0 * palm_weight)

            if nails_res.get("success"):
                trio_labels.append(nails_res["overall_prediction"])
                weighted_scores.append(safe_float(nails_res.get("overall_confidence", 0.0))/100.0 * nail_weight)

            # Final prediction uses majority vote among labels
            final_pred = majority_label(trio_labels) if trio_labels else "not_processed"
            # Confidence is weighted sum instead of mean
            final_conf = round(sum(weighted_scores) * 100 if weighted_scores else 0.0, 2)
            

            # Build nail individual array for JSON (limit to 10 with nail_id 1..10)
            individual_nails = []
            for item in nails_res.get("per_nail", []):
                individual_nails.append({
                    "nail_id": int(item["nail_id"]),
                    "cropped_image_base64": image_to_base64(item["path"]),
                    "confidence": item["det_conf"],             # detector confidence (%)
                    "anemia_confidence": item["anemia_conf"],   # classifier confidence (%)
                    "anemia_prediction": item["anemia_pred"]    # "anemic"/"non_anemic"
                })

            response = {
                "detailed_results": {
                    "eye_analysis": {
                        "diagnosis_note": "",
                        "eyes_processed": int(left_eye_res.get("success", False)) + int(right_eye_res.get("success", False)),
                        "left_eye": {
                            "cropped_conjunctiva_base64": image_to_base64(left_eye_res.get("cropped")),
                            "confidence": safe_float(left_eye_res.get("confidence", 0.0)),
                            "prediction": left_eye_res.get("prediction", "error"),
                            "success": bool(left_eye_res.get("success", False))
                        },
                        "model_type": "eye_analysis",
                        "overall_prediction": eye_overall_pred,
                        "right_eye": {
                            "cropped_conjunctiva_base64": image_to_base64(right_eye_res.get("cropped")),
                            "confidence": safe_float(right_eye_res.get("confidence", 0.0)),
                            "prediction": right_eye_res.get("prediction", "error"),
                            "success": bool(right_eye_res.get("success", False))
                        },
                        "success": True
                    },
                    "nail_analysis": {
                        "individual_nails": individual_nails,
                        "model_type": "nail_analysis",
                        "overall_prediction": nails_res.get("overall_prediction", "not_processed"),
                        "success": bool(nails_res.get("success", False))
                    },
                    "palm_analysis": {
                        "left_palm": {
                            "cropped_palm_base64": image_to_base64(left_palm_res.get("cropped")),
                            "prediction": left_palm_res.get("prediction", "error"),
                            "confidence": safe_float(left_palm_res.get("confidence", 0.0)),
                            "success": bool(left_palm_res.get("success", False))
                        },
                        "overall_confidence": palm_overall_conf,
                        "overall_prediction": palm_overall_pred,
                        "right_palm": {
                            "cropped_palm_base64": image_to_base64(right_palm_res.get("cropped")),
                            "prediction": right_palm_res.get("prediction", "error"),
                            "confidence": safe_float(right_palm_res.get("confidence", 0.0)),
                            "success": bool(right_palm_res.get("success", False))
                        }
                    }
                },
                "final_prediction_3_models_combined": {
                    "confidence": final_conf,
                    "prediction": final_pred if final_pred != "not_processed" else "not_processed"
                },
                "success": True,
                "timestamp": datetime.utcnow().isoformat()
            }

            return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500

# =========================
# Run
# =========================
if __name__ == "__main__":
    # Tip: set threaded=False if using GPU models to avoid context contention
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)