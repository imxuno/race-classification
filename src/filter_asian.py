import os
import sys
import shutil
import cv2
import json
import shutil
import argparse
import warnings
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

from insightface.app import FaceAnalysis

# -----------------------------
# Config & Utils
# -----------------------------
RACE_CLASSES = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eeastern']
DEFAULT_REF5PTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32) # 112x112 템플릿 기준

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def largest_face(faces) -> Optional:
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

def align_face_by_5pts(img_bgr: np.ndarray, kps: np.ndarray, out_size: int = 112) -> Optional[np.ndarray]:
    """
    img_bgr: HxWx3, BGR
    kps: (5,2) landmarks order: [left_eye, right_eye, nose, left_mouth, right_mouth]
    """
    try:
        src = kps.astype(np.float32)
        dst = DEFAULT_REF5PTS.copy()
        # scale dst to out_size (default template is for 112x112)
        scale = out_size / 112.0
        dst *= scale
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(img_bgr, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return aligned
    except Exception:
        return None



# -----------------------------
# Model init
# -----------------------------
def load_fairface_resnet34(model_path: str, device: str = "cuda:0") -> nn.Module:
    model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 18) # FairFace: 18 outputs (race 7 + gender 2 + age 9)
    state = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        warnings.warn(f"State dict mismatch. missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model



# -----------------------------
# Inference helpers
# -----------------------------
def to_pil_rgb(img_bgr: np.ndarray):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def get_transform(size: int = 224):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
@torch.no_grad()
def classify_batch(model: nn.Module, batch_imgs: List[np.ndarray], device: str, tfm) -> List[Dict[str, float]]:
    if len(batch_imgs) == 0:
        return []
    tensors = [tfm(img) for img in batch_imgs]  # img: RGB np array expected by ToPILImage
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    out = model(batch)
    probs = torch.softmax(out[:, :7], dim=1).cpu().numpy()  # first 7 = race
    return [dict(zip(RACE_CLASSES, p)) for p in probs]

# -----------------------------
# Main processing (streaming)
# -----------------------------
def iter_image_files(root: Path, exts=(".png", ".jpg", ".jpeg")):
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                yield Path(dp) / f
        
def process_dataset(
    input_root: Path,
    output_root: Path,
    model_path: Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    threshold: float = 0.7,
    align: bool = True,
    copy_selected: bool = True,
    borderline_save: bool = True, # 원래 False 였음
    borderline_margin: float = 0.05,
    detector_ctx_id: int = 0,           # use 0 for single-threaded GPU detect; use -1 for CPU detect
    detector_size: Tuple[int,int] = (1024, 1024),
    detector_thresh: float = 0.3,
    out_face_size: int = 224,
    max_files: Optional[int] = None
):
    set_seed(42)

    # init detector (single instance, single thread)
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=detector_ctx_id, det_size=detector_size, det_thresh=detector_thresh)
    
    try:
        app.models['detection'].threshold = float(detector_thresh)
    except Exception:
        pass

    # init classifier
    model = load_fairface_resnet34(model_path, device=device)
    tfm = get_transform(out_face_size)

    # logging
    log_rows = []
    debug_dir = output_root / "_debug"
    borderline_dir = debug_dir / "borderline"
    if borderline_save:
        borderline_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_image_files(input_root))
    if max_files is not None:
        files = files[:max_files]
    total = len(files)

    buffer_imgs_rgb: List[np.ndarray] = []
    buffer_meta: List[Tuple[Path, Path, float, Tuple[int,int,int,int], bool]] = []  # (src, dst, det_score, bbox, aligned?)

    pbar = tqdm(files, desc="Detecting & buffering", unit="img")
    for idx, src in enumerate(pbar):
        # Build destination path (mirror structure)
        rel = src.relative_to(input_root)
        dst = output_root / rel

        # Read
        img_bgr = cv2.imread(str(src))
        if img_bgr is None or img_bgr.size == 0:
            log_rows.append({
                "file": str(src), "east_prob": 0.0, "se_prob": 0.0, "total_east": 0.0,
                "is_east_asian": False, "threshold": threshold,
                "det_score": 0.0, "bbox": "", "face_area": 0, "aligned": False,
                "error": "imread_failed", "detected": False,
            })
            continue

        # Detect (single thread)
        faces = app.get(img_bgr)
        if not faces:
            log_rows.append({
                "file": str(src), "east_prob": 0.0, "se_prob": 0.0, "total_east": 0.0,
                "is_east_asian": False, "threshold": threshold,
                "det_score": 0.0, "bbox": "", "face_area": 0, "aligned": False,
                "error": "no_face", "detected": False,
            })
            continue

        face = largest_face(faces)
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = img_bgr.shape[:2]
        x1 = max(0, min(x1, w-1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h-1)); y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            log_rows.append({
                "file": str(src), "east_prob": 0.0, "se_prob": 0.0, "total_east": 0.0,
                "is_east_asian": False, "threshold": threshold,
                "det_score": float(getattr(face, "det_score", 0.0)),
                "bbox": f"{x1},{y1},{x2},{y2}",
                "face_area": 0, "aligned": False, "error": "invalid_bbox",
                "detected": True,
            })
            continue

        # Align or crop
        aligned = False
        face_bgr = None
        if align and hasattr(face, "kps") and face.kps is not None:
            face_bgr = align_face_by_5pts(img_bgr, face.kps, out_size=out_face_size)
            aligned = face_bgr is not None
        if face_bgr is None:
            face_bgr = img_bgr[y1:y2, x1:x2]
        if face_bgr is None or face_bgr.size == 0:
            log_rows.append({
                "file": str(src), "east_prob": 0.0, "se_prob": 0.0, "total_east": 0.0,
                "is_east_asian": False, "threshold": threshold,
                "det_score": float(getattr(face, "det_score", 0.0)),
                "bbox": f"{x1},{y1},{x2},{y2}",
                "face_area": 0, "aligned": aligned, "error": "empty_crop",
                "detected": True,
            })
            continue

        # buffer (RGB for torchvision)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        buffer_imgs_rgb.append(face_rgb)
        face_area = (x2 - x1) * (y2 - y1)
        buffer_meta.append((
            src, dst, float(getattr(face, "det_score", 0.0)),
            (int(x1), int(y1), int(x2), int(y2)), aligned
        ))

        # classify when batch full
        if len(buffer_imgs_rgb) >= batch_size:
            probs_list = classify_batch(model, buffer_imgs_rgb, device, tfm)
            
            # enumerate로 현재 배치 인덱스 추적
            for i, (probs, meta) in enumerate(zip(probs_list, buffer_meta)):
                src_, dst_, det_score, bbox, aligned_flag = meta
                east_prob = float(probs.get('East Asian', 0.0))
                se_prob   = float(probs.get('Southeast Asian', 0.0))
                total_prob = east_prob + se_prob
                is_east = total_prob > threshold
                
                log_rows.append({
                    "file": str(src_),
                    "east_prob": east_prob, "se_prob": se_prob, "total_east": total_prob,
                    "is_east_asian": bool(is_east), "threshold": threshold,
                    "det_score": det_score, "detected": True,                  # ★ 추가
                    "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    "face_area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                    "aligned": bool(aligned_flag), "error": ""
                })

                if borderline_save and (abs(total_prob - threshold) <= borderline_margin):
                    out_path = borderline_dir / src_.relative_to(input_root)  # ★ 파일명 충돌 방지(아래 3번 참조)
                    ensure_dir(out_path)
                    cv2.imwrite(str(out_path), cv2.cvtColor(buffer_imgs_rgb[i], cv2.COLOR_RGB2BGR))

                if is_east and copy_selected:
                    ensure_dir(dst_)
                    shutil.copy2(src_, dst_)

            buffer_imgs_rgb.clear()
            buffer_meta.clear()

    # flush remainder
    if buffer_imgs_rgb:
        probs_list = classify_batch(model, buffer_imgs_rgb, device, tfm)
        
        for i, (probs, meta) in enumerate(zip(probs_list, buffer_meta)):
            src_, dst_, det_score, bbox, aligned_flag = meta
            east_prob = float(probs.get('East Asian', 0.0))
            se_prob   = float(probs.get('Southeast Asian', 0.0))
            total_prob = east_prob + se_prob
            is_east = total_prob > threshold
                
            log_rows.append({
                "file": str(src_),
                "east_prob": east_prob, "se_prob": se_prob, "total_east": total_prob,
                "is_east_asian": bool(is_east), "threshold": threshold,
                "det_score": det_score, "detected": True,                  # ★ 추가
                "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "face_area": (bbox[2]-bbox[0])*(bbox[3]-bbox[1]),
                "aligned": bool(aligned_flag), "error": ""
            })

            if borderline_save and (abs(total_prob - threshold) <= borderline_margin):
                    rel = src_.relative_to(input_root)                # ★ 상대경로
                    out_path = borderline_dir / rel                   #   → 원본 폴더 구조 보존
                    ensure_dir(out_path)                              #   (ensure_dir는 parent mkdir)
                    cv2.imwrite(str(out_path),
                                cv2.cvtColor(buffer_imgs_rgb[i], cv2.COLOR_RGB2BGR))  # ★ i번째 샘플

            if is_east and copy_selected:
                ensure_dir(dst_)
                shutil.copy2(src_, dst_)

    # save log & summary
    ensure_dir(output_root / "filter_log.csv")
    df = pd.DataFrame(log_rows)
    df.to_csv(output_root / "filter_log.csv", index=False, encoding="utf-8-sig")

    detected_faces = int(df['detected'].fillna(False).sum())
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_files": total,
        "detected_faces": detected_faces,
        "selected_asian": int(df["is_east_asian"].sum()),
        "ratio_detected": float(detected_faces / total) if total else 0.0,
        "ratio_selected": float(df["is_east_asian"].sum() / total) if total else 0.0,
        "threshold": threshold,
        "align": align,
        "batch_size": batch_size,
        "detector_ctx_id": detector_ctx_id,
        "detector_size": detector_size,
        "detector_thresh": detector_thresh,
        "model_path": model_path,
    }
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")



# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Asian subsetting with FairFace + InsightFace")
    parser.add_argument("--input_root", type=str, default="D:\\AI-hub")
    parser.add_argument("--output_root", type=str, default="D:\\cGAN_datasets\\filtered_asian\\filtered_aihub(thresh=0.7)_v1")
    parser.add_argument("--model_path", type=str, default="D:\\projects\\race_classification\\src\\model\\res34_fair_align_multi_7_20190809.pt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--no_align", action="store_true", help="disable 5-point alignment")
    parser.add_argument("--no_copy", action="store_true", help="do not copy selected files (log only)")
    parser.add_argument("--borderline_save", action="store_true", help="save borderline samples near threshold")
    parser.add_argument("--borderline_margin", type=float, default=0.05)
    parser.add_argument("--detector_ctx_id", type=int, default=0, help="0: GPU detect; -1: CPU detect")
    parser.add_argument("--detector_w", type=int, default=640)
    parser.add_argument("--detector_h", type=int, default=640)
    parser.add_argument("--detector_thresh", type=float, default=0.3)
    parser.add_argument("--out_face_size", type=int, default=224)
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    process_dataset(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        threshold=args.threshold,
        align=(not args.no_align),
        copy_selected=(not args.no_copy),
        borderline_save=args.borderline_save,
        borderline_margin=args.borderline_margin,
        detector_ctx_id=args.detector_ctx_id,
        detector_size=(args.detector_w, args.detector_h),
        detector_thresh=(args.detector_thresh),
        out_face_size=args.out_face_size,
        max_files=args.max_files
    )

if __name__ == "__main__":
    main()