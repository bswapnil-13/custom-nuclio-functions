"""
Generic Ultralytics YOLO pose → CVAT skeleton JSON.

Customize via environment variables in function.yaml / function-gpu.yaml
(see comments there). Env vars override the defaults below when set.
"""
import base64
import io
import json
import os
from typing import Any, List, Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Defaults when env vars are not set (edit here or prefer YAML env).
_DEFAULT_MODEL_PATH = "/opt/nuclio/weights/best.pt"
_DEFAULT_SKELETON_LABEL = "skeleton_pose"
_DEFAULT_YOLO_TASK = "pose"
_DEFAULT_DEVICE = "auto"  # auto | cpu | cuda | cuda:0 ...
_DEFAULT_BOX_CONF = "0.5"
_DEFAULT_KEYPOINT_CONF = "0.5"
_DEFAULT_INTEGER_COORDS = "0"
_DEFAULT_KEYPOINT_OUTSIDE = "0"
_DEFAULT_VERBOSE = "0"


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _resolve_device(context) -> str:
    mode = _env("DEVICE", _DEFAULT_DEVICE).lower().strip()
    if mode == "cpu":
        return "cpu"
    if mode == "auto":
        if torch.cuda.is_available():
            dev = "cuda:0"
            context.logger.info("DEVICE=auto: using %s (%s)", dev, torch.cuda.get_device_name(0))
            return dev
        context.logger.info("DEVICE=auto: CUDA not available, using cpu")
        return "cpu"
    return mode


def _parse_keypoint_names() -> Optional[List[str]]:
    raw = os.environ.get("KEYPOINT_NAMES", "").strip()
    if not raw:
        return None
    try:
        names = json.loads(raw)
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError("KEYPOINT_NAMES must be a JSON array of strings")
        return names
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Invalid KEYPOINT_NAMES: {e}") from e


def _keypoint_element_label(index: int, names: Optional[List[str]]) -> str:
    if names is not None and index < len(names):
        return names[index]
    return str(index + 1)


def init_context(context):
    model_path = _env("MODEL_PATH", _DEFAULT_MODEL_PATH)
    task = _env("YOLO_TASK", _DEFAULT_YOLO_TASK).strip() or None

    context.logger.info("Loading YOLO model from %s (task=%s)...", model_path, task or "infer")
    load_kw: dict = {}
    if task:
        load_kw["task"] = task
    model = YOLO(model_path, **load_kw)

    device = _resolve_device(context)
    model.to(device)

    context.user_data.model = model
    context.user_data.device = device
    context.user_data.keypoint_names = _parse_keypoint_names()
    context.user_data.skeleton_label = _env("SKELETON_LABEL", _DEFAULT_SKELETON_LABEL)
    context.user_data.default_box_conf = float(_env("BOX_CONF_THRESHOLD", _DEFAULT_BOX_CONF))
    context.user_data.kp_conf_threshold = float(_env("KEYPOINT_CONF_THRESHOLD", _DEFAULT_KEYPOINT_CONF))
    context.user_data.integer_coords = _env("INTEGER_COORDS", _DEFAULT_INTEGER_COORDS).lower() in (
        "1",
        "true",
        "yes",
    )
    context.user_data.keypoint_outside = _env("KEYPOINT_OUTSIDE", _DEFAULT_KEYPOINT_OUTSIDE).lower() in (
        "1",
        "true",
        "yes",
    )

    context.logger.info(
        "Model ready on %s; skeleton_label=%s; keypoint_outside=%s; integer_coords=%s",
        device,
        context.user_data.skeleton_label,
        context.user_data.keypoint_outside,
        context.user_data.integer_coords,
    )


def _decode_image(image_b64: str) -> np.ndarray:
    missing = len(image_b64) % 4
    if missing:
        image_b64 += "=" * (4 - missing)
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(image)


def _to_xy_numpy(tensor_or_array) -> np.ndarray:
    if hasattr(tensor_or_array, "detach"):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


def handler(context, event):
    ud = context.user_data
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        return context.Response(status_code=400, content_type="application/json", body='{"error":"expected JSON body"}')

    image_b64 = data.get("image")
    if not image_b64:
        return context.Response(status_code=400, content_type="application/json", body='{"error":"missing image"}')

    threshold = float(data.get("threshold", ud.default_box_conf))
    verbose = _env("ULTRA_VERBOSE", _DEFAULT_VERBOSE).lower() in ("1", "true", "yes")

    try:
        img = _decode_image(image_b64)
    except Exception as e:
        context.logger.error("Image decode failed: %s", e)
        return context.Response(
            status_code=400,
            content_type="application/json",
            body=json.dumps({"error": "image decode failed", "detail": str(e)}),
        )

    results = ud.model(img, conf=threshold, device=ud.device, verbose=verbose)[0]

    detections: List[dict[str, Any]] = []
    if results.keypoints is None or results.boxes is None or len(results.boxes) == 0:
        return context.Response(body=json.dumps(detections), content_type="application/json", status_code=200)

    names: Optional[List[str]] = ud.keypoint_names
    kp_conf_threshold: float = ud.kp_conf_threshold

    for det_idx in range(len(results.boxes)):
        box_conf = float(results.boxes.conf[det_idx])
        if box_conf < threshold:
            continue

        xy = _to_xy_numpy(results.keypoints.xy[det_idx])
        kp_conf_arr = None
        if results.keypoints.conf is not None:
            kp_conf_arr = _to_xy_numpy(results.keypoints.conf[det_idx]).reshape(-1)

        elements = []
        for idx, (x, y) in enumerate(xy):
            label = _keypoint_element_label(idx, names)
            el: dict[str, Any] = {
                "type": "points",
                "label": label,
                "points": (
                    [int(round(float(x))), int(round(float(y)))]
                    if ud.integer_coords
                    else [float(x), float(y)]
                ),
            }
            if ud.keypoint_outside:
                kpc = float(kp_conf_arr[idx]) if kp_conf_arr is not None and idx < len(kp_conf_arr) else 1.0
                el["attributes"] = []
                el["outside"] = kpc < kp_conf_threshold
            elements.append(el)

        detections.append(
            {
                "confidence": str(box_conf),
                "label": ud.skeleton_label,
                "type": "skeleton",
                "elements": elements,
            }
        )

    return context.Response(body=json.dumps(detections), content_type="application/json", status_code=200)
