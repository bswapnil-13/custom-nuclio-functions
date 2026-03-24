"""
Generic Ultralytics YOLO detection → CVAT rectangle JSON (YOLOv11-style pipeline).

NMS: Ultralytics built-in (iou + optional agnostic_nms). Optional extra torchvision batched_nms
if POST_NMS=1. Configure via function.yaml / function-gpu.yaml env vars.
"""
import base64
import io
import json
import os
from typing import Any, List

import torch
import torchvision.ops as ops
from PIL import Image
from ultralytics import YOLO

_DEFAULT_MODEL_PATH = "/opt/nuclio/weights/best.pt"
_DEFAULT_DEVICE = "auto"
_DEFAULT_BOX_CONF = "0.3"
_DEFAULT_POST_NMS = "1"
_DEFAULT_NMS_IOU = "0.4"
_DEFAULT_YOLO_IOU = "0.7"
_DEFAULT_AGNOSTIC_NMS = "0"
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


def _nms_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def init_context(context):
    model_path = _env("MODEL_PATH", _DEFAULT_MODEL_PATH)
    task_raw = os.environ.get("YOLO_TASK")
    if task_raw is not None and str(task_raw).strip() != "":
        task = str(task_raw).strip()
        context.logger.info("Loading YOLO model from %s (task=%s)...", model_path, task)
        model = YOLO(model_path, task=task)
    else:
        context.logger.info("Loading YOLO model from %s (task inferred by Ultralytics)...", model_path)
        model = YOLO(model_path)

    device = _resolve_device(context)
    model.to(device)

    context.user_data.model = model
    context.user_data.device = device
    context.user_data.default_box_conf = float(_env("BOX_CONF_THRESHOLD", _DEFAULT_BOX_CONF))
    context.user_data.post_nms = _env("POST_NMS", _DEFAULT_POST_NMS).lower() in ("1", "true", "yes")
    context.user_data.nms_iou = float(_env("NMS_IOU_THRESHOLD", _DEFAULT_NMS_IOU))
    context.user_data.yolo_iou = float(_env("YOLO_IOU", _DEFAULT_YOLO_IOU))
    context.user_data.agnostic_nms = _env("YOLO_AGNOSTIC_NMS", _DEFAULT_AGNOSTIC_NMS).lower() in (
        "1",
        "true",
        "yes",
    )

    context.logger.info(
        "Model ready on %s; yolo_iou=%s agnostic_nms=%s; post_nms=%s nms_iou=%s",
        device,
        context.user_data.yolo_iou,
        context.user_data.agnostic_nms,
        context.user_data.post_nms,
        context.user_data.nms_iou,
    )


def _decode_image_b64(image_b64: str) -> Image.Image:
    missing = len(image_b64) % 4
    if missing:
        image_b64 += "=" * (4 - missing)
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def handler(context, event):
    ud = context.user_data
    data = event.body
    if isinstance(data, (bytes, bytearray)):
        data = json.loads(data.decode("utf-8"))
    elif isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        return context.Response(
            status_code=400,
            content_type="application/json",
            body='{"error":"expected JSON body"}',
        )

    image_b64 = data.get("image")
    if not image_b64:
        return context.Response(
            status_code=400,
            content_type="application/json",
            body='{"error":"missing image"}',
        )

    threshold = float(data.get("threshold", ud.default_box_conf))
    try:
        yolo_iou = float(data.get("iou", ud.yolo_iou))
    except (TypeError, ValueError):
        yolo_iou = ud.yolo_iou

    verbose = _env("ULTRA_VERBOSE", _DEFAULT_VERBOSE).lower() in ("1", "true", "yes")

    try:
        image = _decode_image_b64(image_b64)
    except Exception as e:
        context.logger.error("Image decode failed: %s", e)
        return context.Response(
            status_code=400,
            content_type="application/json",
            body=json.dumps({"error": "image decode failed", "detail": str(e)}),
        )

    results = ud.model(
        image,
        conf=threshold,
        iou=yolo_iou,
        agnostic_nms=ud.agnostic_nms,
        device=ud.device,
        verbose=verbose,
    )

    encoded_results: List[dict[str, Any]] = []
    nms_dev = _nms_device(ud.device)

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes: List[List[float]] = []
        scores: List[float] = []
        labels: List[int] = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls)

        if not boxes:
            continue

        names = result.names
        if ud.post_nms:
            boxes_t = torch.tensor(boxes, dtype=torch.float32, device=nms_dev)
            scores_t = torch.tensor(scores, dtype=torch.float32, device=nms_dev)
            keep = ops.nms(boxes_t, scores_t, ud.nms_iou)
            indices = [i.item() for i in keep]
        else:
            indices = list(range(len(boxes)))

        for i in indices:
            encoded_results.append(
                {
                    "confidence": str(scores[i]),
                    "label": names[labels[i]],
                    "points": [
                        float(boxes[i][0]),
                        float(boxes[i][1]),
                        float(boxes[i][2]),
                        float(boxes[i][3]),
                    ],
                    "type": "rectangle",
                }
            )

    return context.Response(
        body=json.dumps(encoded_results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
