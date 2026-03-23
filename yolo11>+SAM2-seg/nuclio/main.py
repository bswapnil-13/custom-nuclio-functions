import json
import base64
from PIL import Image
import io
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import torchvision.ops as ops
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def to_cvat_mask(mask):
    """Convert a binary mask to CVAT's mask format.

    Returns the mask in CVAT RLE format with bounding box computed from actual mask content.
    """
    # Find bounding box from actual mask content
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    ytl, ybr = np.where(rows)[0][[0, -1]]
    xtl, xbr = np.where(cols)[0][[0, -1]]

    # Extract tight mask region
    tight_mask = mask[ytl:ybr + 1, xtl:xbr + 1]

    # Flatten and append bounding box
    flattened = tight_mask.flat[:].tolist()
    flattened.extend([int(xtl), int(ytl), int(xbr), int(ybr)])
    return flattened


def convert_mask_to_polygon(mask):
    """Convert binary mask to polygon contour."""
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]

    if not contours:
        return None

    # Get the largest contour
    contour = max(contours, key=lambda arr: arr.size)
    if contour.shape.count(1):
        contour = np.squeeze(contour)
    if contour.size < 3 * 2:
        return None

    polygon = []
    for point in contour:
        polygon.append(int(point[0]))
        polygon.append(int(point[1]))

    return polygon


def init_context(context):
    context.logger.info("Init context...  0%")

    # Initialize device
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        context.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        context.logger.info("Using CPU")

    context.user_data.device = device

    # Load YOLO model
    context.logger.info("Loading YOLO11 model...")
    yolo_model = YOLO('/opt/nuclio/weights/yolov11s.pt')
    yolo_model.to(device)
    context.user_data.yolo_model = yolo_model
    context.logger.info("YOLO11 model loaded")

    # Load SAM2 model (using tiny model for 6GB VRAM GPUs)
    context.logger.info("Loading SAM2 model...")
    sam2_checkpoint = "/opt/nuclio/sam2/sam2.1_hiera_tiny.pt"
    sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    sam2_model = build_sam2(sam2_model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    context.user_data.sam2_predictor = sam2_predictor
    context.logger.info("SAM2 model loaded")

    context.logger.info("Init context...100%")


def handler(context, event):
    context.logger.info("Run YOLO11 + SAM2 segmentation model")

    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    threshold = float(data.get("threshold", 0.3))
    image = Image.open(buf).convert("RGB")
    image_np = np.array(image)

    device = context.user_data.device
    yolo_model = context.user_data.yolo_model
    sam2_predictor = context.user_data.sam2_predictor

    # Run YOLO detection
    yolo_results = yolo_model(image, conf=threshold)

    encoded_results = []

    for result in yolo_results:
        boxes = []
        scores = []
        labels = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls)

        if not boxes:
            continue

        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)

        # Apply NMS
        keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_threshold=0.4)

        # Filter boxes after NMS
        filtered_boxes = [boxes[i.item()] for i in keep_indices]
        filtered_scores = [scores[i.item()] for i in keep_indices]
        filtered_labels = [labels[i.item()] for i in keep_indices]

        if not filtered_boxes:
            continue

        # Set image for SAM2 (only once per image)
        with torch.inference_mode():
            sam2_predictor.set_image(image_np)

            # Convert boxes to numpy array for SAM2 batch prediction
            input_boxes = np.array(filtered_boxes)

            # Predict masks for all boxes at once
            masks, mask_scores, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

        # Process each detection
        for idx, (box, score, label) in enumerate(zip(filtered_boxes, filtered_scores, filtered_labels)):
            # Get the mask for this detection
            if len(masks.shape) == 4:
                # Shape: (num_boxes, num_masks, H, W)
                mask = masks[idx, 0]
            else:
                # Shape: (num_masks, H, W) for single box
                mask = masks[0]

            # Convert mask to binary uint8
            mask_binary = (mask > 0.5).astype(np.uint8) * 255

            # Convert mask to polygon
            polygon = convert_mask_to_polygon(mask_binary)

            if polygon is None or len(polygon) < 6:
                # Fall back to rectangle if polygon conversion fails
                encoded_results.append({
                    'confidence': str(score),
                    'label': result.names[label],
                    'points': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    'type': 'rectangle'
                })
                continue

            # Create CVAT mask format (compute bbox from actual mask content)
            # Convert 255 mask to binary 0/1 for CVAT
            mask_for_cvat = (mask_binary > 0).astype(np.uint8)
            cvat_mask = to_cvat_mask(mask_for_cvat)

            if cvat_mask is None:
                # Fall back to rectangle if mask is empty
                encoded_results.append({
                    'confidence': str(score),
                    'label': result.names[label],
                    'points': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    'type': 'rectangle'
                })
                continue

            encoded_results.append({
                'confidence': str(score),
                'label': result.names[label],
                'points': polygon,
                'mask': cvat_mask,
                'type': 'mask'
            })

    context.logger.info(f"Detected {len(encoded_results)} objects with masks")

    return context.Response(
        body=json.dumps(encoded_results),
        headers={},
        content_type='application/json',
        status_code=200
    )
