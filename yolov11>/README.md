# Generic YOLOv11 detection (CVAT serverless)

Template Nuclio function: **Ultralytics YOLO object detection** → CVAT **rectangle** JSON, in the same spirit as [`../generic-yolo-pose`](../generic-yolo-pose) and your existing [`../yolov11`](../yolov11) example.

- **`function.yaml`** — CPU (`python:3.11-slim`, PyTorch CPU wheels, Ultralytics via pip; matches Nuclio `python:3.11` runtime).
- **`function-gpu.yaml`** — GPU (CUDA 12.4 PyTorch wheels); copy to `function.yaml` inside the ZIP for dashboard deploy.

## Layout

| Path | Purpose |
|------|---------|
| `nuclio/main.py` | Handler + optional **post-NMS** (`torchvision.ops.batched_nms`, class-aware) |
| `nuclio/function.yaml` | CPU manifest |
| `nuclio/function-gpu.yaml` | GPU manifest |
| `nuclio/weights/best.pt` | **You provide** (or change `spec.build.copy` + `MODEL_PATH`) |

## Before deploy

1. Put weights at `nuclio/weights/best.pt` (or adjust copy + `MODEL_PATH`).
2. Replace `metadata.annotations.spec` with your CVAT label list (`id` / `name` per class); names must match the model’s class names (`result.names`).
3. Adjust `metadata.name` / `namespace` if your Nuclio project differs from `cvat`.

## Deploy (Nuclio dashboard)

Same pattern as [generic-yolo-pose README](../generic-yolo-pose/README.md): zip must contain **`function.yaml`** at the root, plus `main.py` and `weights/best.pt`.

**CPU**

```bash
cd serverless/custom/generic-yolov11/nuclio
zip -r "$HOME/generic-yolov11-cpu.zip" function.yaml main.py weights/best.pt
```

**GPU**

```bash
cd serverless/custom/generic-yolov11/nuclio
tmp=$(mktemp -d)
cp function-gpu.yaml "$tmp/function.yaml"
cp main.py "$tmp/"
mkdir -p "$tmp/weights"
cp weights/best.pt "$tmp/weights/"
( cd "$tmp" && zip -r "$HOME/generic-yolov11-gpu.zip" . )
rm -rf "$tmp"
```

Open [http://localhost:8070](http://localhost:8070), project **`cvat`**, create/import from archive, build until **ready**. See [CVAT auto-annotation install](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/) for prerequisites.

## Environment variables

| Variable | Meaning |
|----------|---------|
| `MODEL_PATH` | Checkpoint inside the image (default `/opt/nuclio/weights/best.pt`) |
| `DEVICE` | `cpu`, `auto`, or `cuda` / `cuda:0`, … |
| `YOLO_TASK` | Optional; e.g. `detect`. If **unset**, `YOLO(path)` only (Ultralytics infers). |
| `BOX_CONF_THRESHOLD` | Default `conf` if the request omits `threshold` |
| `POST_NMS` | `1` — apply extra **batched_nms** after YOLO (like `yolov11/main.py`); `0` — use only Ultralytics output |
| `NMS_IOU_THRESHOLD` | IoU threshold for `POST_NMS` (default `0.4`) |
| `ULTRA_VERBOSE` | `1` — verbose Ultralytics predict logs |

## Request / response

- **Request**: `{"image": "<base64>", "threshold": 0.3}` (`threshold` optional).
- **Response**: JSON list of `{ "type": "rectangle", "label", "confidence" (string), "points": [x1,y1,x2,y2] }`.

## Differences vs `custom/yolov11`

| | `yolov11` | `generic-yolov11` |
|---|-----------|-------------------|
| Device | `DEVICE` env (`auto` default on GPU manifest) | `DEVICE` (`cpu` on CPU YAML, `auto` on GPU YAML) |
| Weights path | `weights/yolov11s.pt` | `MODEL_PATH` + build `copy` |
| NMS | Class-aware `torchvision.ops.batched_nms` | Same; toggle with `POST_NMS` |
| Config | Fixed in code | Env + YAML |

Optional CLI bulk deploy: [`serverless/deploy_cpu.sh`](../../deploy_cpu.sh) / [`deploy_gpu.sh`](../../deploy_gpu.sh) glob `function.yaml` / `function-gpu.yaml`.
