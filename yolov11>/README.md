# Generic YOLOv11 detection (CVAT serverless)

Template Nuclio function: **Ultralytics YOLO object detection** → CVAT **rectangle** JSON, in the same spirit as [`../yolo-pose`](../yolo-pose). Paths below are relative to this repo’s `yolov11>/` directory (quote the folder name in shell: `"yolov11>"`).

- **`function.yaml`** — CPU (`ultralytics:latest-cpu`).
- **`function-gpu.yaml`** — GPU; copy to `function.yaml` inside the ZIP for dashboard deploy.

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

Same pattern as [yolo-pose README](../yolo-pose/README.md): zip must contain **`function.yaml`** at the root, plus `main.py` and `weights/best.pt`.

**CPU**

```bash
cd "yolov11>/nuclio"
zip -r "$HOME/generic-yolov11-cpu.zip" function.yaml main.py weights/best.pt
```

**GPU**

```bash
cd "yolov11>/nuclio"
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

## Differences vs a minimal YOLO example

| | Minimal example | This function |
|---|-----------------|---------------|
| Device | Often hard-coded CPU | `DEVICE` (`auto` on GPU YAML) |
| Weights | Fixed filename in repo | `MODEL_PATH` + build `copy` |
| NMS | Varies | Optional **class-aware** `torchvision.ops.batched_nms` via `POST_NMS` |
| Config | Fixed in code | Env + YAML |

Optional CLI bulk deploy: use CVAT’s `serverless/deploy_cpu.sh` / `deploy_gpu.sh` with this `nuclio/` tree, or dashboard ZIP as above.
