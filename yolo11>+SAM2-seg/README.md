# YOLO11 + SAM2 segmentation (CVAT serverless)

Nuclio function that runs **Ultralytics YOLO11** for detection, then **SAM 2** to produce **instance masks** per box. Output is CVAT-style JSON: each object is either a **`mask`** (polygon + RLE `mask` field) or falls back to a **`rectangle`** if mask extraction fails.

The handler uses **`torchvision.ops.nms`** on YOLO boxes before SAM2; SAM2 runs in batch over the kept boxes.

## Layout

| Path | Purpose |
|------|---------|
| `nuclio/main.py` | `init_context` (YOLO + SAM2 load), `handler` (detect → segment → encode) |
| `nuclio/function.yaml` | **CPU** build (Ubuntu + PyTorch CPU, longer `eventTimeout`) |
| `nuclio/function-gpu.yaml` | **GPU** build (`nvidia.com/gpu: 1`, CUDA 12.4 wheels, SAM2 **tiny** weights) |

## Weights and paths (important)

`main.py` expects:

- **YOLO**: `/opt/nuclio/weights/yolov11s.pt`  
  The packaged YAML does **not** copy this file by default. Before you build, either add a `spec.build.copy` entry in the YAML you deploy (mirroring other functions in this repo) or bake the file into the image another way.

- **SAM2**: `/opt/nuclio/sam2/sam2.1_hiera_tiny.pt` with config `configs/sam2.1/sam2.1_hiera_t.yaml`  
  **function-gpu.yaml** downloads that tiny checkpoint during the image build. **function.yaml** (CPU) currently downloads a **different** checkpoint in its directives; align the downloaded file name and `main.py` (or switch the YAML download URL) so the checkpoint on disk matches what the code loads.

## Before you deploy

1. Place **YOLO** weights at the path your build exposes (e.g. `nuclio/weights/yolov11s.pt` + `copy` in YAML), or edit `main.py` to match your layout.
2. Ensure **SAM2** checkpoint and config in `main.py` match what the **build directives** install.
3. Edit `metadata.annotations.spec` so label **`name`** values match your model’s class names (`result.names` from YOLO).
4. Adjust `metadata.name` and `metadata.namespace` if your Nuclio project is not `cvat`.

## Deploy (Nuclio dashboard)

The archive must have **`function.yaml` at the root**, plus `main.py` and any files referenced by `spec.build.copy` (and satisfy paths the build expects).

**CPU** — from `yolo11+SAM2-seg/nuclio`:

```bash
cd "yolo11+SAM2-seg/nuclio"
zip -r "$HOME/yolo11-sam2-seg-cpu.zip" function.yaml main.py
# Add weights/ and other copied files once your YAML lists them under spec.build.copy
```

**GPU** — Nuclio expects the config file to be named `function.yaml` inside the zip:

```bash
cd "yolo11+SAM2-seg/nuclio"
tmp=$(mktemp -d)
cp function-gpu.yaml "$tmp/function.yaml"
cp main.py "$tmp/"
# mkdir -p "$tmp/weights" && cp weights/yolov11s.pt "$tmp/weights/"  # if you add copy in YAML
( cd "$tmp" && zip -r "$HOME/yolo11-sam2-seg-gpu.zip" . )
rm -rf "$tmp"
```

Open your Nuclio UI (often `http://localhost:8070` with CVAT), select project **`cvat`**, import the archive, build until **ready**. See [CVAT automatic annotation](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/) for cluster prerequisites.

## Environment

- **`PYTHONPATH`** is set to `/opt/nuclio/sam2` in the YAML so SAM2 imports resolve inside the container.

Thresholds and device selection are **not** env-driven in this handler: device is chosen in `init_context` (CUDA if available, else CPU); box confidence uses the request’s `threshold` (default `0.3`).

## HTTP request and response

**Request** (JSON):

- `image` (required): base64-encoded image (same convention as other CVAT serverless detectors).
- `threshold` (optional): YOLO box confidence (default `0.3`).

**Response**: JSON array of objects:

- **`type`: `"mask"`** — `label`, `confidence` (string), `points` (polygon), `mask` (CVAT RLE-style list with trailing bbox ints).
- **`type`: `"rectangle"`** — fallback when polygon/mask conversion fails; `points` are `[x1, y1, x2, y2]`.

## Troubleshooting

- **Missing YOLO weights at runtime** — Add `spec.build.copy` for `yolov11s.pt` (or change `main.py` / `MODEL_PATH` pattern to match your packaging).
- **SAM2 checkpoint not found** — Match filename in `main.py` to the file installed in the Dockerfile directives for the YAML you deployed (CPU vs GPU differ today).
- **OOM on GPU** — GPU YAML uses the **tiny** SAM2 variant; use CPU YAML only if you accept much slower runs, or reduce image size / worker count in Nuclio.
