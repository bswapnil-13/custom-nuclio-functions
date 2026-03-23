# Generic YOLO pose (CVAT serverless)

This folder is a **template Nuclio function** that runs an **Ultralytics YOLO pose** model and returns detections in **CVAT skeleton** JSON. The same `main.py` is used for **CPU** and **GPU** builds. There are **two** manifests: **`function.yaml`** (CPU, default for dashboard zips) and **`function-gpu.yaml`** (GPU — copy/rename to `function.yaml` when you build the archive). Deployment below is described for the **Nuclio dashboard** (no `nuctl` required on your machine for the main flow). Paths are relative to this repository’s `yolo-pose/` directory.

## Layout

| Path | Purpose |
|------|---------|
| `nuclio/main.py` | Handler and `init_context` (model load, device selection) |
| `nuclio/function.yaml` | **CPU** manifest; use as-is in dashboard ZIPs |
| `nuclio/function-gpu.yaml` | **GPU** manifest (`nvidia.com/gpu`); copy to `function.yaml` inside the ZIP for GPU dashboard deploy |
| `nuclio/weights/best.pt` | **You provide** this file before build (see below) |

## Before you deploy

1. **Weights**  
   Copy your trained pose weights to:

   `nuclio/weights/best.pt`

   The build copies them into the image at `/opt/nuclio/weights/best.pt`. If you use another filename, set `MODEL_PATH` in the YAML to match the `dest` path you use in `spec.build.copy`.

2. **CVAT label spec**  
   Edit `metadata.annotations.spec` in the YAML you deploy. The skeleton **`name`** must match **`SKELETON_LABEL`**, and every **sublabel `name`** must match what the function emits for each keypoint (either `"1"`, `"2"`, … or the names in `KEYPOINT_NAMES`). Export or mirror the spec from your CVAT task so IDs and names stay consistent.

3. **Function name / namespace**  
   Adjust `metadata.name` and `metadata.namespace` if your cluster or CVAT project expects different values (often `cvat` for namespace).

## Deploy (Nuclio dashboard)

Use the Nuclio UI to build and run the function on the same Docker host where the Nuclio dashboard runs (typical CVAT setup: [http://localhost:8070](http://localhost:8070)). Exact menu labels depend on your Nuclio version; the flow is **create/select project → create or import function → upload archive (or paste config) → build → wait until Ready**.

**Prerequisites**

- CVAT with serverless is up (see [Semi-automatic and Automatic Annotation](https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/)).
- You completed [Before you deploy](#before-you-deploy) (weights on disk, YAML edited for your task).
- The dashboard shows **healthy** status for the Nuclio container (see troubleshooting in the same CVAT doc).

**1. Build a ZIP the dashboard accepts**

Nuclio expects **`function.yaml` at the root of the archive**, plus `main.py` and `weights/best.pt`.

**CPU** — from `yolo-pose/nuclio`:

```bash
cd yolo-pose/nuclio
zip -r "$HOME/generic-yolo-pose-cpu.zip" function.yaml main.py weights/best.pt
```

**GPU** — Nuclio still requires the config file to be named `function.yaml`. Copy the GPU manifest to that name in a temp folder, then zip:

```bash
cd yolo-pose/nuclio
tmp=$(mktemp -d)
cp function-gpu.yaml "$tmp/function.yaml"
cp main.py "$tmp/"
mkdir -p "$tmp/weights"
cp weights/best.pt "$tmp/weights/"
( cd "$tmp" && zip -r "$HOME/generic-yolo-pose-gpu.zip" . )
rm -rf "$tmp"
```

The GPU manifest requests `nvidia.com/gpu`; the host running Nuclio must have a working GPU stack.

**2. Open the dashboard**

Browse to [http://localhost:8070](http://localhost:8070) (replace `localhost` with your server host if remote).

**3. Project**

Select or create project **`cvat`**. CVAT’s models list expects functions in that project for a standard compose install.

**4. Create / deploy the function**

- Use **Create function**, **Deploy**, or **Import** (wording varies by version).
- Choose **archive / ZIP upload** when offered, and upload the zip from step 1.
- If the UI asks for a **configuration** separately, paste the contents of `function.yaml` (CPU) or `function-gpu.yaml` (GPU), and upload `main.py` and `weights/best.pt` so paths match `spec.build.copy` in the YAML.

**5. Build and wait**

Start the build from the dashboard and wait until the function state is **ready** and logs show no errors.

**6. CVAT integration**

Open CVAT’s **Models** page and confirm the function appears. If it does not, open the function in the Nuclio dashboard and compare **Environment** and **Platform** settings with a stock CVAT function that works (for local Docker, Redis host/port and network often matter). If you use CVAT’s bundled `nuctl` deploy scripts from a full CVAT install, mirror their Redis and Docker network settings in the dashboard when needed.

### Optional: CVAT `deploy_*.sh` (uses `nuctl`)

A standard CVAT checkout includes `serverless/deploy_cpu.sh` and `serverless/deploy_gpu.sh`, which call `nuctl` and glob **`function.yaml`** (CPU) and **`function-gpu.yaml`** (GPU). Point them at this function’s `nuclio/` directory or copy these files into your CVAT `serverless/custom/` tree if you prefer CLI deploy.

## Environment variables

Set these under `spec.env` in the YAML (or override at deploy time). Values are strings in YAML.

| Variable | Meaning |
|----------|---------|
| `MODEL_PATH` | Path inside the container to the `.pt` file (default `/opt/nuclio/weights/best.pt`) |
| `DEVICE` | `cpu` — force CPU; `auto` — use CUDA if available, else CPU; `cuda` / `cuda:0` — explicit GPU |
| `YOLO_TASK` | Usually `pose` (default). Leave empty only if you rely on Ultralytics to infer the task. |
| `SKELETON_LABEL` | Parent label string in the JSON response; must match the skeleton `name` in CVAT’s spec |
| `BOX_CONF_THRESHOLD` | Default detection confidence if the request omits `threshold` |
| `KEYPOINT_CONF_THRESHOLD` | When `KEYPOINT_OUTSIDE` is enabled, keypoints with confidence below this get `outside: true` |
| `INTEGER_COORDS` | `1` / `true` — integer pixel coordinates; `0` / `false` — floats |
| `KEYPOINT_OUTSIDE` | `1` / `true` — add per-keypoint `outside` and `attributes: []` (CVAT-style), using keypoint confidences when present |
| `KEYPOINT_NAMES` | Optional JSON array of sublabel names in **model keypoint order**, e.g. `'["nose","left_eye",...]'`. If unset, labels are `"1"`, `"2"`, … |
| `ULTRA_VERBOSE` | `1` — pass `verbose=True` to Ultralytics predict (noisier logs) |

**function.yaml** sets `DEVICE=cpu`. **function-gpu.yaml** sets `DEVICE=auto` and enables `KEYPOINT_OUTSIDE=1` by default; edit the file you deploy to suit your task.

You can also change the `_DEFAULT_*` constants at the top of `main.py`; environment variables take precedence when set.

## HTTP request and response

**Request** (JSON):

- `image` (required): base64-encoded image (same convention as other CVAT serverless detectors).
- `threshold` (optional): box confidence; overrides the default from `BOX_CONF_THRESHOLD`.

**Response**: JSON array of skeleton objects, each with `label`, `type: "skeleton"`, `confidence`, and `elements` (list of point elements with `type`, `label`, `points`, and optionally `outside` / `attributes`).

## Troubleshooting

- **Dashboard build fails on copy** — The zip must include `weights/best.pt` at `weights/best.pt` relative to `function.yaml`, matching `spec.build.copy` in the YAML you packaged.
- **Build fails on copy** (any method) — Ensure `nuclio/weights/best.pt` exists before building, or change/remove `spec.build.copy` and supply weights another way (e.g. volume).
- **Function error in dashboard** — Open the function on [http://localhost:8070](http://localhost:8070), check **Logs** and build output; fix YAML/env, then redeploy.
- **CVAT shows wrong or empty shapes** — Align `SKELETON_LABEL` and sublabel names with `metadata.annotations.spec`; keypoint order must match your CVAT skeleton definition.
- **GPU not used** — Package `function-gpu.yaml` as `function.yaml`, confirm `nvidia.com/gpu` is available to Nuclio, and set `DEVICE=auto` or `cuda:0`. On CPU-only hosts, `auto` falls back to CPU.
