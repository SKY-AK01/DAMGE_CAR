# ?? Car Damage AI — Project Overview

## What Is This Project?

An end-to-end **AI pipeline for automated car damage detection and localization**. The system will:

1. **Identify which car parts are present** in an image (hood, bumper, door, wheel, etc.)
2. **Detect which of those parts are damaged** (scratch, dent, crack, broken)
3. **Localize the damage** with segmentation masks — pixel-level damage regions

The goal is a production-ready model that can take a photo of a car and output:
- A list of detected parts with confidence scores
- A list of damaged parts with damage type
- Visual masks overlaid on the image showing exactly where the damage is

---

## ??? Two-Phase Plan

### Phase 1 — Car Parts Segmentation *(In Progress)*
Train models to segment and classify **all major car parts** from images.

- **Dataset**: Combined dataset of ~3,500 images with 16,733 annotations across **23 car part classes**
- **Models being trained**:
  | Model | Type | Purpose |
  |-------|------|---------|
  | **YOLOv11m-seg** | Real-time instance segmentation | Fast inference, production deployment |
  | **Mask R-CNN** | High-accuracy instance segmentation | Benchmark accuracy |
  | **Fast R-CNN** | Detection only (no mask) | Speed vs accuracy tradeoff |
- **Goal**: All classes achieve **mAP@50 > 0.60** before moving to Phase 2

---

### Phase 2 — Damage Detection *(Planned)*
Fine-tune Phase 1 weights on a **damage-annotated dataset**.

- Use Phase 1 car parts model as backbone/feature extractor
- Add damage classification head: `scratch`, `dent`, `crack`, `shattered`, `missing`
- Output: `{part: "hood", damage: "dent", confidence: 0.87, mask: [...]}`

---

## ??? Tech Stack

| Layer | Technology |
|-------|-----------|
| Training Cloud | Azure ML (NVIDIA A10-12Q GPU) |
| Container | Docker + CUDA 11.8 + cudnn8 |
| Framework | PyTorch 2.0.1+cu118, Ultralytics 8.4.x |
| Storage | Azure Blob Storage (dataset + model artifacts) |
| Orchestration | Python CLI `orchestrator.py` (9 options) |
| Dataset Format | COCO JSON (compatible with all 3 models) |

---

## ?? Azure ML Pipeline Flow

```
Local PC
  +-- orchestrator.py ? Option 5 (Train on Azure ML)
        +-- [STEP 1] Check/register dataset version (skip if unchanged)
        +-- [STEP 2] Build/cache Docker environment (skip if unchanged)
        +-- [STEP 3] Upload code snapshot (< 0.3 MB via .amlignore)
        +-- [STEP 4] Submit chained job ? Azure ML GPU VM
                      +-- Download dataset to VM local SSD (~15s)
                      +-- Train YOLOv11m-seg   (N epochs)
                      +-- Train Mask R-CNN     (N epochs)
                      +-- Train Fast R-CNN     (N epochs)
```

> Once submitted, closing your local terminal does NOT stop training.

---

## ?? Analytics & Metrics Strategy

### Per-Model Metrics
| Metric | YOLO | Mask R-CNN | Fast R-CNN |
|--------|------|-----------|-----------|
| mAP@50 | ? | ? | ? |
| mAP@50-95 | ? | ? | ? |
| mask_mAP | ? | ? | ? |
| Per-class AP | ? | ? | ? |
| precision / recall | ? | ? | ? |
| Epoch time (sec) | ? | ? | ? |
| GPU memory (GB) | ? | ? | ? |
| Inference FPS | planned | planned | planned |

### Azure Services (Only What We Need)
| Service | Purpose |
|---------|---------|
| **Application Insights** | Live per-epoch metric streaming (loss, mAP) to Azure Portal |
| **Azure Monitor** | GPU %, CPU %, memory on compute — zero code needed |

---

## ?? Planned Implementations

### Immediate (Pipeline Stabilization)
- [x] Single chained Azure ML job (YOLO ? Mask R-CNN ? Fast R-CNN)
- [x] Dataset download mode (VM local SSD ~500 MB/s vs FUSE 0.2 MB/s)
- [x] Live log streaming to terminal
- [x] Smart dataset version caching (no re-upload if unchanged)
- [x] .amlignore (code snapshot < 0.3 MB)
- [ ] Application Insights integration (live metric charts per epoch)
- [ ] Unified comparison table (mAP, FPS, memory — all 3 models)
- [ ] Per-class AP logging per epoch

### Phase 2 — Damage Detection
- [ ] Source/build damage annotation dataset
- [ ] Fine-tune Phase 1 weights on damage classes
- [ ] Dual-head output: part detection + damage classification
- [ ] Damage severity scoring (0–100%)
- [ ] Inference API endpoint (Azure ML managed endpoint or FastAPI)

### Production (Future)
- [ ] REST API: image in ? damage report out
- [ ] Multi-angle image support (front, rear, sides)
- [ ] Cost estimation integration (damage type ? repair cost estimate)

---

## ?? Key Decisions & Rationale

| Decision | Reason |
|----------|--------|
| Train 3 models simultaneously | Compare accuracy vs speed for production deployment choice |
| COCO format as canonical | Compatible with all 3 frameworks, industry standard |
| Single chained Azure ML job | Simpler than pipelines, all 3 models share same GPU session |
| `numpy<2` pinned | torch 2.0.1+cu118 crashes with numpy 2.x |
| `mode=download` not `mount` | Local SSD I/O = 50× faster image reads during training |
| Phase 1 before Phase 2 | Strong part segmentation is prerequisite for damage localization |
