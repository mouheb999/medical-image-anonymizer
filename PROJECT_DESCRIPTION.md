# AI Diagram Generation Guide
## Medical Image Anonymization & Pathology Detection System

> **Purpose:** This document gives an AI tool (ChatGPT, Claude, Mermaid generator, etc.) every
> actor, use case, class, attribute, method, and interaction it needs to produce accurate UML
> **Use Case**, **Class**, and **Sequence** diagrams for this PFE project — without needing
> to read the source code directly.
>
> **Project:** Final-year Engineering Project (PFE) — Medical Imaging AI / Privacy-Preserving Healthcare  
> **Stack:** React 18 + Node.js/Express + Python FastAPI + MongoDB + MinIO + PyTorch (DenseNet-121)

---

## PART 1 — ACTORS & USE CASES (for Use Case Diagrams)

### 1.1 Actors

| Actor | Description |
|-------|-------------|
| **Guest** | Unauthenticated visitor. Can only reach the login and register pages. |
| **Standard User** (`utilisateur`) | Authenticated user who can upload images for anonymization and view personal history. |
| **Medical User** (`utilisateur_medical`) | Extends Standard User. Can also run advanced anonymization with parameter tuning, access the full history page, and use the pathology detector. |
| **Administrator** (`responsable`) | Full system access. Manages user accounts (enable/disable), views all logs, views global statistics. |
| **FastAPI AI Service** | External system actor. Receives image data from Node backend, runs the 7-stage anonymization pipeline or the pathology detection pipeline, and returns structured JSON. |
| **MinIO Storage** | External system actor. Stores anonymized image objects and issues presigned download URLs. |
| **MongoDB** | External system actor. Persists user accounts and anonymization audit logs. |

---

### 1.2 Use Cases — grouped by actor

#### Guest
- `UC-01` Register with role selection (standard, medical, or admin — elevated roles require a secret key)
- `UC-02` Login with email + password (receive JWT)

#### Standard User (inherits Guest post-login)
- `UC-03` Upload medical image for anonymization
- `UC-04` View anonymization result (before/after comparison, download link)
- `UC-05` View personal processing history (read-only)

#### Medical User (inherits Standard User)
- `UC-06` Upload image with advanced OCR parameter tuning (`conf_threshold`, `padding`, `border_margin`, `border_pct`)
- `UC-07` Upload chest X-ray for pathology detection
- `UC-08` View Grad-CAM heatmap overlay and pseudo-localization bounding box
- `UC-09` Read pathology findings with confidence score, severity label, and research disclaimer
- `UC-10` Download anonymized image via presigned MinIO URL
- `UC-11` Browse full per-user history with pagination

#### Administrator (inherits Medical User)
- `UC-12` List all registered users
- `UC-13` Enable or disable a user account
- `UC-14` View system-wide anonymization logs
- `UC-15` View global statistics (total images, by category, by status)

#### FastAPI AI Service (system actor, invoked by Node backend)
- `UC-16` Classify image anatomy via CLIP zero-shot (chest / dental / pelvic / skull / non-medical / other)
- `UC-17` Execute 7-stage anonymization pipeline
- `UC-18` Validate image (file type + DICOM integrity)
- `UC-19` Clean DICOM metadata (remove 12 PHI tags)
- `UC-20` Preprocess borders with CLAHE contrast enhancement
- `UC-21` Detect burned-in text via dual OCR (PaddleOCR + EasyOCR, IoU deduplication)
- `UC-22` Redact detected text regions (OpenCV TELEA inpainting)
- `UC-23` Upload anonymized image to MinIO and return presigned URL
- `UC-24` Validate chest X-ray identity (CLIP gate before pathology detection)
- `UC-25` Run DenseNet-121 (TorchXRayVision) pathology inference
- `UC-26` Generate Grad-CAM heatmap for top predicted class
- `UC-27` Extract pseudo-localization bounding box from heatmap

#### Include / Extend relationships
- `UC-03` **includes** `UC-16` (classification is mandatory before anonymization)
- `UC-03` **includes** `UC-17` (the pipeline runs after classification)
- `UC-17` **includes** `UC-18`, `UC-19`, `UC-20`, `UC-21`, `UC-22`, `UC-23` (sequential stages)
- `UC-07` **includes** `UC-24`, `UC-25`, `UC-26`, `UC-27`
- `UC-06` **extends** `UC-03` (adds tunable parameters)
- `UC-08` **extends** `UC-07` (visualization of heatmap result)

---

## PART 2 — CLASSES (for Class Diagrams)

### 2.1 Backend Domain Classes (Node.js / MongoDB)

#### `User`
*File: `backend/src/models/User.js`*
```
User
──────────────────────────────
- _id: ObjectId
- name: String [max 50]
- email: String [unique, validated]
- password: String [bcrypt, select:false]
- role: 'utilisateur' | 'utilisateur_medical' | 'responsable'
- createdAt: Date
- updatedAt: Date
──────────────────────────────
+ comparePassword(plain): Boolean
```

#### `Log`
*File: `backend/src/models/Log.js`*
```
Log
──────────────────────────────
- _id: ObjectId
- user: ObjectId [ref: User]
- originalFilename: String
- anonymizedFilename: String
- classification: String
- confidence: Number
- format: String
- isDicom: Boolean
- tagsAnonymized: Number
- paddleRegions: Number
- easyRegions: Number
- totalRegions: Number
- redacted: Number
- skipped: Number
- minioUri: String
- minioKey: String
- downloadUrl: String
- processingTime: Number
- status: 'success' | 'failed' | 'pending'
- errorMessage: String
- createdAt: Date
- updatedAt: Date
```

---

### 2.2 Backend Controller Classes (Node.js / Express)

#### `AuthController`
*File: `backend/src/controllers/authController.js`*
```
AuthController
──────────────────────────────
+ register(req, res): void
+ login(req, res): void
```

#### `AnonymizeController`
*File: `backend/src/controllers/anonymizeController.js`*
```
AnonymizeController
──────────────────────────────
+ anonymize(req, res): void
  - Forwards multipart image + OCR params to FastAPI POST /anonymize
  - Saves Log entry on success
  - Returns MinIO URL + statistics
```

#### `PathologyController`
*File: `backend/src/controllers/pathologyController.js`*
```
PathologyController
──────────────────────────────
+ detect(req, res): void
  - Forwards chest X-ray to FastAPI POST /detect-pathology
  - Returns pathologies[], heatmap (base64 PNG), bbox, summary, disclaimer
```

#### `HistoryController`
*File: `backend/src/controllers/historyController.js`*
```
HistoryController
──────────────────────────────
+ getHistory(req, res): void   [paginated, filtered by user]
+ getAll(req, res): void       [admin only, all users]
```

#### `AdminController`
*File: `backend/src/controllers/adminController.js`*
```
AdminController
──────────────────────────────
+ getUsers(req, res): void
+ toggleUser(req, res): void
+ getLogs(req, res): void
+ getStats(req, res): void
```

---

### 2.3 Backend Middleware

#### `AuthMiddleware`
*File: `backend/src/middleware/auth.js`*
```
AuthMiddleware
──────────────────────────────
+ protect(req, res, next): void         [verifies JWT, attaches req.user]
+ restrictTo(...roles)(req,res,next): void  [role-based guard]
```

#### `UploadMiddleware`
*File: `backend/src/middleware/upload.js`*
```
UploadMiddleware
──────────────────────────────
+ upload: multer.Multer  [50 MB max, accepts image/*, .dcm]
```

---

### 2.4 AI Service Classes (Python / FastAPI)

#### `PathologyDetector`
*File: `services/pathology/pathology_detector.py`*
```
PathologyDetector
──────────────────────────────
- model: torch.nn.Module          [TorchXRayVision DenseNet-121, 18-class]
- confidence_threshold: float = 0.6
- max_results: int = 2
- CONFIDENCE_CAP: float = 0.95
- CONFLICT_SUPPRESSION: Dict[str, Set[str]]
──────────────────────────────
+ __init__(confidence_threshold, max_results)
+ detect(image: ndarray) -> Dict
    Returns: { pathologies, summary, top_class_idx, disclaimer }
+ get_model() -> Module
+ _preprocess(image: ndarray) -> Tensor
+ _infer(tensor: Tensor) -> ndarray         [calibrated probabilities]
+ _postprocess(probs: ndarray) -> List[Dict]
    Each dict: { name, confidence, severity, description }
+ _build_summary(pathologies) -> str
- _load_model() -> Module
```

#### `HeatmapGenerator` (module-level functions, `services/pathology/heatmap.py`)
```
HeatmapGenerator  [module]
──────────────────────────────
Constants:
  _BBOX_THRESHOLD: float = 0.6
  _GAUSSIAN_KSIZE: int = 31
  _SHARPEN_CUTOFF: float = 0.55
  _BBOX_SHRINK: float = 0.10
  _LUNG_MASK_TOP/BOTTOM/LEFT/RIGHT: float
──────────────────────────────
+ generate_heatmap(model, image_tensor, original_image,
                   target_class_idx, alpha) -> Dict
    Returns: { overlay_image, raw_heatmap, bbox, note }
+ extract_bbox_from_heatmap(heatmap, threshold) -> List[int] | None
- _compute_gradcam(model, image_tensor, target_class_idx) -> ndarray
- _get_gradcam_target(model) -> Module
- _apply_lung_mask(heatmap) -> ndarray
- _build_overlay(original_image, heatmap, alpha) -> ndarray
- _draw_bbox_with_label(overlay, bbox, image_width) -> None
- _to_rgb(image) -> ndarray
```

#### `MedicalImageClassifier`
*File: `image_classifier/improved_medical_classifier.py`*
```
MedicalImageClassifier
──────────────────────────────
- model: CLIPModel             [openai/clip-vit-base-patch32]
- processor: CLIPProcessor
- text_prompts: List[str]     [6 prompts: chest, dental, pelvic, skull, non-medical, other]
──────────────────────────────
+ classify_image(image_path: str) -> Tuple[str, float, Dict]
    Returns: (category, confidence, metadata)
```

#### `AnonymizationPipeline`
*File: `anonymizer/pipeline.py`*
```
AnonymizationPipeline
──────────────────────────────
- classifier: MedicalImageClassifier
- validator: ImageValidator
- metadata_anonymizer: MetadataAnonymizer
- border_preprocessor: BorderPreprocessor
- text_detector: TextDetector          [PaddleOCR]
- easy_detector: EasyTextDetector      [EasyOCR]
- pixel_redactor: PixelRedactor
- storage: MinIOStorage
──────────────────────────────
+ run(image_path, params) -> Dict
    Executes stages 1-7 sequentially; returns statistics + MinIO URL
```

#### `ImageValidator`
*File: `anonymizer/image_validator.py`*
```
ImageValidator
──────────────────────────────
+ validate(image_path: str) -> ValidationResult
    Checks: file extension, magic bytes, DICOM integrity, image dimensions
```

#### `MetadataAnonymizer`
*File: `anonymizer/metadata_anonymizer.py`*
```
MetadataAnonymizer
──────────────────────────────
- PHI_TAGS: List[str]   [12 DICOM tags: PatientName, PatientID, PatientBirthDate, ...]
──────────────────────────────
+ anonymize(dicom_path: str) -> Dict
    Replaces PHI tag values with placeholders; removes overlay planes
    Returns: { tags_anonymized: int }
```

#### `BorderPreprocessor`
*File: `ocr/border_preprocessor.py`*
```
BorderPreprocessor
──────────────────────────────
+ preprocess(image: ndarray, border_pct: float = 0.20) -> ndarray
    Applies CLAHE contrast enhancement to the border strip of the image
```

#### `TextDetector` (PaddleOCR)
*File: `ocr/text_detector.py`*
```
TextDetector
──────────────────────────────
- paddle_ocr: PaddleOCR
──────────────────────────────
+ detect(image: ndarray, conf_threshold: float) -> List[BoundingBox]
```

#### `EasyTextDetector` (EasyOCR)
*File: `ocr/easy_text_detector.py`*
```
EasyTextDetector
──────────────────────────────
- reader: easyocr.Reader
──────────────────────────────
+ detect(image: ndarray, conf_threshold: float) -> List[BoundingBox]
+ merge_with(paddle_boxes, iou_threshold: float = 0.5) -> List[BoundingBox]
    IoU-based deduplication; returns merged unique box set
```

#### `PixelRedactor`
*File: `anonymizer/pixel_redactor.py`*
```
PixelRedactor
──────────────────────────────
+ redact(image: ndarray, boxes: List[BoundingBox],
         padding: int, border_margin: float) -> Tuple[ndarray, Dict]
    Fills detected text boxes with OpenCV TELEA inpainting
    Returns: (redacted_image, { redacted, skipped })
```

#### `MinIOStorage`
*File: `api/storage.py`*
```
MinIOStorage
──────────────────────────────
- client: Minio
- bucket: str
──────────────────────────────
+ upload(image_path: str, category: str, filename: str) -> str
    Uploads under <category>/<filename>, returns object key
+ get_presigned_url(key: str, expiry_hours: int = 24) -> str
```

---

### 2.5 Frontend Classes (React components)

#### `AuthContext`
```
AuthContext
──────────────────────────────
- user: { id, name, email, role } | null
- token: string | null
──────────────────────────────
+ login(email, password): Promise<void>
+ logout(): void
+ register(name, email, password, role, key): Promise<void>
```

#### Page Components (each is a React function component)
```
LoginPage          → calls AuthContext.login()
RegisterPage       → calls AuthContext.register()
Dashboard          → calls POST /api/anonymize, renders UploadZone + result
MedicalDashboard   → extends Dashboard with OCR parameter sliders
PathologyDetector  → calls POST /api/pathology, renders heatmap overlay + bbox + findings
AdminDashboard     → calls GET /api/admin/users, /logs, /stats
History            → calls GET /api/history
Result             → renders before/after viewer + download button
```

#### Shared Components
```
Navbar             → reads AuthContext, shows role-appropriate links
ProtectedRoute     → wraps routes; redirects to /login if no token; blocks by role
UploadZone         → drag-and-drop file input; previews selected image
ResultViewer       → side-by-side before/after with zoom
ThemeToggle        → switches light/dark CSS variables
```

---

### 2.6 Class Relationships Summary

```
User           "1"  ---<  "many"  Log          [aggregation: user owns logs]
AnonymizationPipeline  o-- ImageValidator
AnonymizationPipeline  o-- MetadataAnonymizer
AnonymizationPipeline  o-- BorderPreprocessor
AnonymizationPipeline  o-- TextDetector
AnonymizationPipeline  o-- EasyTextDetector
AnonymizationPipeline  o-- PixelRedactor
AnonymizationPipeline  o-- MinIOStorage
PathologyDetector      o-- HeatmapGenerator    [uses generate_heatmap]
MedicalImageClassifier <-- PathologyDetector   [dependency: CLIP gating]
AnonymizeController    --> AnonymizationPipeline   [via HTTP call to FastAPI]
PathologyController    --> PathologyDetector        [via HTTP call to FastAPI]
AuthController         --> User
HistoryController      --> Log
AdminController        --> User, Log
AuthMiddleware         --> User
PathologyDetector      ..> MedicalImageClassifier  [dependency: validates chest X-ray first]
```

---

## PART 3 — INTERACTION FLOWS (for Sequence Diagrams)

### Sequence 1 — User Registration

**Participants:** Browser, Node API (`AuthController`), MongoDB (`User`)

1. Browser → Node API: `POST /api/auth/register { name, email, password, role, adminKey? }`
2. Node API: validates role; if `responsable` or `utilisateur_medical`, checks provided secret key
3. Node API: hashes password with bcrypt (12 rounds)
4. Node API → MongoDB: `User.create({ name, email, password_hash, role })`
5. MongoDB → Node API: saved User document
6. Node API → Browser: `201 { message: "Registered successfully" }`

---

### Sequence 2 — User Login

**Participants:** Browser, Node API (`AuthController`), MongoDB (`User`)

1. Browser → Node API: `POST /api/auth/login { email, password }`
2. Node API → MongoDB: `User.findOne({ email }).select("+password")`
3. MongoDB → Node API: User document (with password hash)
4. Node API: `bcrypt.compare(password, hash)` → true/false
5. [if false] Node API → Browser: `401 Unauthorized`
6. [if true] Node API: signs JWT (HS256, 7-day expiry) with `{ id, role }`
7. Node API → Browser: `200 { token, user: { id, name, email, role } }`

---

### Sequence 3 — Medical Image Anonymization (Standard Flow)

**Participants:** Browser, Node API (`AnonymizeController`), FastAPI AI Service, MinIO, MongoDB (`Log`)

1. Browser → Node API: `POST /api/anonymize` (multipart: image file + OCR params)
2. Node API: `AuthMiddleware.protect` → verify JWT → attach `req.user`
3. Node API → FastAPI: `POST /anonymize` (multipart forward with params)
4. FastAPI: **Stage 1** — `MedicalImageClassifier.classify_image()` → `(category, confidence)`
5. FastAPI: **Stage 2** — `ImageValidator.validate()` → checks file type + DICOM integrity
6. [if DICOM] FastAPI: **Stage 3** — `MetadataAnonymizer.anonymize()` → strip 12 PHI tags
7. FastAPI: **Stage 4** — `BorderPreprocessor.preprocess()` → CLAHE on border strip
8. FastAPI: **Stage 5a** — `TextDetector.detect()` → PaddleOCR boxes
9. FastAPI: **Stage 5b** — `EasyTextDetector.detect()` → EasyOCR boxes
10. FastAPI: `EasyTextDetector.merge_with(paddle_boxes)` → deduplicated box list
11. FastAPI: **Stage 6** — `PixelRedactor.redact()` → TELEA inpainting → anonymized image
12. FastAPI: **Stage 7** — `MinIOStorage.upload()` → stores under `<category>/<filename>`
13. FastAPI: `MinIOStorage.get_presigned_url()` → 24-h download URL
14. FastAPI → Node API: `200 { status, category, stats, minioKey, downloadUrl }`
15. Node API → MongoDB: `Log.create({ user, originalFilename, ..., status:"success" })`
16. Node API → Browser: `200 { downloadUrl, stats, category }`

---

### Sequence 4 — Pathology Detection with Grad-CAM

**Participants:** Browser, Node API (`PathologyController`), FastAPI AI Service

1. Browser → Node API: `POST /api/pathology` (multipart: chest X-ray)
2. Node API: `AuthMiddleware.protect` → verify JWT
3. Node API → FastAPI: `POST /detect-pathology` (multipart forward)
4. FastAPI: `MedicalImageClassifier.classify_image()` → category check
5. [if not "chest"] FastAPI → Node API: `400 { message: "Only chest X-rays supported" }`
6. [if "chest"] FastAPI: `PathologyDetector.detect(image_np)`
   - `_preprocess()` → 224×224 tensor, normalize
   - `_infer()` → 18-class sigmoid probabilities (TorchXRayVision op_norm calibration)
   - `_postprocess()` → filter by confidence ≥ 0.6, cap at 0.95, apply conflict suppression, keep top-2, add severity + description
   - `_build_summary()` → human-readable summary string
7. FastAPI: `generate_heatmap(model, image_tensor, image_np, top_class_idx)`
   - `_compute_gradcam()` → hook `denseblock4`, backward pass, weighted sum, ReLU, normalize
   - Upsample CAM to original resolution
   - Gaussian blur (31×31 kernel)
   - `_apply_lung_mask()` → zero border activations
   - Re-normalize → sharpen (zero below 0.55)
   - `_build_overlay()` → JET colormap alpha-blend
   - `extract_bbox_from_heatmap()`:
     - Binarize at 0.6
     - Discard contour bboxes < 5% image area
     - Score by `mean_heat × bbox_area`
     - Verify bbox overlaps top-30% activation region
     - Shrink inward 10%
   - `_draw_bbox_with_label()` → green rectangle + "Model Attention Region (Approximate)"
8. FastAPI: encode overlay as base64 PNG
9. FastAPI → Node API: `200 { pathologies[], summary, heatmap (base64), bbox, localization_note, disclaimer, warning }`
10. Node API → Browser: forward full response (no storage)
11. FastAPI: `finally` block → delete temp file

---

### Sequence 5 — Admin: View Users and Toggle Status

**Participants:** Browser, Node API (`AdminController`), MongoDB (`User`)

1. Browser → Node API: `GET /api/admin/users` (JWT with role = `responsable`)
2. Node API: `AuthMiddleware.protect` + `restrictTo("responsable")`
3. Node API → MongoDB: `User.find({})` (all users)
4. MongoDB → Node API: User array
5. Node API → Browser: `200 { users[] }`
6. Browser → Node API: `PATCH /api/admin/users/:id/toggle`
7. Node API → MongoDB: `User.findByIdAndUpdate(id, { isActive: !current })`
8. Node API → Browser: `200 { message: "User status updated" }`

---

### Sequence 6 — View Processing History (Medical User)

**Participants:** Browser, Node API (`HistoryController`), MongoDB (`Log`)

1. Browser → Node API: `GET /api/history?page=1&limit=10` (JWT attached)
2. Node API: `AuthMiddleware.protect`
3. Node API → MongoDB: `Log.find({ user: req.user.id }).skip().limit().sort(-createdAt)`
4. MongoDB → Node API: Log array
5. Node API → Browser: `200 { logs[], totalCount, page }`

---

## PART 4 — ADDITIONAL DIAGRAM GUIDANCE

### 4.1 Recommended Use Case Diagram layout

Draw **one system boundary** labeled _"Medical Image Platform"_ containing all use cases.

Place actors on the outside:
- Left: `Guest`, `Standard User`, `Medical User`, `Administrator` (vertical inheritance chain)
- Right: `FastAPI AI Service`, `MinIO`, `MongoDB`

Use **generalization arrows** between user actors (Standard User extends Guest, Medical User extends Standard User, Administrator extends Medical User).

Use **«include»** arrows for mandatory sub-flows (e.g., UC-03 → UC-16 → UC-17).

Use **«extend»** arrows for optional/conditional behavior (e.g., UC-06 extends UC-03).

---

### 4.2 Recommended Class Diagram scope

Organize into three packages / swim-lanes:

**Package: Frontend**
- `AuthContext`, `LoginPage`, `RegisterPage`, `Dashboard`, `MedicalDashboard`,
  `PathologyDetector`, `AdminDashboard`, `History`, `Result`, `Navbar`,
  `ProtectedRoute`, `UploadZone`

**Package: Backend (Node.js)**
- `User`, `Log`, `AuthController`, `AnonymizeController`, `PathologyController`,
  `HistoryController`, `AdminController`, `AuthMiddleware`, `UploadMiddleware`

**Package: AI Service (Python)**
- `AnonymizationPipeline`, `MedicalImageClassifier`, `ImageValidator`,
  `MetadataAnonymizer`, `BorderPreprocessor`, `TextDetector`, `EasyTextDetector`,
  `PixelRedactor`, `MinIOStorage`, `PathologyDetector`, `HeatmapGenerator`

Show dependency arrows between packages:
- Frontend → Backend (HTTP REST)
- Backend → AI Service (HTTP multipart proxy)
- Backend → MongoDB (Mongoose)
- AI Service → MinIO (MinIO SDK)

---

### 4.3 Recommended Sequence Diagram priority

For a PFE academic defense, the three most important sequences are:

1. **Sequence 3** (anonymization) — demonstrates the full 7-stage AI pipeline, dual OCR, inpainting, MinIO storage, and MongoDB audit trail.
2. **Sequence 4** (pathology detection) — demonstrates TorchXRayVision inference, Grad-CAM, bbox extraction, and research-safe output.
3. **Sequence 2** (login) — demonstrates JWT auth flow, bcrypt verification.

Use UML lifelines for each participant. Mark async HTTP calls with open arrowheads. Mark `finally` cleanup steps with a dashed return arrow labeled "delete temp file".

---

*End of AI diagram generation guide. Use sections 1–3 as the sole source of truth for diagram content.*
