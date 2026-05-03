# Image Thresholding and Segmentation App

Full-stack image processing project with a Django REST API backend and React frontend.

The app is split into two parts:

- Part 1: thresholding techniques
- Part 2: unsupervised segmentation techniques

All segmentation methods in this repository are implemented from scratch with NumPy/Python logic (no external ML clustering libraries).

## Features

### Part 1: Thresholding

- Optimal (Isodata)
- Otsu
- Spectral / Multi-Otsu
- Local Adaptive Thresholding

### Part 2: Segmentation

- K-means (from scratch)
- Region Growing (from scratch)
- Agglomerative Clustering (from scratch)
- Mean Shift (from scratch)

### UI/UX

- Image upload and preview
- Tabs for Part 1 and Part 2
- Dynamic parameter controls by selected algorithm
- Original and processed output display
- API-driven processing via multipart/form-data

### Image support

- Grayscale and color (RGB) images are supported
- Segmentation runs in unsupervised mode on pixel features

## Tech Stack

- Backend: Django, Django REST Framework, NumPy, Pillow, scikit-image
- Frontend: React (Vite), Axios

## Project Structure

```text
-image-thresholding-app/
├─ backend/
│  ├─ backend/                  # Django project settings/urls
│  ├─ thresholding/             # Thresholding app + threshold endpoints
│  ├─ segmentation/             # Segmentation app + segmentation endpoints
│  │  ├─ kmeans.py
│  │  ├─ region_growing.py
│  │  ├─ agglomerative.py
│  │  ├─ mean_shift.py
│  │  ├─ segmentation_controller.py
│  │  └─ views.py
│  ├─ requirements.txt
│  └─ manage.py
└─ frontend/
   ├─ src/
   ├─ package.json
   └─ vite.config.js
```

## Prerequisites

- Python 3.10+ (recommended)
- Node.js 18+ and npm

## Setup and Run

## 1) Backend (Django API)

From repository root:

```bash
cd backend
```

Create virtual environment:

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run migrations:

```bash
python manage.py migrate
```

Start backend server:

```bash
python manage.py runserver
```

Backend runs on `http://localhost:8000`.

## 2) Frontend (React + Vite)

Open a new terminal, from repository root:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:3000`.

Vite proxy is configured to forward `/api` to `http://localhost:8000`.

## API Endpoints

Base prefix:

- `/api/`

Health:

- `GET /api/health/`

Thresholding:

- `POST /api/threshold/`

Segmentation:

- `POST /api/segment/` (optional `method` field/query; defaults to `kmeans`)
- `POST /api/segment/<method_slug>/`

Supported segmentation slugs:

- `kmeans`
- `region-growing` (or `region_growing`)
- `agglomerative`
- `mean-shift` (or `mean_shift`)

## Request Format

Use `multipart/form-data` with:

- `image` (required): uploaded image file
- additional method-specific parameters (optional)

## Segmentation Parameters

### K-means

- `k` (int, default `3`)
- `max_iterations` (int, default `100`)
- `convergence_threshold` (float, default `5.0`)

### Region Growing

- `seed_x` (int, default image center x)
- `seed_y` (int, default image center y)
- `threshold` (float, default `15.0`)
- `connectivity` (`4` or `8`, default `8`)

### Agglomerative

- `n_clusters` (int, default `4`)
- `linkage` (`ward`, `average`, `complete`, `single`; default `ward`)
- `max_samples` (int, default `8000`)

### Mean Shift

- `bandwidth` (float, optional; auto if missing)
- `quantile` (float, default `0.2`)
- `max_samples` (int, default `8000`)

## Example cURL Calls

K-means:

```bash
curl -X POST "http://localhost:8000/api/segment/kmeans/" \
  -F "image=@/path/to/image.png" \
  -F "k=4" \
  -F "max_iterations=100"
```

Region Growing:

```bash
curl -X POST "http://localhost:8000/api/segment/region-growing/" \
  -F "image=@/path/to/image.png" \
  -F "seed_x=120" \
  -F "seed_y=90" \
  -F "threshold=25" \
  -F "connectivity=8"
```

Thresholding:

```bash
curl -X POST "http://localhost:8000/api/threshold/" \
  -F "image=@/path/to/image.png" \
  -F "spectral_classes=4" \
  -F "local_block=35" \
  -F "local_offset=10"
```

## Response Shape (Typical)

Thresholding endpoint returns:

- `original_image` (base64 PNG)
- `histogram`
- `results[]` (one entry per threshold method)

Segmentation endpoint returns:

- `method`
- `description`
- `parameters`
- `original_image` (base64 PNG)
- `result_image` (base64 PNG)

Some segmentation methods may also return extra fields such as cluster visualization metadata.

## Development Notes

- Backend upload parser supports multipart and form data.
- CORS is enabled for local development.
- API payload size limits are configured in Django settings.
- The segmentation controller is designed to make adding new methods straightforward.

## How to Add a New Segmentation Method

1. Create a new module in `backend/segmentation/` (for example `my_method.py`).
2. Implement a runner function with a signature similar to:
   - `run_my_method_segmentation(image_file, **params) -> dict`
3. Register it in `backend/segmentation/segmentation_controller.py`:
   - add slug mapping in `METHOD_SLUGS`
   - add runner function in `RUNNERS`
   - parse its parameters in `parse_params`
4. Add frontend controls in `frontend/src/components/SegmentationControls.jsx`.
5. Test via UI and direct API call.

## Troubleshooting

- Frontend cannot reach backend:
  - Ensure Django is running on port `8000`
  - Ensure frontend is running on port `3000`
- `No image file provided`:
  - Send file under form key `image`
- Slow processing on very large images:
  - Lower method sample-related parameters
  - Use smaller input images during development

## License

For academic/lab use. Add your preferred license file if you plan to publish publicly.

