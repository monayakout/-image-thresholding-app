# 🖼️ Image Thresholding and Segmentation App

Full-stack image processing project with a Django REST API backend and React frontend ⚙️✨

The app is split into two parts:

- 🧪 Part 1: thresholding techniques  
- 🧠 Part 2: unsupervised segmentation techniques  

All segmentation methods in this repository are implemented from scratch with NumPy/Python logic (no external ML clustering libraries) 🚫📚

---

## 🚀 Features

### 🧪 Part 1: Thresholding

- Optimal (Isodata)
- Otsu
- Spectral / Multi-Otsu
- Local Adaptive Thresholding

### 🧠 Part 2: Segmentation

- K-means (from scratch)
- Region Growing (from scratch)
- Agglomerative Clustering (from scratch)
- Mean Shift (from scratch)

---

## 🎨 UI/UX

- 📤 Image upload and preview
- 📑 Tabs for Part 1 and Part 2
- 🎛️ Dynamic parameter controls by selected algorithm
- 🖼️ Original and processed output display
- 🔗 API-driven processing via multipart/form-data

---

## 🖼️ Image support

- 🩶 Grayscale and 🟥🟩🟦 RGB images are supported
- 🤖 Segmentation runs in unsupervised mode on pixel features

---

## 🛠️ Tech Stack

- Backend: Django 🐍, Django REST Framework, NumPy, Pillow, scikit-image
- Frontend: React ⚛️ (Vite ⚡), Axios

---

## 📁 Project Structure
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



#
