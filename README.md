# Lumina Breast AI 🩺✨

**Lumina Breast AI** is a full-stack, AI-powered diagnostic dashboard designed to assist radiologists and clinicians in analyzing breast ultrasound images. It combines a sleek, modern React frontend with a robust PyTorch and FastAPI backend to deliver real-time predictions, explainable AI heatmaps, and detailed clinical reports.

## 🚀 Key Features

* **Real-Time AI Inference:** Upload breast ultrasound images and receive instant classifications (Benign, Malignant, Normal) powered by a fine-tuned EfficientNet-B0 PyTorch model.
* **Explainable AI (Grad-CAM):** Demystifies the "black box" by generating gradient-weighted class activation heatmaps, showing clinicians exactly which tissue regions triggered the AI's prediction.
* **Interactive Image Viewer:** A specialized UI component allowing doctors to seamlessly toggle between the original ultrasound and the AI heatmap overlay.
* **Automated Clinical Reporting:** Generates highly structured, professional radiology reports complete with BI-RADS categorization, tumor dimensions, T-Staging, and specific sonographic findings. Ready for PDF export.
* **Human-in-the-Loop (Correction Mode):** Allows clinicians to override AI predictions, add custom medical notes, and flag cases, ensuring human oversight is always prioritized.
* **Persistent Patient Database:** Uses a local SQLite database to securely store patient records, images, and diagnostic history across sessions. Includes functionality to review or delete past cases.
* **Modern UI/UX:** Built with Tailwind CSS, featuring glass-morphism design, smooth Framer Motion transitions, and a fully functional Dark Mode toggle.

---

## 🛠️ Tech Stack

**Frontend:**
* React + Vite
* TypeScript
* Tailwind CSS (Styling & Dark Mode)
* Framer Motion (Animations)
* Lucide React (Icons)

**Backend & AI Engine:**
* Python 3
* FastAPI (REST API & CORS management)
* PyTorch (EfficientNet-B0 Model)
* `pytorch-grad-cam` & OpenCV (Heatmap Generation)
* SQLite3 (Local Database)

---

## ⚙️ Prerequisites

Before running this project, ensure you have the following installed:
* [Node.js](https://nodejs.org/) (v16 or higher)
* [Python](https://www.python.org/) (v3.8 or higher)
* `pip` (Python package manager)

---

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/lumina-breast-ai.git](https://github.com/yourusername/lumina-breast-ai.git)
cd lumina-breast-ai
```

### 2. Frontend Setup
Open a terminal in the root directory and install the Node dependencies:
```bash
npm install
```

### 3. Backend Setup
It is recommended to use a Python virtual environment.
```bash
# Create and activate a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the required Python packages
pip install fastapi uvicorn python-multipart pillow torch torchvision grad-cam opencv-python numpy
```

### 4. Place the AI Model Weights
For the AI to function, you must place your trained PyTorch model weights in the correct directory.
1. Create a folder named `models` in the root directory (if it doesn't exist).
2. Place your fine-tuned EfficientNet model file into this folder and name it `best_model.pth`.
*(Note: If the file is missing, the backend will still run using a randomly initialized head for testing purposes, but predictions will not be accurate).*

---

## 💻 Running the Application

You will need two terminal windows open to run both the frontend and the backend simultaneously.

**Terminal 1: Start the FastAPI Backend**
```bash
# Ensure your virtual environment is activated
python main.py
```

**Terminal 2: Start the React Frontend**
```bash
npm run dev
```

---

## 🏥 Workflow / How to Use
1.  **Dashboard:** Enter the patient's name and age, then drag-and-drop a breast ultrasound image (`.jpg` or `.png`).
2.  **Analysis:** View the AI's prediction. Toggle the "AI Heatmap" switch to see the Grad-CAM activation map overlaid on the tissue.
3.  **Correction:** If you disagree with the AI, click "Correction", select the right diagnosis, and add your clinical notes.
4.  **Database:** Navigate to the "Patient Database" tab on the left sidebar to view past records, access their full reports, or delete them.
5.  **Reporting:** Click "View Report" to see the auto-generated clinical document. Use your browser's print function (`Ctrl+P` or `Cmd+P`) to save it as a cleanly formatted PDF.

---

## ⚠️ Disclaimer
*Lumina Breast AI is a prototype designed for educational and research purposes only. It is not FDA-approved and should not be used as a primary diagnostic tool in a clinical setting without the direct supervision and final judgment of a certified medical professional.*
