# 🧠 AI-Based Diagnostic Assistant


## 🚀 Project Overview

The **AI-Based Diagnostic Assistant** is a smart healthcare platform designed to assist doctors in analyzing medical images (X-ray, MRI, CT scans) using machine learning, explainable AI, and large language models. It delivers accurate predictions, highlights critical regions in scans, and generates diagnostic reports—enhancing the speed and quality of medical decisions.

---

## 🧩 Key Features

- 🧠 **AI-Powered Image Analysis** using CNN/YOLO with >90% accuracy.
- 🔍 **Explainable AI with GRADCAM**: Highlights key regions to improve diagnostic confidence.
- 🧾 **Automated Report Generation** using **LangChain + LLMs**.
- 🧬 **Vector-Based Image Search**: Retrieve similar past cases from database.
- 🧯 **Anomaly Detection**: Identifies irregular or abnormal scan patterns.
- 🌐 **Frontend**: Simple UI using HTML, CSS, JavaScript.
- 🌈 **Dynamic Image Display**: Toggle between HSV, RGB, BGR formats.

---

### 🔄 System Architecture

### 📊 AI Workflow
1. **Image Collection** → X-ray, MRI, CT images are preprocessed.
2. **ML Model** → CNN/YOLO models trained on labeled data.
3. **Visual Explanation** → GRADCAM overlays highlight key regions.
4. **Report Generation** → LLM generates structured diagnosis using LangChain.
5. **Image Storage** → Processed data stored in PostgreSQL database.
6. **Vector Retrieval** → Similar image cases fetched via vector database for comparison.

### 🌐 Frontend Workflow
- Built using HTML, CSS, JS.
- Image upload, preview with GRADCAM heatmap.
- View historical patient records and similar case search.

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python (Flask) |
| Machine Learning | CNN, YOLO, GRADCAM |
| Database | **PostgreSQL** |
| LLM & Reports | LangChain |
| Image Search | Vector DB  |
| Storage | Local filesystem |

---

## 🧪 Use Cases

| Use Case | Description |
|----------|-------------|
| Auto Reporting | Converts image data into a readable diagnostic report. |
| Case Comparison | Doctors compare new cases with past similar cases. |


---

## 📉 Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| Model Interpretability | Used GRADCAM to visualize predictions. |
| Image Storage | Stored locally and linked via PostgreSQL. |
| Efficient Search | Integrated vector similarity on patient image history. |
| Easy UI | Designed clean, simple web interface without frameworks. |

---

## ✅ Feasibility

- **Technical**: Scalable ML models, local PostgreSQL database for storage.
- **Operational**: Streamlined workflows for image review and diagnosis.
- **Economic**: Local storage and open-source tools reduce infrastructure cost.
- **Medical**: Assists doctors with clear visuals and automated documentation.

---
