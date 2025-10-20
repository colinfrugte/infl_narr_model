# 🧠 FastAPI Backend

This is a simple FastAPI backend hosted on **Render**.  
It serves machine learning models (Transformers, Sentence-Transformers, GLiNER) to extract **nodes and edges** from text.

---

## 🚀 Tech Stack

- **FastAPI** + **Uvicorn**
- **PyTorch**, **Transformers**, **Sentence-Transformers**
- **GLiNER** for entity extraction
- **Render** for hosting (Python 3.11, CPU)

---

## ⚙️ Run locally

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
uvicorn main:app --reload
```
