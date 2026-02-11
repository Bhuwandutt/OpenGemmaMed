# 🩺 Med-Gemma AI Assistant

A high-performance medical AI backend built with **FastAPI** and **Google's Med-Gemma-2b**. Optimized for low-latency clinical inference and secure data persistence using **MongoDB**.



## 🚀 Key Features

* **Asynchronous Inference**: Leverages FastAPI's `async` capabilities to handle multiple requests without blocking.
* **Medical Domain Expertise**: Powered by `google/med-gemma-2b`, fine-tuned for clinical reasoning.
* **Persistent Memory**: Automatic conversation logging in NoSQL (MongoDB) for patient history tracking.
* **Performance Metrics**: Real-time tracking of inference latency and hardware utilization.
* **Auto-Generated Docs**: Interactive API playground powered by Swagger UI.

---

## 🛠 Tech Stack

* **OS**: CachyOS (Arch Linux)
* **Language**: Python 3.11+
* **Framework**: FastAPI
* **Model**: Med-Gemma-2b (via Transformers/BitsAndBytes)
* **Database**: MongoDB (NoSQL)

---

## 📦 Installation & Setup

1. **Clone the repository**
   ```bash
   git clone [https://github.com/your-username/med-gemma-app.git](https://github.com/your-username/med-gemma-app.git)
   cd med-gemma-app