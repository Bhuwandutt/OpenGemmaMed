# ⚕️ Med-Gemma Clinical Assistant

A high-performance, local medical AI assistant built on **CachyOS** using **Google's Med-Gemma-2b**. This application provides a Streamlit-based UI for clinical reasoning and uses **MongoDB** for persistent, document-based chat history.

> **⚠️ MEDICAL DISCLAIMER**: This software is for educational and research purposes only. It is NOT a clinical tool and should NOT be used for diagnosis or treatment. Always consult a licensed healthcare professional.

## 🚀 Key Features
* **Model**: Med-Gemma-2b (4-bit quantized via bitsandbytes).
* **High Performance**: Optimized for CachyOS and NVIDIA GPUs (CUDA 12.1).
* **Persistent Memory**: Uses NoSQL (MongoDB) to store clinical sessions and metadata.
* **Modern UI**: Streamlit-based chat interface with session management.

## 🛠️ Tech Stack
* **OS**: CachyOS (Arch-based)
* **Env Manager**: Mamba (Miniforge)
* **Inference**: Transformers, Accelerate, BitsAndBytes
* **Database**: MongoDB (Local)
* **Frontend**: Streamlit

## 📦 Installation

1. **Clone the repo**:
   ```bash
   git clone [https://github.com/yourusername/med-gemma-assistant.git](https://github.com/yourusername/med-gemma-assistant.git)
   cd med-gemma-assistant