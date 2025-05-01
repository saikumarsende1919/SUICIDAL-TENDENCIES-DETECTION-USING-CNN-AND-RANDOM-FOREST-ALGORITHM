# SUICIDAL-TENDENCIES-DETECTION-USING-CNN-AND-RANDOM-FOREST-ALGORITHM

```markdown
# 🧠 Suicidal Tendencies Detection using CNN and Random Forest

This project aims to detect suicidal tendencies from multimodal data using deep learning (CNN) and classical machine learning (Random Forest). The model processes **text, image, and audio** inputs to determine if a subject shows suicidal behavior.

---

## 📁 Project Structure

```
SUICIDAL-TENDENCIES-DETECTION-USING-CNN-AND-RANDOM-FOREST-ALGORITHM/
├── dataset/
│   ├── Suicide_Detection.csv
│   ├── TRAIN.csv
│   ├── audio_dataset/          # <-- Add your audio data here
│   ├── image_dataset/          # <-- Add your image data here
│   ├── audio.py
│   └── image.py
├── weights/
│   ├── text.h5
│   ├── image.h5
│   ├── audio.h5
│   ├── audio.pkl
│   └── tokenizer.pkl
├── app1.py
├── requirements.txt
├── .gitattributes
└── README.md
```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/saikumarsende1919/SUICIDAL-TENDENCIES-DETECTION-USING-CNN-AND-RANDOM-FOREST-ALGORITHM.git
   cd SUICIDAL-TENDENCIES-DETECTION-USING-CNN-AND-RANDOM-FOREST-ALGORITHM
   ```

2. **Create and activate a virtual environment** (recommended):
   - On **Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On **Linux/Mac**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Make sure Git LFS is installed for large files**:
   ```bash
   git lfs install
   git lfs pull
   ```

> **⚠️ Note:**  
> If you face issues installing or running PyTorch, manually install it using the following command:
> ```bash
> pip install torch==2.5.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```
> ✅ This works for:  
> - Python 3.11  
> - CUDA 11.8  
> - Windows or Linux  
> - Torch 2.5.1  

---

## 🚀 Running the App

To start the project:
```bash
python app1.py
```

---

## 📌 Features

- 🔠 Text analysis using CNN
- 🖼️ Image analysis using CNN
- 🎵 Audio analysis using preprocessed features
- 🌲 Final prediction using Random Forest

---

## 📚 Dataset

Includes suicide-related text, image, and audio data.

### 📄 Files:
- `dataset/Suicide_Detection.csv` – Main text dataset
- `dataset/audio_dataset/` – Folder containing audio samples
- `dataset/image_dataset/` – Folder containing image samples

---

## 🧪 Models and Weights

Pre-trained models and tokenizers are stored in the `weights/` folder. Git LFS is used to manage large files.

---

## 🧑‍💻 Author

**Sai Kumar Sende**  
📫 GitHub: [@saikumarsende1919](https://github.com/saikumarsende1919)

