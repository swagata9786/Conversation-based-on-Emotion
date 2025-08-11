# 🎭 Emotion-Based Conversational Bot with Gemini AI

An AI-powered interactive bot that detects **human emotions via webcam**, identifies **user's language** through speech, and then engages in a natural conversation using **Google Gemini AI**.  
The bot uses **DeepFace** for emotion recognition, **speech recognition** for voice commands, and supports **multilingual conversations**.

---

## ✨ Features
- **Real-time emotion detection** via webcam using `DeepFace`
- **Face clustering** for personalized conversations
- **Language auto-detection** using `langdetect`
- **Multilingual Text-to-Speech** with `gTTS` and `pyttsx3`
- **Voice input** with `SpeechRecognition`
- **Dynamic conversation generation** using **Google Gemini AI**
- **CSV-based predefined messages** for personalized responses
- **Cluster-based matching** using K-Means for better interaction

---


---

## 🛠️ Installation
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install opencv-python deepface pyttsx3 SpeechRecognition pandas gTTS playsound scikit-learn numpy google-generativeai langdetect googletrans==4.0.0-rc1


