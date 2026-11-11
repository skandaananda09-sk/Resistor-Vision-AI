# 🔬 Resistor Vision AI
## 🌐 Live Demo

[Click here to try Resistor Vision AI](https://resistor-vision-ai-ggpjagpusne2lxfudw38qz.streamlit.app)

An AI-powered resistor colour-band recognition and resistance calculation system using Google Gemini Vision and Python.

## 🚀 Features
- Resistor image upload and camera capture
- AI-based colour-band detection using Gemini Vision
- Supports 4-band and 5-band resistors
- Automatic resistance calculation
- Tolerance, minimum and maximum resistance calculation
- Visual confidence reporting

## 🛠️ Technologies Used
- Python
- Google Gemini Vision API
- Streamlit
- OpenCV
- NumPy
- Pillow

## ⚙️ Working Principle
1. Upload or capture an image of a resistor.
2. Gemini Vision identifies the colour bands.
3. Python decodes the detected colours using standard resistor colour codes.
4. The application displays resistance, tolerance and resistance range.

## 💻 Installation

```bash
git clone https://github.com/skandaananda09-sk/Resistor-Vision-AI.git
cd Resistor-Vision-AI
pip install -r requirements.txt
```

## 🔑 API Configuration

Set your Gemini API key as an environment variable:

```bash
GEMINI_API_KEY=your_api_key
```

Never upload your actual API key to GitHub.

## ▶️ Run the Application

```bash
streamlit run app.py
```

## 📌 Project Status
Working prototype. Further validation of 5-band resistors and camera capture is pending.

## 👨‍💻 Developer
Skanda Ananda

