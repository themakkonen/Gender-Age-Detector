# Gender-Age-Detector
A Flask-based web application that uses deep learning to detect gender and age from faces in live camera feeds and uploaded images. 
This are the Models (age_net.caffemodel, gender_net.caffemodel) you can get it from the HuggingFace.
# Gender & Age Detector 👤📷

A Flask-based web application that uses deep learning to detect gender and age from a live webcam stream or uploaded images. It provides real-time predictions, supports dark mode, image capture, admin history view, and management features.

![App Screenshot](static/icons/cover.png) <!-- Optional: Replace with your screenshot path -->

## 🔍 Features

- 🎥 Real-time webcam-based detection
- 🖼️ Upload images for prediction
- 🧠 Predicts gender and exact age using pre-trained Caffe models
- 🌙 Dark mode toggle
- 💾 Capture and download results
- 🧹 Reset/Delete uploaded or captured images
- 🛠️ Admin panel for managing prediction history

## 🚀 Technologies Used

- **Python**
- **Flask**
- **OpenCV**
- **Caffe Pre-trained Deep Learning Models**
- **HTML, CSS & JavaScript**
- **Jinja2 Templating**

## 🖼️ Model Info

- **Age detection model**: `age_net.caffemodel`
- **Gender detection model**: `gender_net.caffemodel`
- Uses OpenCV’s DNN module for prediction.
- 
## 🛠️ Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/themakkonen/Gender-Age-Detector.git
   cd Gender-Age-Detector
pip install -r requirements.txt

python app.py
http://127.0.0.1:5000


## 📂 Project Structure

