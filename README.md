# American-sign-language-detection
🤟 Real-Time ASL Detection using Deep Learning 🎥🧠
This project implements a real-time American Sign Language (ASL) alphabet recognition system using a custom-trained deep learning model with OpenCV and TensorFlow/Keras. The model was trained on a dataset of 3000 images per class (A-Z), resized to 200x200 pixels for optimal performance.
DataSet:
https://www.kaggle.com/datasets

🚀 Features
📷 Live camera detection of ASL hand signs.

🧠 Custom CNN model trained on a high-volume dataset (A1–A3000, B1–B3000, ..., Z1–Z3000).

🗂️ No dependency on text label files – label mapping is handled directly in the code.

⚡ Real-time feedback with frame annotation.

🛠️ Built using TensorFlow, Keras, OpenCV, and NumPy.

🏗️ Dataset Structure
css
Copy
Edit
asl_alphabet_train/
├── A/
│   ├── A1.jpg
│   ├── ...
│   └── A3000.jpg
├── B/
│   └── ...
└── Z/
    └── Z3000.jpg
🔧 Tech Stack
Python

OpenCV

TensorFlow / Keras

NumPy

🧪 Model Training
Model trained on 200x200 color images with 26 output classes (A-Z), using a Convolutional Neural Network with dropout for generalization.

🎯 How to Run
Train the model (optional, model already provided).

Run cam.py to activate your webcam and start ASL prediction.

Press Q to quit.

📁 Folder Structure
Copy
Edit
project/
├── model/
│   └── asl_model.h5
├── cam.py
├── train_model.py
└── ...!(https://github.com/user-attachments/assets/033f99c2-9bcd-48fe-bbbc-140546fa4d3f)

💡 Future Work
Add support for dynamic gestures (e.g., "hello", "thank you").

Integrate voice output or subtitles.

Deploy as a web or mobile app.
