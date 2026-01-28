## *💤 Drowsiness Detection using YOLOv5*

This project implements a real-time drowsiness detection system using YOLOv5 and PyTorch.
A custom dataset is created using a webcam, manually labeled, and trained to detect two classes:

awake

drowsy

The trained model supports image-based detection and real-time webcam inference.

🚀 Features

Custom dataset collection using webcam

Manual annotation using LabelImg

Training YOLOv5 on custom data

Image detection

Real-time webcam detection

CUDA GPU support

🧠 Tech Stack

Python

PyTorch

YOLOv5

OpenCV

NumPy

Matplotlib

📂 Project Structure

data/
├── images/
├── labels/
yolov5/
dataset.yml
README.md

⚙️ Installation & Setup
1️⃣ Install Dependencies

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1
-f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

git clone https://github.com/ultralytics/yolov5

cd yolov5
pip install -r requirements.txt

📦 Import Libraries

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt

🧩 Load Pre-trained YOLOv5 Model

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

🖼️ Image Detection

img = 'image_path_or_url'
results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

🎥 Real-Time Detection (Webcam)

cap = cv2.VideoCapture(0)

while cap.isOpened():
ret, frame = cap.read()
results = model(frame)
cv2.imshow('YOLO Detection', np.squeeze(results.render()))

if cv2.waitKey(10) & 0xFF == ord('q'):  
    break  


cap.release()
cv2.destroyAllWindows()

📸 Dataset Collection

labels = ['awake', 'drowsy']
number_imgs = 5

Images are captured via webcam and saved under:

data/images/

Each image is stored using a UUID for uniqueness.

🏷️ Image Annotation (LabelImg)

git clone https://github.com/tzutalin/labelImg

pip install pyqt5 lxml --upgrade
cd labelImg
pyrcc5 -o libs/resources.py resources.qrc

🏋️ Training the Model

cd yolov5
python train.py --img 320 --batch 16 --epochs 500
--data dataset.yml --weights yolov5s.pt --workers 2

📥 Load Custom Trained Model

model = torch.hub.load(
'ultralytics/yolov5',
'custom',
path='yolov5/runs/train/exp15/weights/last.pt',
force_reload=True
)

✅ Test Custom Model

img = 'data/images/sample.jpg'
results = model(img)
results.print()

plt.imshow(np.squeeze(results.render()))
plt.show()

🖥️ Real-Time Detection (Custom Model)

cap = cv2.VideoCapture(0)

while cap.isOpened():
ret, frame = cap.read()
results = model(frame)
cv2.imshow('Drowsiness Detection', np.squeeze(results.render()))

if cv2.waitKey(10) & 0xFF == ord('q'):  
    break  


cap.release()
cv2.destroyAllWindows()

📊 Classes

awake

drowsy

📌 Future Improvements

Increase dataset size

Improve low-light performance

Deploy on edge devices

Integrate eye-aspect ratio (EAR)

🙌 Acknowledgements

Ultralytics YOLOv5

PyTorch Community

👨‍💻 Author

Arkojit Sen



