# Helmet Detection Website

A Flask-based helmet violation detection prototype. The app uses a webcam feed, a custom YOLO model, EasyOCR, and MongoDB to detect motorcycle riders without helmets, read license plates, save violation records, and show recent detections on a web dashboard.

## Features

- Detects helmets, no-helmet violations, motorcycles, and license plates
- Reads Thai license plate text using EasyOCR
- Saves violation records to MongoDB
- Shows recent violations on a dashboard
- Includes a live stream page layout
- Draws bounding boxes and confidence labels on detected objects

## Tech Stack

- Python
- Flask
- OpenCV
- PyTorch / YOLOv5
- EasyOCR
- MongoDB
- HTML, CSS, JavaScript

## Project Structure

```txt
.
├── app.py
├── best.pt
├── requirements.txt
├── Procfile
├── templates/
│   ├── index.html
│   ├── livestream.html
│   ├── contact.html
│   └── aboutus.html
└── static/
    ├── style.css
    ├── style_livestream.css
    ├── style_contact.css
    ├── style_about.css
    └── script.js
```

## Main Requirements

The core libraries this project needs are:

```txt
flask
flask-cors
opencv-python
torch
torchvision
pymongo
numpy
easyocr
pandas
requests
pyyaml
pillow
tqdm
matplotlib
seaborn
```

The current `requirements.txt` may include extra packages from the development environment. You can keep it for compatibility, but a smaller requirements file is easier to install and maintain.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure MongoDB is running locally.

The app currently connects to:

```txt
mongodb://localhost:27017/?appName=MongoDB+Compass&directConnection=true&serverSelectionTimeoutMS=2000
```

Database name:

```txt
riderdata
```

Collection name:

```txt
detections
```

4. Make sure the YOLO model file exists:

```txt
best.pt
```

## Running The App

Start the Flask server:

```bash
python app.py
```

Then open:

```txt
http://localhost:5001
```

The dashboard fetches violation records from:

```txt
/violations
```

## Notes

- The webcam loop currently uses `cv2.VideoCapture(0)`, so it expects a local camera.
- The app opens an OpenCV window to show processed camera frames.
- The `livestream.html` page exists, but the backend does not yet expose a video streaming endpoint for the processed frames.
- EasyOCR is initialized with Thai language support.
- GPU is enabled in EasyOCR with `gpu=True`; change this to `gpu=False` if running on a machine without GPU support.

## Future Improvements

- Add a real `/video_feed` route for the live stream page
- Move MongoDB URI and model paths into environment variables
- Save violation images along with database records
- Improve duplicate violation detection
- Clean and minimize `requirements.txt`
- Add installation and deployment instructions for production

