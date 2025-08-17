import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from pymongo import MongoClient
from paddleocr import PaddleOCR
import numpy as np
import cv2
from ultralytics import YOLO

# --------------------
# Config
# --------------------
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "HelmetDetec-Test"
COLLECTION = "riderdata"
API_KEY = ""  # Optional security key
MODEL_PATH = ""  # YOLOv8 model path

# --------------------
# Init Flask
# --------------------
app = Flask(__name__)

# --------------------
# Init MongoDB
# --------------------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["HelmetDetec-Test"]
collection = db["riderdata"]

# --------------------
# Init YOLO & OCR
# --------------------
helmet_model = YOLO(MODEL_PATH)
plate_model = YOLO()
ocr = PaddleOCR(use_angle_cls=True, lang='th')

# # process them webcams😘😘😘 not necessary
# def processwebcam():
#     cap = cv2.VideoCapture(0) # for webcam = 0
    
# --------------------
# Helpers
# --------------------
def decode_image(img_b64):
    img_data = base64.b64decode(img_b64)
    img_array = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def detect_helmet_and_plate(img):
    results = helmet_model(img)
    helmets, plates = [], []                          

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 2:  # plate
                crop = img[y1:y2, x1:x2]
                plates.append(crop)
            elif cls in [0, 1]:  # helmet / no helmet
                helmets.append({
                    "cls": cls,
                    "label": helmet_model.names[cls],
                    "conf": conf
                })
    return helmets, plates

def is_registered(plate_text):
    plate_text = plate_text.strip()
    record = collection.find_one({"plate" : plate_text})
    if record:
        return record.get("registered", False)
    return False



# --------------------
# API Endpoint
# --------------------
@app.route('/process', methods=['POST'])
def process_endpoint():
    token = request.headers.get('X-API-Key')
    if token != API_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.json
    img_b64 = data.get('image_base64')
    if not img_b64:
        return jsonify({'error': 'No image provided'}), 400

    img = decode_image(img_b64)
    helmets, plates = detect_helmet_and_plate(img)

    texts = []
    for plate in plates:
        ocr_result = ocr.ocr(plate, cls=True)
        for line in ocr_result:
            for box in line:
                text, conf = box[1]
                texts.append({'text': text, 'confidence': float(conf)})
    
    # def is_registered(plate): ยังไม่มีระบบจับป้าย

    record = {
        'timestamp': datetime.utcnow(),
        'helmetstatus': helmets,
        'texts': texts,
        'image_base64': img_b64,
        'registered' : is_registered,
        'image_base64' : img_b64
    }
    rec_id = collection.insert_one(record).inserted_id

    return jsonify({'status': 'saved', 'id': str(rec_id), 'helmets': helmets, 'texts': texts})



# --------------------
# Run
# --------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
