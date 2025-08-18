import cv2
import torch
from flask import Flask, jsonify, render_template
from datetime import datetime, timezone
import threading
import pathlib  
from pymongo import MongoClient
from bson.objectid import ObjectId
import time
from flask_cors import CORS 
temp = pathlib.PosixPath   
pathlib.PosixPath = pathlib.WindowsPath

# --------------------
# Config
# --------------------
MODEL_PATH = "best.pt"  # YOLOv5 model ตรวจหมวกอย่างเดียว
API_KEY = ""  # Optional security key
app = Flask(__name__)

MONGO_URI = "mongodb://localhost:27017/?appName=MongoDB+Compass&directConnection=true&serverSelectionTimeoutMS=2000"
  # ถ้าใช้ MongoDB Atlas ต้องใส่ URI ของ Atlas
DB_NAME = "riderdata"
COLLECTION_NAME = "violations"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# --------------------
# Load YOLOv5 model
# --------------------
helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# --------------------
# Shared results
# --------------------
latest_results = []  # เก็บผล detection ล่าสุด
lock = threading.Lock()  # สำหรับ thread-safe

# --------------------
# Helper function
# --------------------
def detect_helmet(img):
    results = helmet_model(img[..., ::-1])  # BGR -> RGB
    df = results.pandas().xyxy[0]

    helmets = []
    for _, row in df.iterrows():
        cls = int(row['class'])
        label = row['name']
        conf = float(row['confidence'])
        helmets.append({'cls': cls, 'label': label, 'conf': conf})

        
    return helmets

# --------------------
# Webcam thread
# --------------------

last_save_time = 0
SAVE_INTERVAL = 2


def webcam_loop():
    global latest_results
    cap = cv2.VideoCapture(0)  # เปิด webcam
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        helmets = detect_helmet(frame)
        
        if helmets:
            record = {
                "_id" : ObjectId(),
                "timestamp" : datetime.now(timezone.utc),
                "result" : helmets
            }
            
            try:
                collection.insert_one(record)
                last_save_time = time.time()
                print(f"✅ Inserted record at {record['timestamp']} with {len(helmets)} helmets")
            except Exception as e:
                print(f"❌ Insert failed: {e}")

        # แสดงผลบน frame
        for h in helmets:
            label = f"{h['label']} {h['conf']:.2f}"
            cv2.putText(frame, label, (10, 30 + 30*helmets.index(h)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow('Helmet Detection', frame)

        # เก็บผล detection ล่าสุด
        with lock:
            latest_results = helmets

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------
# Flask API Endpoint
# --------------------
@app.route('/helmets', methods=['GET'])
# Main dashboard route
@app.route('/')
def dashboard():
    return render_template('index.html')

# API endpoint for violations data
@app.route('/violations', methods=['GET'])
def get_violations():
    try:
        data = list(collection.find().sort("timestamp", -1).limit(10))
        # Convert ObjectId and datetime for JSON serialization
        for d in data:
            d["_id"] = str(d["_id"])
            d["timestamp"] = d["timestamp"].isoformat()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Keep your original helmets endpoint
@app.route('/helmets', methods=['GET'])
def get_results():
    try:
        data = list(collection.find().sort("timestamp", -1).limit(10))
        for d in data:
            d["_id"] = str(d["_id"])
            d["timestamp"] = d["timestamp"].isoformat()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# --------------------
# Run server + webcam thread
# --------------------
if __name__ == '__main__':
    # เริ่ม webcam ใน thread แยก
    t = threading.Thread(target=webcam_loop, daemon=True)
    t.start()

    # รัน Flask server
    app.run(host='0.0.0.0', port=5000)