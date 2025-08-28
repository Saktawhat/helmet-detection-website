#!/usr/bin/env python3
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
from ultralytics import YOLO

# --------------------
# Config
# --------------------
API_KEY = ""  
app = Flask(__name__)
CORS(app)

MONGO_URI = "mongodb://182.52.170.115:27017"
DB_NAME = "riderdata"
COLLECTION_NAME = "violations"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt", force_reload=True) 


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
    
    count = 0
    if time:
        print("🎥 Starting webcam")
    
    # State tracking
    helmet_detected_last_frame = False
    
    while True:  
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
       
        helmets = detect_helmet(frame)
        helmet_detected_this_frame = helmets is not None and len(helmets) > 0
        
        # Only save if helmet is detected THIS frame but NOT in the previous frame
        if helmet_detected_this_frame and not helmet_detected_last_frame:
            record = {
                "_id": ObjectId(),
                "timestamp": datetime.now(timezone.utc),
                "result": helmets
            }
            try:
                collection.insert_one(record)
                print(f"✅ Saved record at {record['timestamp']} with {len(record['result'])} detections")
            except Exception as e:
                print(f"❌ Insert failed: {e}")
        elif helmet_detected_this_frame and helmet_detected_last_frame:
            print("🔄 Helmet still present (not saving)")
        elif not helmet_detected_this_frame and helmet_detected_last_frame:
            print("👋 Helmet disappeared - ready to detect new helmet")
        
        cv2.imshow('Helmet Detection', frame)
        
        helmet_detected_last_frame = helmet_detected_this_frame
        
        with lock:
            latest_results = helmets
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped")
        
    cap.release()
    cv2.destroyAllWindows()


# --------------------
# Flask API Endpoint
# --------------------
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
    t = threading.Thread(target=webcam_loop, daemon=True)
    t.start()
    
    # รัน Flask server
    app.run(host='0.0.0.0', port=5000)