import cv2
import torch
from flask import Flask, jsonify, render_template
from datetime import datetime, timezone
import threading
from pymongo import MongoClient
from bson.objectid import ObjectId
from flask_cors import CORS
import numpy as np
import os
import easyocr

# Fix Qt warning (optional)
os.environ['QT_QPA_PLATFORM'] = 'xcb'

ocr = easyocr.Reader(['th'], gpu=True)

# --------------------
# Config 
# --------------------
API_KEY = ""  
app = Flask(__name__)
CORS(app) 

MONGO_URI = "mongodb://localhost:27017/?appName=MongoDB+Compass&directConnection=true&serverSelectionTimeoutMS=2000"
DB_NAME = "riderdata"
COLLECTION_NAME = "violations"

# Create directory for saving violation images
VIOLATION_IMAGE_DIR = "violation_images"
os.makedirs(VIOLATION_IMAGE_DIR, exist_ok=True)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# โหลด YOLOv5 custom model
helmet_model = torch.hub.load('ultralytics/yolov5', 'custom', path="best.pt") 
plate_model = torch.hub.load('ultralytics/yolov5', 'custom', path="bestplatenigga.pt")
ALLOWED_PLATE_CLASSES = ['license-plate', 'motorcycle']

latest_results = []  # เก็บผล detection ล่าสุด
lock = threading.Lock()  # สำหรับ thread-safe

# --------------------
# Helper function
# --------------------

def preprocess_plate_image(img):
    """
    ปรับปรุงคุณภาพภาพป้ายทะเบียนก่อนส่ง OCR
    """
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ลด noise
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # เพิ่ม contrast ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # ทำ adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # ลบ noise เล็กๆ
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned


def read_plate_text(img, bbox, padding=0.15):
    """
    อ่านข้อความจากป้ายทะเบียน
    
    Args:
        img: ภาพต้นฉบับ
        bbox: [x1, y1, x2, y2]
        padding: ขยาย bbox เท่าไหร่
        
    Returns:
        dict: {'text': str, 'confidence': float, 'all_results': list}
    """
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    
    # ขยาย bbox
    width = x2 - x1
    height = y2 - y1
    pad_w = int(width * padding)
    pad_h = int(height * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    # ตัดภาพป้าย
    plate_img = img[y1:y2, x1:x2]
    
    if plate_img.size == 0:
        return {'text': '', 'confidence': 0.0, 'all_results': []}
    
    # ปรับปรุงภาพ
    processed_img = preprocess_plate_image(plate_img)
    
    # อ่าน OCR ทั้งภาพต้นฉบับและภาพที่ปรับปรุง
    try:
        results_original = ocr.readtext(plate_img, detail=1)
        results_processed = ocr.readtext(processed_img, detail=1)
    except Exception as e:
        print(f"OCR Error: {e}")
        return {'text': '', 'confidence': 0.0, 'all_results': []}
    
    # รวมผลลัพธ์
    all_texts = []
    for result in results_original + results_processed:
        bbox_ocr, text, conf = result
        if conf > 0.3:  # กรองข้อความที่ confidence ต่ำ
            # ลบ space และอักขระพิเศษที่ไม่ต้องการ
            cleaned_text = text.strip().replace(' ', '')
            all_texts.append({'text': cleaned_text, 'confidence': float(conf)})
    
    # หาข้อความที่มี confidence สูงสุด
    if all_texts:
        best_result = max(all_texts, key=lambda x: x['confidence'])
        # รวมข้อความทั้งหมดที่มี confidence > 0.5
        high_conf_texts = [r['text'] for r in all_texts if r['confidence'] > 0.5]
        combined_text = ' '.join(set(high_conf_texts))  # ลบข้อความซ้ำ
        avg_conf = np.mean([r['confidence'] for r in all_texts])
        
        return {
            'text': combined_text if combined_text else best_result['text'],
            'confidence': float(avg_conf),
            'all_results': all_texts
        }
    
    return {'text': '', 'confidence': 0.0, 'all_results': []}


def draw_bounding_boxes(img, detections, color_map=None):
    img_draw = img.copy()
    
    # Default colors for different classes
    if color_map is None:
        color_map = {
            'helmet': (0, 255, 0),      # Green
            'Helmet': (0, 255, 0),      # Green
            'no-helmet': (0, 0, 255),   # Red
            'No_helmet': (0, 0, 255),   # Red
            'plate': (255, 255, 0),     # Cyan
            'rider': (255, 0, 255),     # Magenta
            'motorcycle': (0, 255, 255) # Yellow
        }
    
    for _, row in detections.iterrows():
        # Get bounding box coordinates
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = float(row['confidence'])
        label = row['name']
        
        # Get color for this class
        color = color_map.get(label, (0, 255, 255))  # Default yellow if class not in map
        
        # Draw rectangle
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        text = f'{label}: {conf:.2f}'
        
        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        cv2.rectangle(img_draw, (x1, y1 - text_h - baseline - 10), 
                     (x1 + text_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(img_draw, text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img_draw


def detect_plate(img):
    """ตรวจจับป้ายทะเบียนและอ่านข้อความ"""
    results = plate_model(img[..., ::-1])  # BGR -> RGB
    df = results.pandas().xyxy[0]
    df = df[df['name'].isin(ALLOWED_PLATE_CLASSES)]
    plates = []
    
    for _, row in df.iterrows():
        cls = int(row['class'])
        label = row['name']
        conf = float(row['confidence'])
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # อ่านข้อความจากป้าย
        ocr_result = read_plate_text(img, [x1, y1, x2, y2])
        
        plates.append({
            'cls': cls, 
            'label': label, 
            'conf': conf,
            'bbox': [x1, y1, x2, y2],
            'plate_number': ocr_result['text'].encode('utf-8').decode('utf-8'),  # รองรับไทย
            'ocr_confidence': ocr_result['confidence'],  
            'ocr_details': ocr_result['all_results']    
        })

        
        print(f"🔍 Detected plate: {ocr_result['text']} (conf: {ocr_result['confidence']:.2f})")
    
    return plates, df


def detect_helmet(img):
    results = helmet_model(img[..., ::-1])  # BGR -> RGB
    df = results.pandas().xyxy[0]
    helmets = []
    has_no_helmet = False
    
    for _, row in df.iterrows():
        cls = int(row['class'])
        label = row['name']
        conf = float(row['confidence'])
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Check if this is a no-helmet violation
        if label.lower() in ['no-helmet', 'no_helmet']:
            has_no_helmet = True
        
        helmets.append({
            'cls': cls, 
            'label': label, 
            'conf': conf,
            'bbox': [x1, y1, x2, y2]
        })
    return helmets, df, has_no_helmet


# --------------------
# Webcam thread
# --------------------
def webcam_loop():
    global latest_results
    url = "http://100.66.178.110:5000/video_feed"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("🎥 Starting webcam")
    
    # ติดตามสถานะการตรวจจับ
    no_helmet_detected_last_frame = False
    violation_saved = False  # ป้องกันการบันทึกซ้ำ
    
    while True:  
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        helmets, helmet_df, has_no_helmet = detect_helmet(frame)
        plates, plate_df = detect_plate(frame)
        
        # Draw bounding boxes on frame
        frame_with_boxes = frame.copy()
        if not helmet_df.empty:
            frame_with_boxes = draw_bounding_boxes(frame_with_boxes, helmet_df)
        
        if not plate_df.empty:
            frame_with_boxes = draw_bounding_boxes(frame_with_boxes, plate_df)
            
            # วาดข้อความป้ายทะเบียนบนภาพ
            for plate in plates:
                if plate['plate_number']:
                    x1, y1, x2, y2 = plate['bbox']
                    plate_text = f"Plate: {plate['plate_number']}"
                    cv2.putText(frame_with_boxes, plate_text, (x1, y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # ตรวจสอบว่ามีการฝ่าฝืน (ไม่สวมหมวก) หรือไม่
        no_helmet_detected_this_frame = has_no_helmet
        has_motorcycle = any(p["label"] == "motorcycle" for p in plates)
        
        total_detection = 1
        
        if total_detection > 0:
            # print(f"🚗 Detect car this frame: {plate_count}, Total so far: {total_detection}")
             if (
                no_helmet_detected_this_frame 
                and has_motorcycle 
                and not no_helmet_detected_last_frame 
                and not violation_saved
            ):            
                # ลบ _id ออกจาก plates ถ้ามีติดมา
                clean_plates = []
                for p in plates:
                    p.pop("_id", None)
                    clean_plates.append(p)

                record = {
                    "timestamp": datetime.now(timezone.utc),
                    "result": helmets,
                    "has_violation": True,
                    # "image_path": image_filename,
                    "plate": plates,
                    "plate_count": len(plates),
                }
                try:
                        collection.insert_one(record)
                        violation_saved = True  # ทำเครื่องหมายว่าบันทึกแล้ว 
                        print(f"✅ Saved violation record at {record['timestamp']}")
                        print(f"   - Helmets: {len(helmets)} detections")
                        print(f"   - Plates: {len(plates)} detections")
                        if plates:
                            for i, p in enumerate(plates, 1):
                                print(f"   - Plate {i}: {p['plate_number']} (OCR conf: {p['ocr_confidence']:.2f})")
                except Exception as e:
                        print(f"❌ Insert failed: {e}")
                        violation_saved = False  # ลองบันทึกใหม่ในเฟรมถัดไป
                        
        elif no_helmet_detected_this_frame and no_helmet_detected_last_frame:
                    print("🔄 No-helmet violation still present (already saved)")
                    
        elif not no_helmet_detected_this_frame and no_helmet_detected_last_frame:
                    print("👋 No-helmet violation ended - ready to detect new violation")
                    violation_saved = False  # รีเซ็ตสำหรับการตรวจจับครั้งถัดไป
                
                # แสดงภาพ
        cv2.imshow('Helmet Detection', frame_with_boxes)
                
                # อัพเดทสถานะเฟรมก่อนหน้า
        no_helmet_detected_last_frame = no_helmet_detected_this_frame
                
        with lock:
                latest_results = helmets
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("🛑 Webcam stopped")

# --------------------
# Flask API Endpoint
# --------------------
@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/violations', methods=['GET'])
def get_results():
    try:
        data = list(collection.find().sort("timestamp", -1).limit(10))
        for d in data:
            d["_id"] = str(d["_id"])
            d["timestamp"] = d["timestamp"].isoformat()
            d["plate_numbers"] = [p["plate_number"] for p in d.get("plate", []) if p.get("plate_number")]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------
# Run server + webcam thread python app.py
# --------------------
if __name__ == '__main__':
    t = threading.Thread(target=webcam_loop, daemon=True)
    t.start()
    
    app.run(host='0.0.0.0', port=5001)
