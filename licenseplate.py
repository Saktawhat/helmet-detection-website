import cv2
import torch
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
from matplotlib import pyplot as plt

# โหลดโมเดล YOLOv5 (ใส่ path ของโมเดลที่คุณ train เอง)
yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5su.pt')

# โหลด PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en+th')

# run ts pmo sbau

image_path = 'testlicense.jpg'
img = cv2.imread(image_path)

result = ocr.ocr(image_path, cls=True)

# 4. แสดงผล
for line in result[0]:
    bbox, (text, confidence) = line
    print(f"Detected text: {text} (confidence: {confidence:.2f})")

# 5. วาดกรอบผลลัพธ์
boxes = [elements[0] for elements in result[0]]   # พิกัดกรอบ
txts = [elements[1][0] for elements in result[0]] # ข้อความ
scores = [elements[1][1] for elements in result[0]] # ความมั่นใจ

# แปลงสี BGR -> RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# วาด OCR บนภาพ
annotated_img = draw_ocr(img_rgb, boxes, txts, scores, font_path='path/to/THSarabunNew.ttf')

# แสดงภาพ
plt.imshow(annotated_img)
plt.axis('off')
plt.show()
