import cv2
import os

def video_to_images(video_path, output_folder, image_format='jpg', interval=5):
    # ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่หรือไม่ ถ้าไม่มีให้สร้าง
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # อ่านวิดีโอ
    cap = cv2.VideoCapture(video_path)

    # นับเฟรม
    frame_count = 0

    while True:  
        # อ่านเฟรม
        ret, frame = cap.   read()

        # ถ้าจบวิดีโอหรือไม่สามารถอ่านไฟล์ได้
        if not ret:
            break

        # เพิ่มจำนวนเฟรมที่อ่านได้
        frame_count += 1

        # ถ้าเป็นเฟรมที่ต้องการ (ตามช่วงเวลาที่กำหนด)
        if frame_count % interval == 0:
            # สร้างชื่อไฟล์รูปภาพ
            image_filename = f"{frame_count // interval:04d}.{image_format}"
            image_path = os.path.join(output_folder, image_filename)

            # บันทึกรูปภาพ
            cv2.imwrite(image_path, frame)

    # ปิดวิดีโอ
    cap.release()

# ระบุที่อยู่ของวิดีโอ
video_path = "D:\วิดีโอ\c538e495-ecd3-4cad-b57d-e4498d8944d2.mp4"

# ระบุโฟลเดอร์ปลายทางที่ต้องการเก็บรูปภาพ
output_folder = "D:\Classify\ใบเตย"

# เรียกใช้ฟังก์ชัน
video_to_images(video_path, output_folder, image_format='jpg', interval=5)