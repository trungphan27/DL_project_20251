# Script để chuẩn bị dataset với masked faces cho training PremiumGAN.
# 1. Sử dụng OpenCV DNN Face Detector để detect khuôn mặt
# 2. Đặt một box mask đen che vùng nửa dưới khuôn mặt (chỉ vùng mũi-miệng-cằm)
# 3. Lưu ảnh đã mask

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm
import urllib.request

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_DIR = SCRIPT_DIR.parent
SOURCE_DIR = PROJECT_DIR / "celeba_hq_256"
OUTPUT_DIR = SOURCE_DIR

WITHOUT_MASK_DIR = OUTPUT_DIR / "without_mask"
WITH_MASK_DIR = OUTPUT_DIR / "with_mask"

MODEL_DIR = SCRIPT_DIR / "models"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def setup_directories():
    WITHOUT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    WITH_MASK_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Folder created: {WITHOUT_MASK_DIR}")
    print(f"Folder created: {WITH_MASK_DIR}")


def download_face_detector_model():
    """Download OpenCV DNN Face Detector model nếu chưa có"""
    prototxt_path = MODEL_DIR / "deploy.prototxt"
    caffemodel_path = MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not prototxt_path.exists():
        print("Downloading deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, prototxt_path)
    
    if not caffemodel_path.exists():
        print("Downloading caffemodel (10MB)...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, caffemodel_path)
    
    return str(prototxt_path), str(caffemodel_path)


def copy_original_images():
    print("\n Copying original images to without_mask folder...")
    
    # Lấy danh sách các file ảnh (không bao gồm subfolder)
    image_files = [f for f in SOURCE_DIR.iterdir() 
                   if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    copied_count = 0
    skipped_count = 0
    for img_path in tqdm(image_files, desc="Copying"):
        dst_path = WITHOUT_MASK_DIR / img_path.name
        if not dst_path.exists():
            shutil.copy2(img_path, dst_path)
            copied_count += 1
        else:
            skipped_count += 1
    
    print(f"✅ Copied {copied_count} new images to {WITHOUT_MASK_DIR}")
    print(f"   Skipped {skipped_count} existing images")
    print(f"   Total images in without_mask: {len(list(WITHOUT_MASK_DIR.glob('*.jpg')))}")
    return image_files


def apply_rectangular_mask(image, face_bbox):
    """
    Áp dụng mask hình chữ nhật đen lên nửa dưới khuôn mặt.
    Chỉ che vùng mũi-miệng-cằm (khoảng 45% dưới của khuôn mặt).
    
    Args:
        image: Ảnh gốc (BGR)
        face_bbox: Bounding box của khuôn mặt (x1, y1, x2, y2)
    
    Returns:
        Ảnh đã được mask
    """
    result = image.copy()
    x1, y1, x2, y2 = map(int, face_bbox)
    
    # Tính chiều cao và chiều rộng của khuôn mặt
    face_height = y2 - y1
    face_width = x2 - x1
    
    # Vùng mask: chỉ phần dưới của khuôn mặt - lấy khoảng 45% dưới của box khuôn mặt

    mask_start_y = y1 + int(face_height * 0.55)
    mask_end_y = y2
    
    # Giữ nguyên chiều rộng của khuôn mặt
    mask_x1 = x1
    mask_x2 = x2
    
    # Đảm bảo không vượt quá biên ảnh
    mask_start_y = max(0, mask_start_y)
    mask_end_y = min(image.shape[0], mask_end_y)
    mask_x1 = max(0, mask_x1)
    mask_x2 = min(image.shape[1], mask_x2)
    
    # Đặt vùng mask thành màu đen
    result[mask_start_y:mask_end_y, mask_x1:mask_x2] = 0
    
    return result


def process_images_with_opencv_dnn():
    """
    Xử lý tất cả ảnh: detect face với OpenCV DNN và tạo mask hình chữ nhật đen.
    """
    print("\n Processing images with OpenCV DNN Face Detector...")
    
    # Download và load model
    prototxt_path, caffemodel_path = download_face_detector_model()
    
    print(" Loading Face Detector model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
 
    image_files = list(WITHOUT_MASK_DIR.glob("*.jpg"))
    
    processed_count = 0
    failed_count = 0
    no_face_count = 0
    
    for img_path in tqdm(image_files, desc="Creating masks"):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                failed_count += 1
                continue
            
            h, w = image.shape[:2]
            
            # Chuẩn bị blob cho DNN
            blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )

            net.setInput(blob)
            detections = net.forward()

            best_confidence = 0
            best_box = None
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_box = (x1, y1, x2, y2)
            
            if best_box is not None:
                # Áp dụng mask hình chữ nhật đen
                masked_image = apply_rectangular_mask(image, best_box)

                output_path = WITH_MASK_DIR / img_path.name
                cv2.imwrite(str(output_path), masked_image)
                processed_count += 1
            else:
                no_face_count += 1
                # Fallback cho ảnh CelebA: khuôn mặt thường ở giữa ảnh
                # Giả định khuôn mặt chiếm phần lớn ảnh 256x256
                masked_image = image.copy()

                mask_start_y = int(h * 0.55)
                masked_image[mask_start_y:, :] = 0
                
                output_path = WITH_MASK_DIR / img_path.name
                cv2.imwrite(str(output_path), masked_image)
                processed_count += 1
                
        except Exception as e:
            print(f"\n Error processing {img_path.name}: {e}")
            failed_count += 1
    
    print(f"\n Processed {processed_count} images")
    print(f" No face detected (using fallback): {no_face_count} images")
    print(f" Failed: {failed_count} images")
    print(f" Results saved to: {WITH_MASK_DIR}")


def main():
    print("=" * 60)
    print(" PREPARING DATASET WITH MASKED FACES")
    print(" (OpenCV DNN Face Detector + Black Rectangle Mask)")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        print(f" Not found source directory: {SOURCE_DIR}")
        return
    
    source_images = list(SOURCE_DIR.glob("*.jpg"))
    print(f"\n📊 Number of images in source: {len(source_images)}")

    setup_directories()
    copy_original_images()
    process_images_with_opencv_dnn()
    
    print("\n" + "=" * 60)
    print("COMPLETED!")
    print("=" * 60)
    print(f"\n without_mask directory: {WITHOUT_MASK_DIR}")
    print(f" with_mask directory: {WITH_MASK_DIR}")
    

if __name__ == "__main__":
    main()
