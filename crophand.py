import cv2
import mediapipe as mp
import numpy as np
import os
import math


# --- CẤU HÌNH XỬ LÝ HÀNG LOẠT ---
CLASS_NAME = "Y"  # Tên nhãn (Chữ cái) đang xử lý
INPUT_FOLDER = f"data/raw_videos/{CLASS_NAME}"
OUTPUT_FOLDER = f"data/input_images/{CLASS_NAME}"  # Folder lưu ảnh đầu ra

# --- CẤU HÌNH QUAN TRỌNG ---
CLASS_NAME = "H"  # Tên folder con muốn lưu
INPUT_FOLDER = f"D:\\sign_language\\data\\raw_video\\{CLASS_NAME}"
OUTPUT_FOLDER = f"D:\\sign_language\\data\\input_images\\{CLASS_NAME}"


# --- CẤU HÌNH CHO EFFICIENTNET / MOBILENET ---
IMG_SIZE = 224  # Size đầu vào cho ảnh
TARGET_COUNT_PER_VIDEO = 65  # Mục tiêu: Muốn lấy khoảng 60-65 ảnh mỗi video


# Tạo thư mục đầu ra
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- KHỞI TẠO MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Cấu hình Bounding Box
DESIRED_ASPECT_RATIO = 1.0
PADDING = 40


def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)


def calculate_bounding_box(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)
    x_min = max(0, x_min - PADDING)
    y_min = max(0, y_min - PADDING)
    x_max = min(w, x_max + PADDING)
    y_max = min(h, y_max + PADDING)
    return x_min, y_min, x_max, y_max


def enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame_shape, desired_aspect_ratio):
    h, w, _ = frame_shape
    box_width = x_max - x_min
    box_height = y_max - y_min
    current_aspect_ratio = box_height / box_width
    if current_aspect_ratio < desired_aspect_ratio:
        new_height = int(box_width * desired_aspect_ratio)
        y_center = (y_min + y_max) // 2
        y_min = max(0, y_center - new_height // 2)
        y_max = min(h, y_center + new_height // 2)
    elif current_aspect_ratio > desired_aspect_ratio:
        new_width = int(box_height / desired_aspect_ratio)
        x_center = (x_min + x_max) // 2
        x_min = max(0, x_center - new_width // 2)
        x_max = min(w, x_center + new_width // 2)
    return x_min, y_min, x_max, y_max


def crop_hand(frame, x_min, y_min, x_max, y_max):
    return frame[y_min:y_max, x_min:x_max]


def main():
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Không tìm thấy thư mục: {INPUT_FOLDER}")
        return

    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov', '.MOV'))]
    print(f"📂 Tìm thấy {len(video_files)} video. Bắt đầu xử lý thông minh...")

    total_images_all_videos = 0

    for video_name in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_name)
        cap = cv2.VideoCapture(video_path)

        # --- BƯỚC 1: TÍNH TOÁN BƯỚC NHẢY (DYNAMIC STEP) ---
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            print(f"⚠️ Lỗi đọc video {video_name}, bỏ qua.")
            continue

        # Tính toán: Cần nhảy bao nhiêu frame để lấy đủ số lượng mong muốn?
        # Ví dụ: 300 frame / 60 ảnh = 5 (Cứ 5 frame lấy 1)
        skip_step = max(1, int(total_frames / TARGET_COUNT_PER_VIDEO))

        print(f"▶️ Xử lý: {video_name}")
        print(f"   ℹ️ Tổng frame: {total_frames} | Mục tiêu: ~{TARGET_COUNT_PER_VIDEO} ảnh | Bước nhảy: {skip_step}")

        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # --- BƯỚC 2: KIỂM TRA ĐIỀU KIỆN LẤY ẢNH ---
            # Chỉ xử lý nếu frame hiện tại nằm trong bước nhảy
            if frame_idx % skip_step == 0:

                # --- Xử lý Detect & Crop ---
                # frame = cv2.flip(frame, 1) # Mở lại nếu cần lật ảnh

                result = process_frame(frame)

                if result.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                        if True:
                            x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)
                            x_min, y_min, x_max, y_max = enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame.shape,
                                                                              DESIRED_ASPECT_RATIO)

                            hand_crop = crop_hand(frame, x_min, y_min, x_max, y_max)

                            if hand_crop.size != 0:
                                try:
                                    # Resize về 224x224 cho EfficientNet/MobileNet
                                    hand_crop_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))

                                    # Đặt tên file
                                    filename = f"{os.path.splitext(video_name)[0]}_fr{frame_idx}.jpg"
                                    save_path = os.path.join(OUTPUT_FOLDER, filename)

                                    cv2.imwrite(save_path, hand_crop_resized)
                                    saved_count += 1

                                    # In ra mỗi 10 ảnh cho đỡ spam terminal
                                    if saved_count % 10 == 0:
                                        print(f"      ---> Đã lưu {saved_count} ảnh...")
                                except Exception as e:
                                    print(f"⚠️ Lỗi save frame {frame_idx}: {e}")

                            # Break để chỉ lấy 1 tay ưu tiên trong 1 frame (tránh trùng lặp nếu có 2 tay)
                            break

            frame_idx += 1

        cap.release()
        total_images_all_videos += saved_count
        print(f"   ✅ Xong video {video_name}. Kết quả: {saved_count} ảnh (Target: {TARGET_COUNT_PER_VIDEO})")

    cv2.destroyAllWindows()
    print(f"\n🎉 TỔNG KẾT: Đã tạo ra {total_images_all_videos} ảnh chuẩn 224x224.")


if __name__ == "__main__":
    main()