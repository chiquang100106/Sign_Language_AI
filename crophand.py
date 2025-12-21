import cv2
import mediapipe as mp
import numpy as np
import os

# --- CẤU HÌNH XỬ LÝ HÀNG LOẠT ---
CLASS_NAME = "A"  # Tên nhãn (Chữ cái) đang xử lý
INPUT_FOLDER = f"D:\\sign_language\\data\\raw_video\\{CLASS_NAME}"  # Folder chứa video đầu vào
OUTPUT_FOLDER = f"D:\\sign_language\\data\\input_images\\{CLASS_NAME}"  # Folder lưu ảnh đầu ra

IMG_SIZE = 64  # Kích thước ảnh cho CNN (Khầy khuyên nên dùng 64x64 thay vì 28x28)
FRAME_SKIP = 10  # Cứ 10 frame thì lấy 1 frame (Tránh data bị trùng lặp)

# Tạo thư mục đầu ra nếu chưa có
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# --- KHỞI TẠO MEDIAPIPE (GIỮ NGUYÊN) ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Tỷ lệ khung hình cho bounding box (GIỮ NGUYÊN)
DESIRED_ASPECT_RATIO = 1.0
PADDING = 40
STILLNESS_THRESHOLD = 5


# --- CÁC HÀM CỦA CON (GIỮ NGUYÊN 100%) ---
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


def draw_bounding_box_and_landmarks(frame, x_min, y_min, x_max, y_max, hand_landmarks):
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)


def crop_hand(frame, x_min, y_min, x_max, y_max):
    return frame[y_min:y_max, x_min:x_max]


def is_right_hand(handedness):
    # Lưu ý: Code này giả định ảnh bị ngược (Mirror) nên Left = Tay phải
    return handedness.classification[0].label == 'Left'


# --- HÀM MAIN MỚI (XỬ LÝ VIDEO TỰ ĐỘNG) ---
def main():
    # 1. Kiểm tra folder đầu vào
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Không tìm thấy thư mục video: {INPUT_FOLDER}")
        return

    # Lấy danh sách video
    video_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    print(f"📂 Tìm thấy {len(video_files)} video. Bắt đầu xử lý...")

    total_images = 0

    # 2. Duyệt qua từng video
    for video_name in video_files:
        video_path = os.path.join(INPUT_FOLDER, video_name)
        cap = cv2.VideoCapture(video_path)

        frame_idx = 0
        saved_count = 0
        print(f"▶️ Đang xử lý: {video_name}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            if frame_idx % FRAME_SKIP != 0:
                continue

            # --- SỬA 1: TẠM THỜI BỎ FLIP ĐỂ VIDEO ĐÚNG CHIỀU ---
            # frame = cv2.flip(frame, 1) # <--- Comment dòng này lại

            # Detect
            result = process_frame(frame)

            if result.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

                    # Lấy nhãn tay mà AI nhìn thấy
                    ai_label = handedness.classification[0].label

                    # --- SỬA 2: IN RA ĐỂ XEM AI ĐANG THẤY GÌ ---
                    print(f"Frame {frame_idx}: AI thấy tay '{ai_label}'")

                    # --- SỬA 3: TẠM THỜI BỎ ĐIỀU KIỆN LỌC TAY ---
                    # Cứ thấy tay là lưu hết (để test xem crop được chưa)
                    # if is_right_hand(handedness):  <--- Comment dòng này lại

                    if True:  # <--- Thay bằng True để luôn chạy
                        # Tính Box
                        x_min, y_min, x_max, y_max = calculate_bounding_box(hand_landmarks, frame.shape)
                        x_min, y_min, x_max, y_max = enforce_aspect_ratio(x_min, y_min, x_max, y_max, frame.shape,
                                                                          DESIRED_ASPECT_RATIO)

                        # Crop
                        hand_crop = crop_hand(frame, x_min, y_min, x_max, y_max)

                        if hand_crop.size != 0:
                            try:
                                hand_crop_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                                filename = f"{os.path.splitext(video_name)[0]}_frame{frame_idx}.jpg"
                                save_path = os.path.join(OUTPUT_FOLDER, filename)
                                cv2.imwrite(save_path, hand_crop_resized)
                                saved_count += 1
                                print(f"   ---> Đã lưu ảnh: {filename}")  # Báo đã lưu
                            except Exception as e:
                                print(f"Lỗi save: {e}")

        cap.release()
        print(f"   ✅ Xong video này. Đã lưu: {saved_count} ảnh.")

    cv2.destroyAllWindows()
    print(f"\n🎉 HOÀN TẤT! Tổng cộng đã tạo ra {total_images} ảnh data trong folder '{OUTPUT_FOLDER}'.")


if __name__ == "__main__":
    main()