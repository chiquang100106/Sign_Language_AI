import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# 👇 Hai dòng quan trọng con đang thiếu đây:
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import time
import os

# ==========================================
# CẤU HÌNH HỆ THỐNG (CONFIG)
# ==========================================
class Config:

    MODEL_PATH = r"model_efficientnet_b0.keras"

    LABELS = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X',
              'Y']
    IMG_SIZE = 224
    CONFIDENCE_THRESHOLD = 0.75  # Ngưỡng tự tin để hiển thị màu xanh
    SMOOTHING_FACTOR = 0.0  # Hệ số làm mượt (0.1 -> 0.9). Càng cao càng mượt nhưng trễ hơn xíu.
    CAMERA_ID = 0  # ID Camera (thường là 0 hoặc 1)
    FRAME_WIDTH = 1280  # Độ phân giải HD cho nét
    FRAME_HEIGHT = 720


# Tắt cảnh báo TensorFlow cho sạch màn hình console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==========================================
# MODULE 1: PHÁT HIỆN TAY (HAND DETECTOR)
# ==========================================
class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        # Cấu hình MediaPipe tối ưu cho tốc độ và độ chính xác
        self.hands = self.mp_hands.Hands(
            model_complexity=1,  # 1: Cân bằng, 0: Nhanh, 2: Chính xác nhất
            max_num_hands=1,  # Chỉ bắt 1 tay để tránh nhiễu
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def find_hands(self, frame):
        """Nhận diện và trả về landmarks + bounding box"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        hands_data = []
        h, w, c = frame.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Tính Bounding Box
                x_vals = [lm.x * w for lm in hand_landmarks.landmark]
                y_vals = [lm.y * h for lm in hand_landmarks.landmark]

                pad = 40  # Padding vừa đủ
                bbox = {
                    'x_min': max(0, int(min(x_vals)) - pad),
                    'y_min': max(0, int(min(y_vals)) - pad),
                    'x_max': min(w, int(max(x_vals)) + pad),
                    'y_max': min(h, int(max(y_vals)) + pad)
                }

                hands_data.append({
                    'landmarks': hand_landmarks,
                    'bbox': bbox
                })
        return hands_data, results

    def draw_styled_landmarks(self, frame, results):
        """Vẽ xương tay đẹp mắt"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())


# ==========================================
# MODULE 2: PHÂN LOẠI KÝ HIỆU (SIGN CLASSIFIER)
# ==========================================
class SignClassifier:
    def __init__(self, model_path, labels, img_size):
        print(f"⏳ Đang nạp AI Model từ: {model_path}...")
        try:
            self.model = load_model(model_path)
            print("✅ AI Model đã sẵn sàng!")
        except Exception as e:
            print(f"❌ Lỗi nạp Model: {e}")
            exit()
        self.labels = labels
        self.img_size = img_size
        # Biến để lưu xác suất cũ (cho thuật toán làm mượt)
        self.prev_probs = None

    def preprocess(self, frame, bbox):
        """Cắt ảnh và giữ nguyên giá trị gốc cho Model tự xử lý"""
        x_min, y_min = bbox['x_min'], bbox['y_min']
        x_max, y_max = bbox['x_max'], bbox['y_max']

        # Kiểm tra cắt ảnh hợp lệ
        if x_max - x_min <= 0 or y_max - y_min <= 0:
            return None

        hand_crop = frame[y_min:y_max, x_min:x_max]
        if hand_crop.size == 0: return None

        # 1. Resize về 224x224
        img = cv2.resize(hand_crop, (self.img_size, self.img_size))

        # 2. Chuyển BGR sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. GIỮ NGUYÊN (Không chia 255, không preprocess)
        # Model EfficientNet con train đã có lớp xử lý bên trong rồi.
        img_batch = np.expand_dims(img, axis=0)

        return img_batch

    def predict_with_smoothing(self, img_batch):
        """Dự đoán và áp dụng thuật toán làm mượt (Exponential Moving Average)"""
        current_probs = self.model.predict(img_batch, verbose=0)[0]

        if self.prev_probs is None:
            self.prev_probs = current_probs
        else:
            # Công thức làm mượt: Probs mới = alpha * Probs hiện tại + (1-alpha) * Probs cũ
            self.prev_probs = (Config.SMOOTHING_FACTOR * self.prev_probs +
                               (1 - Config.SMOOTHING_FACTOR) * current_probs)

        smoothed_probs = self.prev_probs
        idx = np.argmax(smoothed_probs)
        label = self.labels[idx]
        confidence = smoothed_probs[idx]
        return label, confidence


# ==========================================
# MODULE 3: CHƯƠNG TRÌNH CHÍNH (MAIN APP)
# ==========================================
class SignLanguageApp:
    def __init__(self):
        self.detector = HandDetector()
        self.classifier = SignClassifier(Config.MODEL_PATH, Config.LABELS, Config.IMG_SIZE)
        self.cap = cv2.VideoCapture(Config.CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        self.fps_start_time = 0

    def draw_ui(self, frame, bbox, label, conf, fps):
        """Vẽ giao diện chuyên nghiệp"""
        # 1. Vẽ BBox và Nhãn trên tay
        if bbox:
            color = (0, 255, 0) if conf > Config.CONFIDENCE_THRESHOLD else (0, 165, 255)

            # Vẽ khung
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min']), (bbox['x_max'], bbox['y_max']), color, 2)

            # Vẽ nền chữ (Semi-transparent)
            label_text = f"{label} ({conf * 100:.0f}%)"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (bbox['x_min'], bbox['y_min'] - 35), (bbox['x_min'] + w + 10, bbox['y_min']), color,
                          -1)

            # Viết chữ
            cv2.putText(frame, label_text, (bbox['x_min'] + 5, bbox['y_min'] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Vẽ FPS và thông tin góc màn hình
        cv2.rectangle(frame, (0, 0), (250, 50), (0, 0, 0), -1)  # Nền đen góc
        cv2.putText(frame, f"FPS: {int(fps)} | 'Q' to Exit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def run(self):
        print("🎥 Đang khởi động Camera... Vui lòng chờ!")
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)  # Lật gương
            hands_data, results = self.detector.find_hands(frame)

            label = ""
            conf = 0.0
            bbox = None

            # Chỉ xử lý nếu phát hiện tay
            if hands_data:
                hand = hands_data[0]  # Lấy tay đầu tiên
                bbox = hand['bbox']

                # Xử lý ảnh
                img_batch = self.classifier.preprocess(frame, bbox)

                if img_batch is not None:
                    # Dự đoán với làm mượt
                    label, conf = self.classifier.predict_with_smoothing(img_batch)

                # Vẽ xương (Tùy chọn, comment dòng dưới nếu muốn tắt xương)
                #self.detector.draw_styled_landmarks(frame, results)

            # Tính FPS
            fps_end_time = time.time()
            time_diff = fps_end_time - self.fps_start_time
            fps = 1 / time_diff if time_diff > 0 else 0
            self.fps_start_time = fps_end_time

            # Vẽ giao diện
            self.draw_ui(frame, bbox, label, conf, fps)

            cv2.imshow('Sign Language AI - Pro Version', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()