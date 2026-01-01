# 🤟 Sign Language Recognition AI (MediaPipe & OpenCV)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

Hệ thống nhận diện ngôn ngữ ký hiệu thời gian thực sử dụng thị giác máy tính. Dự án này cho phép máy tính hiểu được các cử chỉ tay thông qua Webcam và chuyển đổi chúng thành các ký tự tương ứng.

---

## 🌟 Tính năng nổi bật
- **Phát hiện bàn tay:** Sử dụng MediaPipe cho độ chính xác cao và độ trễ thấp.
- **Xử lý thời gian thực:** Nhận diện trực tiếp qua Webcam.
- **Dễ dàng mở rộng:** Có thể huấn luyện thêm các ký hiệu mới một cách nhanh chóng.
- **Trực quan hóa:** Hiển thị khung xương bàn tay và nhãn dự đoán ngay trên màn hình.

## 🛠 Công nghệ sử dụng
* **Ngôn ngữ:** Python
* **Thư viện chính:**
    * `OpenCV`: Xử lý hình ảnh và luồng video.
    * `MediaPipe`: Giải pháp ML của Google để theo dõi bàn tay (Hand Tracking).
    * `NumPy`: Xử lý mảng dữ liệu.
    * `Scikit-learn / TensorFlow`: (Tùy chỉnh theo model bạn dùng để train).

## 📂 Cấu trúc dự án
```text
├── main.py                    # Script chính để chạy ứng dụng nhận diện (Inference)
├── augmentation.py            # Xử lý tăng cường dữ liệu (Xoay, lật, đổi màu ảnh...)
├── crophand.py                # Thuật toán cắt vùng chứa bàn tay để tối ưu hóa đầu vào
├── mobilenetv3small.py        # Định nghĩa kiến trúc mạng MobileNetV3 (Dòng máy nhẹ)
├── requirements.txt           # Danh sách các thư viện cần cài đặt (TensorFlow, OpenCV...)
├── best_20251230_161353.keras # Model tốt nhất được lưu lại sau khi huấn luyện
└── efficientnet_b0_landmark.keras # Model sử dụng kiến trúc EfficientNet-B0
```
## 🚀 Hướng dẫn cài đặt

Để chạy dự án này trên máy cục bộ, bạn hãy thực hiện theo các bước sau:

1. Clone repository:
```
git clone https://github.com/chiquang100106/Sign_Language_AI.git
cd Sign_Language_AI
```
2. Thiết lập môi trường
Khuyên dùng Python 3.8+ và môi trường ảo để tránh xung đột thư viện:
```
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo
# Trên Windows:
venv\Scripts\activate
# Trên macOS/Linux:
source venv/bin/activate
```
3. Cài đặt thư viện
Cài đặt tất cả các phụ thuộc chỉ với một câu lệnh:
```
pip install -r requirements.txt
```
(Nếu chưa có file requirements.txt, hãy cài: pip install opencv-python mediapipe scikit-learn)

## 📖 Cách sử dụng

1. Chuẩn bị dữ liệu (Nếu muốn huấn luyện lại)
Nếu bạn muốn mở rộng tập dữ liệu hiện có, hãy sử dụng script tăng cường dữ liệu:
```
python augmentation.py
```
2. Chạy nhận diện trực tiếp
Để khởi động hệ thống nhận diện qua Webcam, bạn chỉ cần chạy file main.py:
```
python main.py
```
3. Cấu hình Model
Bạn có thể thay đổi mô hình sử dụng (EfficientNet hoặc MobileNet) bằng cách chỉnh sửa đường dẫn trong main.py:
```
# Mở main.py và tìm dòng load model
model = load_model('efficientnet_b0_landmark.keras') # Thay đổi tên file model tại đây
```
## 🛠 Quy trình kỹ thuật (Technical Pipeline)
1. Input: Thu nhận hình ảnh từ Webcam theo thời gian thực.

2. Preprocessing: Sử dụng crophand.py để định vị bàn tay, đảm bảo AI chỉ tập trung vào các đặc trưng quan trọng nhất của cử chỉ.

3. Inference: Hình ảnh sau khi cắt được đưa vào mạng Neural (MobileNetV3 hoặc EfficientNet) để phân loại.

4. Output: Hiển thị nhãn ngôn ngữ ký hiệu tương ứng trực tiếp lên màn hình.

## 🤝 Đóng góp
Mọi đóng góp nhằm cải thiện độ chính xác của model hoặc tối ưu hóa code đều được hoan nghênh. Vui lòng mở một Issue hoặc tạo Pull Request.

Tác giả: 

Võ Chí Quang

Phan Việt Hoàng Thành

Hoàng Nguyễn Duy Tâm 

Huỳnh Phúc Thịnh

Ngày cập nhật: 31/12/2025
