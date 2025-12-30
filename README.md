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
