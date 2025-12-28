import cv2
import numpy as np
import os
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# --- CẤU HÌNH ---
CLASS_NAME = "H"
DATA_DIR = f"D:\\sign_language\\data\\input_images\\{CLASS_NAME}"
AUGMENT_RATIO = 0.3  # Sinh thêm 30%

# --- CẤU HÌNH BIẾN HÌNH TỐI ƯU ---
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=[0.95, 1.1],
    brightness_range=[0.9, 1.1],

    # --- TỐI ƯU 1: Dùng 'reflect' để tránh bị vệt sọc ở rìa ảnh ---
    fill_mode='reflect',

    # Nếu chỉ train tay phải thì để False
    horizontal_flip=False
)


def increase_contrast(image_array):
    """ Hàm tăng tương phản CLAHE (Giữ nguyên vì đã tốt) """
    img = image_array.astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final.astype(np.float32)


def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Không tìm thấy: {DATA_DIR}")
        return

    # Lấy danh sách ảnh gốc sạch
    all_files = os.listdir(DATA_DIR)
    original_files = [f for f in all_files if f.endswith(('.jpg', '.png')) and not f.startswith('aug_')]

    num_original = len(original_files)
    if num_original == 0:
        print("❌ Không có ảnh gốc nào để augment!")
        return

    target_count = int(num_original * AUGMENT_RATIO)
    print(f"📂 Gốc: {num_original} ảnh.")
    print(f"🎯 Mục tiêu sinh thêm: {target_count} ảnh.")

    # --- TỐI ƯU 3: LOGIC CHỌN ẢNH CÔNG BẰNG (FAIR SAMPLING) ---
    # Thay vì random.choice, ta nhân bản danh sách lên để đảm bảo ảnh nào cũng được chọn
    # Ví dụ: Cần thêm 30% -> Lấy 30% đầu danh sách sau khi đã xáo trộn

    # Copy danh sách để không ảnh hưởng list gốc
    files_to_augment = original_files.copy()
    np.random.shuffle(files_to_augment)  # Xáo trộn ngẫu nhiên

    # Nếu cần sinh nhiều hơn số lượng gốc (VD: ratio 2.0), thì nhân đôi, nhân ba danh sách lên
    while len(files_to_augment) < target_count:
        extra_files = original_files.copy()
        np.random.shuffle(extra_files)
        files_to_augment.extend(extra_files)

    # Cắt lấy đúng số lượng cần thiết
    files_to_process = files_to_augment[:target_count]

    print("🚀 Bắt đầu xử lý...")
    count = 0

    for file_name in files_to_process:
        img_path = os.path.join(DATA_DIR, file_name)

        try:
            # Load và Convert
            image = load_img(img_path)
            image = img_to_array(image)

            # Tăng tương phản
            image = increase_contrast(image)
            image = image.reshape((1,) + image.shape)

            # Sinh ảnh (Augment)
            # Lưu ý: flow() là generator vô tận, nên ta dùng next() để lấy đúng 1 ảnh
            batch = next(datagen.flow(image, batch_size=1))

            aug_img = batch[0].astype('uint8')
            aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

            # Đặt tên file: aug_SốThứTự_TênGốc
            new_filename = f"aug_{count}_{file_name}"
            save_path = os.path.join(DATA_DIR, new_filename)

            cv2.imwrite(save_path, aug_img)

            count += 1
            if count % 50 == 0:
                print(f"   ---> Đã sinh {count}/{target_count} ảnh...")

        except Exception as e:
            print(f"⚠️ Lỗi file {file_name}: {e}")

    print(f"\n🎉 XONG! Tổng cộng folder giờ có {len(os.listdir(DATA_DIR))} ảnh.")


if __name__ == "__main__":
    main()