# Cài đặt các thư viện cần thiết để vẽ biểu đồ đẹp và tính toán xịn
#!pip install "protobuf==4.25.3"
#!pip install -q scikit-learn seaborn matplotlib

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(f"🔥 TensorFlow Version: {tf.__version__}")
print("✅ Đã sẵn sàng!")

# ==========================================
# CẤU HÌNH HỆ THỐNG
# ==========================================
DATA_PATH = "/kaggle/input/raw-images/input_images"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25

# ==========================================
# HÀM ĐIỀU CHỈNH TỐC ĐỘ HỌC (LR SCHEDULER)
# ==========================================
def lr_scheduler(epoch, lr):
    # 5 vòng đầu: Khởi động nhẹ (Warmup)
    if epoch < 5:
        return lr + (0.001 - 1e-5) / 5
    # Các vòng sau: Giảm dần theo hình Sin (Cosine Decay)
    else:
        return 0.001 * 0.5 * (1 + np.cos(np.pi * (epoch - 5) / (EPOCHS - 5)))

print("✅ Đã thiết lập cấu hình!")

# ==========================================
# LOAD DỮ LIỆU & TÍNH TOÁN CÂN BẰNG
# ==========================================
print("⏳ Đang đọc dữ liệu...")

# Load tập Train
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="training", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
)

# Load tập Validation
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH, validation_split=0.2, subset="validation", seed=42,
    image_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, label_mode='categorical'
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
print(f"📦 Tìm thấy {NUM_CLASSES} lớp: {class_names}")

# --- TÍNH TOÁN CLASS WEIGHTS ---
# (Giúp model chú ý kỹ hơn vào các chữ cái có ít ảnh)
print("⚖️ Đang tính toán trọng số (Class Weights)... Đợi xíu nhé!")
y_train = []
# Duyệt qua 1 vòng data để lấy nhãn (Mất khoảng 1-2 phút)
for _, labels in train_ds:
    y_train.extend(np.argmax(labels.numpy(), axis=1))

class_weights_vals = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights_vals))
print("✅ Đã cân bằng xong! Model sẽ học công bằng hơn.")

# Tối ưu hóa bộ nhớ đệm
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# XÂY DỰNG MODEL (FINE-TUNING NGAY TỪ ĐẦU)
# ==========================================
# 1. Augmentation mạnh mẽ (Giả lập môi trường xấu)
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.GaussianNoise(0.1)
])

# 2. Tải EfficientNetB0
base_model = EfficientNetB0(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')

# Mở khóa 30 lớp cuối để học sâu
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 3. Ghép nối
inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x) # Chống học vẹt

# Output layer
outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)

model = models.Model(inputs, outputs)

# 4. Compile với Label Smoothing (Chống ảo tưởng sức mạnh)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
              loss=losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

model.summary()
print("✅ Model đã lên nòng! Sẵn sàng Training!")

# ==========================================
# START TRAINING
# ==========================================
# Lưu model tốt nhất
checkpoint = callbacks.ModelCheckpoint("best_model_pro.keras", save_best_only=True, monitor='val_accuracy')
# Điều chỉnh tốc độ học
lr_callback = callbacks.LearningRateScheduler(lr_scheduler)

print("\n🚀 BẮT ĐẦU HUẤN LUYỆN (CHẾ ĐỘ PRO)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, lr_callback],
    class_weight=class_weights_dict # Áp dụng trọng số
)
print("🎉 ĐÃ TRAIN XONG!")

# Vẽ biểu đồ
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Độ chính xác (Accuracy)')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Mức độ lỗi (Loss)')
plt.show()

# ==========================================
# PHÂN TÍCH CHUYÊN SÂU
# ==========================================
print("📊 Đang tạo ma trận nhầm lẫn (Confusion Matrix)...")

# Load lại model tốt nhất (để đảm bảo không dùng cái model ở epoch cuối cùng nếu nó bị lởm)
model.load_weights("best_model_pro.keras")

y_true = []
y_pred = []

# Dự đoán toàn bộ tập validation
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Vẽ Heatmap
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Model Dự Đoán')
plt.ylabel('Thực Tế (Nhãn Đúng)')
plt.title('BẢN ĐỒ NHẦM LẪN')
plt.show()

# In báo cáo chi tiết
print(classification_report(y_true, y_pred, target_names=class_names))