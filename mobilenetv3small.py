# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:50:26.817462Z","iopub.execute_input":"2025-12-30T15:50:26.818113Z","iopub.status.idle":"2025-12-30T15:50:26.840730Z","shell.execute_reply.started":"2025-12-30T15:50:26.818082Z","shell.execute_reply":"2025-12-30T15:50:26.840082Z"}}
import os, json, random, datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:50:45.168309Z","iopub.execute_input":"2025-12-30T15:50:45.168984Z","iopub.status.idle":"2025-12-30T15:50:45.175060Z","shell.execute_reply.started":"2025-12-30T15:50:45.168955Z","shell.execute_reply":"2025-12-30T15:50:45.174499Z"}}
# ===== PATHS =====
DATASET_ROOT = "/kaggle/input/sign-language-vn"
DATA_DIR = os.path.join(DATASET_ROOT, "input_images", "input_images")

OUT_DIR = "/kaggle/working"
MODELS_DIR = os.path.join(OUT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ===== CONFIG =====
IMG_SIZE = 224
BATCH = 32
SEED = 42

EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 12

tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BEST_RAW = os.path.join(MODELS_DIR, f"best_{STAMP}.keras")
PATCHED_EXPORT = os.path.join(MODELS_DIR, f"sign_model_{STAMP}_patched.keras")
OUT_CLASSES = os.path.join(OUT_DIR, "class_names.json")

print("DATA_DIR:", DATA_DIR)


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:50:55.833124Z","iopub.execute_input":"2025-12-30T15:50:55.833426Z","iopub.status.idle":"2025-12-30T15:50:55.900226Z","shell.execute_reply.started":"2025-12-30T15:50:55.833400Z","shell.execute_reply":"2025-12-30T15:50:55.899616Z"}}
class_names = sorted([
    d for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d))
])
num_classes = len(class_names)

print("Num classes:", num_classes)
print("Classes:", class_names)

with open(OUT_CLASSES, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

paths, labels = [], []
exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for idx, cname in enumerate(class_names):
    cdir = os.path.join(DATA_DIR, cname)
    files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(exts)]
    paths.extend(files)
    labels.extend([idx] * len(files))

paths = np.array(paths)
labels = np.array(labels)

print("Total images:", len(paths))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths, labels,
    test_size=0.2,
    random_state=SEED,
    stratify=labels
)

print("Train:", len(train_paths), "Val:", len(val_paths))


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:51:14.132676Z","iopub.execute_input":"2025-12-30T15:51:14.133487Z","iopub.status.idle":"2025-12-30T15:51:14.443894Z","shell.execute_reply.started":"2025-12-30T15:51:14.133455Z","shell.execute_reply":"2025-12-30T15:51:14.443244Z"}}
AUTOTUNE = tf.data.AUTOTUNE

def decode_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    return img, tf.one_hot(label, depth=num_classes)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.15),
    tf.keras.layers.RandomContrast(0.15),
])

def aug_map(img, label):
    img = augment(img, training=True)
    return img, label

def preprocess(img, label):
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)
    return img, label

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(2000, seed=SEED)
train_ds = train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(aug_map, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(decode_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH).prefetch(AUTOTUNE)


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:51:26.113545Z","iopub.execute_input":"2025-12-30T15:51:26.113852Z","iopub.status.idle":"2025-12-30T15:51:27.034991Z","shell.execute_reply.started":"2025-12-30T15:51:26.113824Z","shell.execute_reply":"2025-12-30T15:51:27.034447Z"}}
def build_model(num_classes, weights="imagenet"):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=weights,
    )

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base

model, base = build_model(num_classes, weights="imagenet")
model.summary()


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:51:39.630170Z","iopub.execute_input":"2025-12-30T15:51:39.631082Z","iopub.status.idle":"2025-12-30T15:51:39.636085Z","shell.execute_reply.started":"2025-12-30T15:51:39.631042Z","shell.execute_reply":"2025-12-30T15:51:39.635345Z"}}
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        BEST_RAW, monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=6,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=2,
        factor=0.5, min_lr=1e-6, verbose=1
    ),
]


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T15:51:46.976329Z","iopub.execute_input":"2025-12-30T15:51:46.977010Z","iopub.status.idle":"2025-12-30T15:55:20.030832Z","shell.execute_reply.started":"2025-12-30T15:51:46.976979Z","shell.execute_reply":"2025-12-30T15:55:20.030088Z"}}
base.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

h1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
)


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T16:00:13.357418Z","iopub.execute_input":"2025-12-30T16:00:13.358085Z","iopub.status.idle":"2025-12-30T16:05:38.138465Z","shell.execute_reply.started":"2025-12-30T16:00:13.358055Z","shell.execute_reply":"2025-12-30T16:05:38.137721Z"}}
base.trainable = True
n = len(base.layers)

for layer in base.layers[: int(n * 0.60)]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

h2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD + EPOCHS_FINETUNE,
    initial_epoch=EPOCHS_HEAD,
    callbacks=callbacks,
)


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T16:06:57.542932Z","iopub.execute_input":"2025-12-30T16:06:57.543492Z","iopub.status.idle":"2025-12-30T16:06:57.858738Z","shell.execute_reply.started":"2025-12-30T16:06:57.543462Z","shell.execute_reply":"2025-12-30T16:06:57.858080Z"}}
def merge_hist(h1, h2):
    hist = {}
    for k in set(h1.history.keys()).union(h2.history.keys()):
        hist[k] = h1.history.get(k, []) + h2.history.get(k, [])
    return hist

history = merge_hist(h1, h2)
epochs = range(1, len(history["loss"]) + 1)

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(epochs, history["accuracy"], label="Train (aug)")
plt.plot(epochs, history["val_accuracy"], label="Val")
plt.axvline(EPOCHS_HEAD, linestyle="--")
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, history["loss"], label="Train (aug)")
plt.plot(epochs, history["val_loss"], label="Val")
plt.axvline(EPOCHS_HEAD, linestyle="--")
plt.title("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# %% [code] {"execution":{"iopub.status.busy":"2025-12-30T16:11:30.252757Z","iopub.execute_input":"2025-12-30T16:11:30.253131Z","iopub.status.idle":"2025-12-30T16:11:43.232478Z","shell.execute_reply.started":"2025-12-30T16:11:30.253098Z","shell.execute_reply":"2025-12-30T16:11:43.231704Z"}}
# ===== LOAD BEST MODEL =====
best_model = tf.keras.models.load_model(BEST_RAW, compile=False)

# ===== PREDICT ON VAL =====
y_true, y_pred = [], []

for xb, yb in val_ds:
    preds = best_model.predict(xb, verbose=0)
    y_true.extend(np.argmax(yb.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ===== CLASSIFICATION REPORT =====
print("=== Classification Report (Validation) ===")
print(classification_report(
    y_true, y_pred,
    target_names=class_names,
    digits=4
))

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

# giữ các class có xuất hiện trong val (để không bị hàng/cột rỗng)
support = cm.sum(axis=1)
keep = np.where(support > 0)[0]

cm2 = cm[np.ix_(keep, keep)]
names2 = [class_names[i] for i in keep]

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm2,
    annot=True,        # ✅ HIỆN SỐ HẾT
    fmt="d",           # ✅ số nguyên (0, 45, 74…)
    cmap="Blues",
    xticklabels=names2,
    yticklabels=names2,
    linewidths=0.3,
    linecolor="white",
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Validation)")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
