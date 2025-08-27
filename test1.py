# Python 3.10+ / TensorFlow 2.15+ / Keras 3.x
import os, json, shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

NPZ_PATH = "gestures_keras.npz"
LABEL_TXT = "gestures_keras.csv.labels.txt"
OUT_DIR   = "artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 데이터 로드
data = np.load(NPZ_PATH)
X = data["X"].astype("float32")    # (N, 63)
y = data["y"].astype("int64")      # (N,)
num_classes = int(y.max()) + 1
print("X:", X.shape, "y:", y.shape, "classes:", num_classes)

# 2) train/val split
rng = np.random.default_rng(42)
idx = np.arange(len(X)); rng.shuffle(idx)
X, y = X[idx], y[idx]
val_ratio = 0.15
n_val = int(len(X) * val_ratio)
X_train, y_train = X[n_val:], y[n_val:]
X_val,   y_val   = X[:n_val], y[:n_val]

# 3) 표준화
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-6
X_train = (X_train - mu) / sigma
X_val   = (X_val   - mu) / sigma

# 4) MLP 모델
model = keras.Sequential([
    layers.Input(shape=(63,)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.15),
    layers.Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

early = callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True)
ckpt  = callbacks.ModelCheckpoint(
    os.path.join(OUT_DIR, "gesture_keras_best.keras"),  # ✅ Keras 3 포맷
    monitor="val_accuracy",
    save_best_only=True
)

hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=64,
    callbacks=[early, ckpt],
    verbose=2
)

# 5) 평가
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"[VAL] acc={val_acc:.4f}, loss={val_loss:.4f}")

# 6) 산출물 저장
with open(os.path.join(OUT_DIR, "norm.json"), "w") as f:
    json.dump({"mu": mu.tolist(), "sigma": sigma.tolist()}, f, ensure_ascii=False)

shutil.copyfile(LABEL_TXT, os.path.join(OUT_DIR, "labels.txt"))
model.save(os.path.join(OUT_DIR, "gesture_keras_final.keras"))  # 최종본도 저장

# 7) TF‑Lite 내보내기 (권장)
best_model = keras.models.load_model(os.path.join(OUT_DIR, "gesture_keras_best.keras"))
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)  # ✅ tf.lite 사용
tflite_model = converter.convert()
with open(os.path.join(OUT_DIR, "gesture_keras.tflite"), "wb") as f:
    f.write(tflite_model)

print("[DONE] saved to:", OUT_DIR)
