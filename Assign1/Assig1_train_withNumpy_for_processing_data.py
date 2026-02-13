# %%
import sys
import os
import cv2
import random
import pickle
import numpy as np #(only for data loading/processing, not for model computations)

# --- 1. Link the C++ Engine ---
# Add current directory to path so Python finds cnn_engine.so
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cnn_engine


# %%
# --- Automatic Data Loading ---
def load_and_process_data(base_path):
    print(f"Scanning directory: {base_path}")

    images, labels = [], []
    detected_size = None

    # -------- Detect Classes --------
    class_folders = sorted([
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ])

    class_to_idx = {name: idx for idx, name in enumerate(class_folders)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    num_classes = len(class_folders)

    print("Class Mapping:", class_to_idx)

    # -------- Loop Through Each Class --------
    for folder in class_folders:
        class_path = os.path.join(base_path, folder)
        label_idx = class_to_idx[folder]

        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                continue

            # -------- Remove Alpha Channel If Exists --------
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # -------- Convert BGR → RGB (Important) --------
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # -------- Detect Image Size (First Image Only) --------
            if detected_size is None:
                if len(img.shape) == 2:
                    rows, cols = img.shape
                    bands = 1
                else:
                    rows, cols, bands = img.shape

                detected_size = rows

                print("\nFirst Image Info:")
                print(f"Rows (Height): {rows}")
                print(f"Columns (Width): {cols}")
                print(f"Bands (Channels): {bands}\n")

            # -------- Resize --------
            img = cv2.resize(img, (detected_size, detected_size))

            # -------- Normalize --------
            img = img.astype(np.float32) / 255.0

            # -------- Convert Shape to [C, H, W] --------
            if len(img.shape) == 2:
                # Grayscale → [1, H, W]
                img = np.expand_dims(img, axis=0)
            else:
                # HWC → CHW
                img = np.transpose(img, (2, 0, 1))

            # -------- One-Hot Label --------
            label_vec = [0.0] * num_classes
            label_vec[label_idx] = 1.0

            images.append(img.tolist())
            labels.append(label_vec)

    print(f"Total Images Loaded: {len(images)}\n")

    return images, labels, detected_size, class_to_idx, idx_to_class


# %%
# STILL HAVE TO CONVERT THE DATA/LABELS TO NUMBER, FOR CALCULATING LOSS LATER

import pickle
import numpy as np
import cnn_engine


class CNNModel:
    def __init__(self, img_size, num_classes, in_channels):
        self.img_size = img_size
        self.num_classes = num_classes
        self.in_channels = in_channels

        self.conv1 = cnn_engine.ConvLayer(3, in_channels, 16)
        self.relu1 = cnn_engine.ReLULayer()
        self.pool1 = cnn_engine.MaxPoolLayer()


        size1 = img_size - 3 + 1
        size1 = size1 // 2  

        self.conv2 = cnn_engine.ConvLayer(3, 16, 32)
        self.relu2 = cnn_engine.ReLULayer()
        self.pool2 = cnn_engine.MaxPoolLayer()

        size2 = size1 - 3 + 1
        size2 = size2 // 2 

        self.final_side = size2

        self.flat = cnn_engine.FlattenLayer()
        self.flatten_dim = 32 * self.final_side * self.final_side

        hidden_dim = 128
        self.fc1 = cnn_engine.FCLayer(self.flatten_dim, hidden_dim)
        self.relu_fc = cnn_engine.ReLU1DLayer()
        self.fc2 = cnn_engine.FCLayer(hidden_dim, num_classes)

        self.loss = cnn_engine.Loss()

    def print_summary(self):
        # ---- Dimension Calculation ----
        size1 = self.img_size - 3 + 1
        size1_pool = size1 // 2

        size2 = size1_pool - 3 + 1
        size2_pool = size2 // 2

        # ---- Conv1 ----
        conv1_params = self.conv1.getParams()
        conv1_macs = self.conv1.getMACs(self.img_size, self.img_size)
        conv1_flops = conv1_macs * 2

        # ---- Conv2 ----
        conv2_params = self.conv2.getParams()
        conv2_macs = self.conv2.getMACs(size1_pool, size1_pool)
        conv2_flops = conv2_macs * 2

        # ---- FC1 ----
        fc1_params = self.fc1.getParams()
        fc1_macs = self.fc1.getMACs()
        fc1_flops = fc1_macs * 2

        # ---- FC2 ----
        fc2_params = self.fc2.getParams()
        fc2_macs = self.fc2.getMACs()
        fc2_flops = fc2_macs * 2

        # ---- Totals ----
        total_params = conv1_params + conv2_params + fc1_params + fc2_params
        total_macs = conv1_macs + conv2_macs + fc1_macs + fc2_macs
        total_flops = total_macs * 2

        print("\n================ CNN SUMMARY =================")
        print(f"Input Shape: [{self.in_channels}, {self.img_size}, {self.img_size}]\n")

        print("Layer Details:")
        print(f"Conv1  → Output: [16, {size1_pool}, {size1_pool}] "
            f"| Params: {conv1_params:,} "
            f"| MACs: {conv1_macs:,} "
            f"| FLOPs: {conv1_flops:,}")

        print(f"Conv2  → Output: [32, {size2_pool}, {size2_pool}] "
            f"| Params: {conv2_params:,} "
            f"| MACs: {conv2_macs:,} "
            f"| FLOPs: {conv2_flops:,}")

        print(f"FC1    → Output: [128] "
            f"| Params: {fc1_params:,} "
            f"| MACs: {fc1_macs:,} "
            f"| FLOPs: {fc1_flops:,}")

        print(f"FC2    → Output: [{self.num_classes}] "
            f"| Params: {fc2_params:,} "
            f"| MACs: {fc2_macs:,} "
            f"| FLOPs: {fc2_flops:,}")

        print("\n----------------------------------------------")
        print("TOTAL MODEL COMPLEXITY")
        print(f"Total Parameters: {total_params:,}")
        print(f"Total MACs:       {total_macs:,}")
        print(f"Total FLOPs:      {total_flops:,}")
        print("==============================================\n")


    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = x.tolist()

        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)

        out = self.flat.forward(out)

        out = self.fc1.forward(out)
        out = self.relu_fc.forward(out)
        logits = self.fc2.forward(out)

        return logits

    def train_step(self, x, target, lr):
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(target, np.ndarray):
            target = target.tolist()

        logits = self.forward(x)
        loss_val = self.loss.calculate(logits, target)
        grad = self.loss.backward(logits, target)

        # -------- Backward --------
        grad = self.fc2.backward(grad, lr)
        grad = self.relu_fc.backward(grad)
        grad = self.fc1.backward(grad, lr)

        grad = self.flat.backward(grad)

        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad, lr)

        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad, lr)

        return loss_val, logits

    def save_model(self, path):
        model_data = {
            "img_size": self.img_size,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,

            "conv1_filters": self.conv1.filters,
            "conv1_biases": self.conv1.biases,

            "conv2_filters": self.conv2.filters,
            "conv2_biases": self.conv2.biases,

            "fc1_weights": self.fc1.weights,
            "fc1_bias": self.fc1.bias,

            "fc2_weights": self.fc2.weights,
            "fc2_bias": self.fc2.bias
        }
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

    def load_model(self, path):
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.conv1.filters = model_data["conv1_filters"]
        self.conv1.biases  = model_data["conv1_biases"]

        self.conv2.filters = model_data["conv2_filters"]
        self.conv2.biases  = model_data["conv2_biases"]

        self.fc1.weights = model_data["fc1_weights"]
        self.fc1.bias    = model_data["fc1_bias"]

        self.fc2.weights = model_data["fc2_weights"]
        self.fc2.bias    = model_data["fc2_bias"]


# %%
# --- Execution ---
import time

# ---------------- DATASET 1 ----------------
start_time = time.time()

X1, Y1, size1, class_to_idx1, idx_to_class1 = load_and_process_data(
    "Assignment_1_Datasets/data_1"
)

end_time = time.time()

print(f"\nDataset 1 Loaded in {(end_time - start_time)/60:.4f} minutes")
print(f"Classes in Data_1: {len(class_to_idx1)}")
print(f"Image Size: {size1}x{size1}")
print("-" * 50)


# ---------------- DATASET 2 ----------------
start_time = time.time()

X2, Y2, size2, class_to_idx2, idx_to_class2 = load_and_process_data(
    "Assignment_1_Datasets/data_2"
)

end_time = time.time()

print(f"\nDataset 2 Loaded in {(end_time - start_time)/60:.4f} minutes")
print(f"Classes in Data_2: {len(class_to_idx2)}")
print(f"Image Size: {size2}x{size2}")
print(" ")
print("-" * 50)
print(" ")



#%%
def train_dataset(X, Y, img_size, num_classes, model_name):
    if not X:
        print("Dataset empty!")
        return

    import time
    import numpy as np
    import random

    start_time = time.time()

    # -------- Shuffle & Split --------
    combined = list(zip(X, Y))
    random.shuffle(combined)

    split = int(0.8 * len(combined))
    train_data = combined[:split]
    val_data = combined[split:]

    # -------- Detect Channels --------
    in_channels = len(train_data[0][0])

    # -------- Initialize Model --------
    model = CNNModel(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels
    )

    model.print_summary()

    # -------- Training Settings --------
    EPOCHS = 30
    LR = 0.005
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting Training for {model_name}...")

    # -------- Training Loop --------
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0

        random.shuffle(train_data)

        for x_img, y_label in train_data:
            loss_val, logits = model.train_step(x_img, y_label, LR)
            total_loss += loss_val

            probs = model.loss.softmax(logits)

            if np.argmax(probs) == np.argmax(y_label):
                correct += 1

        train_loss = total_loss / len(train_data)
        train_acc = 100 * correct / len(train_data)

        # -------- Validation --------
        val_loss = 0
        val_correct = 0

        for x_img, y_label in val_data:
            logits = model.forward(x_img)
            val_loss += model.loss.calculate(logits, y_label)

            probs = model.loss.softmax(logits)

            if np.argmax(probs) == np.argmax(y_label):
                val_correct += 1

        val_loss /= len(val_data)
        val_acc = 100 * val_correct / len(val_data)

        print(f"Epoch {epoch+1:3d} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # -------- Early Stopping --------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_model(f"{model_name}.pkl")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    end_time = time.time()
    print(f"\nTraining Time: {(end_time - start_time)/60:.2f} minutes")

    model.load_model(f"{model_name}.pkl")
    print(f"Training completed for {model_name}\n")


#%%
# ---------- TRAIN DATASET 1 ----------
train_dataset(
    X1,
    Y1,
    img_size=size1,
    num_classes=len(class_to_idx1),
    model_name="model_digits"
)

# ---------- TRAIN DATASET 2 ----------
train_dataset(
    X2,
    Y2,
    img_size=size2,
    num_classes=len(class_to_idx2),
    model_name="model_objects"
)


#%%


# -------- Load Dataset --------
X, Y, auto_size, class_to_idx, idx_to_class = load_and_process_data(
    "Assignment_1_Datasets/data_1"
)

num_classes = len(class_to_idx)

# -------- Decide Model Based on Size + Class Count --------
if auto_size == 28 and num_classes == 10:
    model_path = "model_digits.pkl"
    print("Detected 28x28 + 10 classes → Using DIGITS model")

elif auto_size == 32 and num_classes == 100:
    model_path = "model_objects.pkl"
    print("Detected 32x32 + 100 classes → Using OBJECTS model")

else:
    raise ValueError("Dataset does not match any trained model.")

# -------- Shuffle & Split (80/20) --------
combined = list(zip(X, Y))
random.shuffle(combined)

split = int(0.8 * len(combined))
test_data = combined[split:]

# -------- Detect Channels --------
# Loader always returns [C][H][W]
in_channels = len(X[0])

# -------- Create Model --------
model = CNNModel(
    img_size=auto_size,
    num_classes=num_classes,
    in_channels=in_channels
)

# -------- Load Saved Weights --------
model.load_model(model_path)
print("Model loaded successfully.")

# -------- Testing --------
print("\nTesting on Unseen Data...")
test_correct = 0

for x_img, y_label in test_data:
    logits = model.forward(x_img)
    probs = model.loss.softmax(logits)

    if np.argmax(probs) == np.argmax(y_label):
        test_correct += 1

accuracy = 100 * test_correct / len(test_data)
print(f"Final Test Accuracy: {accuracy:.2f}%")


# %%
