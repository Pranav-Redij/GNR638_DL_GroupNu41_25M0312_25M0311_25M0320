# %%
import sys
import os
import cv2
import numpy as np
import random


#%%
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

    if not os.path.exists(base_path):
        print(f"Error: Path '{base_path}' not found.")
        return [], [], None

    class_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    for folder in class_folders:
        class_path = os.path.join(base_path, folder)
        try:
            label_idx = int(folder)
        except ValueError: continue 

        for file_name in os.listdir(class_path):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img = cv2.imread(os.path.join(class_path, file_name), cv2.IMREAD_GRAYSCALE)
                if img is None: continue

                # Automatically detect size from the first valid image
                if detected_size is None:
                    detected_size = img.shape[0] # Assumes square images (H)
                    print(f"Auto-detected image size: {detected_size}x{detected_size}")

                # Ensure all images are resized to the detected size of the first image
                img = cv2.resize(img, (detected_size, detected_size))
                
                img_list = (img / 255.0).tolist()
                label_vec = [0.0] * 10
                label_vec[label_idx] = 1.0

                images.append(img_list)
                labels.append(label_vec)

    return images, labels, detected_size

# STILL HAVE TO CONVERT THE DATA/LABELS TO NUMBER, FOR CALCULATING LOSS LATER

# %%
class CNNModel:
    def __init__(self, img_size):
        # Calculation: (size - kernel + 1) / pool_stride
        # We store img_size to use in the summary later
        self.img_size = img_size
        self.flatten_dim = ((img_size - 3 + 1) // 2) ** 2 
        
        self.conv1 = cnn_engine.ConvLayer(3)      
        self.relu1 = cnn_engine.ReLULayer()
        self.pool1 = cnn_engine.MaxPoolLayer()
        self.flat  = cnn_engine.FlattenLayer()
        self.fc1   = cnn_engine.FCLayer(self.flatten_dim, 10) 
        self.sig1  = cnn_engine.SigmoidLayer()
        self.loss  = cnn_engine.Loss()

    def print_summary(self):
        # Retrieve values from C++ objects
        k = self.conv1.k_size
        s = self.pool1.stride
        
        # Calculate intermediate shapes
        c_out = self.img_size - k + 1
        p_out = c_out // s
        
        # Get Stats from C++ methods
        conv_p = self.conv1.getParams()
        conv_m = self.conv1.getMACs(self.img_size, self.img_size)
        fc_p   = self.fc1.getParams()
        fc_m   = self.fc1.getMACs()

        print("\n--- CNN Model Summary ---")
        print(f"Layer      | Output Shape | Params    | MACs")
        print(f"-----------|--------------|-----------|-----------")
        print(f"Input      | {self.img_size}x{self.img_size:<10} | 0         | 0")
        print(f"ConvLayer  | {c_out}x{c_out:<10} | {conv_p:<9} | {conv_m}")
        print(f"MaxPool    | {p_out}x{p_out:<10} | 0         | 0")
        print(f"FCLayer    | 10{'':<11} | {fc_p:<9} | {fc_m}")
        print(f"-----------|--------------|-----------|-----------")
        print(f"Total Params: {conv_p + fc_p:,}")
        print(f"Total MACs:   {conv_m + fc_m:,}")
        print(f"Total FLOPs:  {(conv_m + fc_m) * 2:,} (Approx)")
        print("--------------------------------------------------\n")

    def forward(self, x):
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.flat.forward(out)
        out = self.fc1.forward(out)
        return self.sig1.forward(out)

    def train_step(self, x, target, lr):
        pred = self.forward(x)
        grad = self.loss.backward(pred, target)
        grad = self.sig1.backward(grad)
        grad = self.fc1.backward(grad, lr)
        grad = self.flat.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        self.conv1.backward(grad, lr)
        return self.loss.calculate(pred, target), pred


# %%
# --- Execution ---
X, Y, auto_size = load_and_process_data("Assignment_1_Datasets/data_1")


#%%
if X:
    # Shuffle & Split
    combined = list(zip(X, Y))
    random.shuffle(combined)
    split = int(0.8 * len(combined))
    train_data = combined[:split]
    test_data = combined[split:]

    # Initialize model with the DETECTED size
    model = CNNModel(img_size=auto_size)
    
    # NEW: Show the architecture stats before training
    model.print_summary()

    EPOCHS = 100
    LR = 0.02

    print("Starting Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        
        for x_img, y_label in train_data:
            # Note: x_img is now the normalized 2D list
            loss_val, prediction = model.train_step(x_img, y_label, LR)
            total_loss += loss_val
            if np.argmax(prediction) == np.argmax(y_label):
                correct += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / len(train_data)
            acc = 100 * correct / len(train_data)
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # Final Evaluation
    print("\nTesting on Unseen Data...")
    test_correct = 0
    for x_img, y_label in test_data:
        pred = model.forward(x_img)
        if np.argmax(pred) == np.argmax(y_label):
            test_correct += 1
    print(f"Final Test Accuracy: {100*test_correct/len(test_data):.2f}%")
# %%
