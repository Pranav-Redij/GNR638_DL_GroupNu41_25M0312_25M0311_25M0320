# %%
import cv2
import os
import math

# %%
def load_training_data(base_path):
    images = []
    labels = []

    # Automatically find class folders
    class_folders = sorted(
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    )

    for folder in class_folders:
        class_path = os.path.join(base_path, folder)

        # Folder name is the label
        
        if not folder.isdigit():
            label = folder
        else:    
            label = int(folder)


        for file_name in os.listdir(class_path):
            if file_name.endswith(".png"):
                img_path = os.path.join(class_path, file_name)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                images.append(img)
                labels.append(label)

    return images, labels

# STILL HAVE TO CONVERT THE DATA/LABELS TO NUMBER, FOR CALCULATING LOSS LATER

# %%
# Test the function
base_path = "Assignment_1_Datasets/data_2"

images, labels = load_training_data(base_path)

print("Total images loaded:", len(images))
print("First image shape:", images[0].shape)
print("First label:", labels[0])

# Show first image using OpenCV
#cv2.imshow("First Image", images[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()



# %%
