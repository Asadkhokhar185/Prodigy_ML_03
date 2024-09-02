import os
import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# loading training images and labeling
def load_training_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resizing image to 64x64 pixels
            images.append(img)
            if 'cat' in filename:
                labels.append(0)  # Label for cats
            else:
                labels.append(1)  # Label for dogs
    return images, labels

# loading test images
def load_test_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            images.append(img)
    return images

# Paths to the folders
train_folder = 'C:/Users/PMLS/Downloads/dogs-vs-cats/train/train'  # Folder with cat and dog images
test_folder = 'C:/Users/PMLS/Downloads/dogs-vs-cats/test1/test1'    # Folder with test images

# Loading training images and labels
train_images, train_labels = load_training_images(train_folder)

# Loading test images
test_images = load_test_images(test_folder)

# Converting lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)

# Flattening the images (reshape)
train_images = train_images.reshape(len(train_images), -1)  # Flatten each training image
test_images = test_images.reshape(len(test_images), -1)  # Flatten each test image

# Spliting the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_images = scaler.transform(test_images)

# Training the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Validating the model
y_val_pred = svm.predict(X_val)
validation_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")

# Predicting on the test set
test_predictions = svm.predict(test_images)

# Saving test predictions
test_image_ids = [int(os.path.splitext(filename)[0]) for filename in os.listdir(test_folder)]

# Create a DataFrame for submission
submission_df = pd.DataFrame({'id': test_image_ids, 'label': test_predictions})

# Ensure the IDs are in ascending order for consistency
submission_df = submission_df.sort_values('id')

submission_file_path = '/mnt/data/my_submission.csv'
submission_df.to_csv(submission_file_path, index=False)

print(f"Submission file saved to {submission_file_path}")