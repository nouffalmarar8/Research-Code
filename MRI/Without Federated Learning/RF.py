# Import necessary libraries
import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the paths to your train and test data folders for each device
device_data_dirs = {
    "Device1": {
        "train_data_dir": '../train1',
        "test_data_dir": '../test1'
    },
    "Device2": {
        "train_data_dir": '../train2',
        "test_data_dir": '../test2'
    },
    "Device3": {
        "train_data_dir": '../train3',
        "test_data_dir": '../test3'
    },
    "Device4": {
        "train_data_dir": '../train4',
        "test_data_dir": '../test4'
    },
    "Device5": {
        "train_data_dir": '../train5',
        "test_data_dir": '../test5'
    }
}

# Create empty lists to store metrics for all devices
all_precisions = []
all_accuracies = []
all_f1_scores = []
all_recalls = []

# Loop through all devices
for device_name, data_dirs in device_data_dirs.items():
    train_data_dir = data_dirs["train_data_dir"]
    test_data_dir = data_dirs["test_data_dir"]

    # Initialize empty lists to store image data and labels for training and testing
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Loop through the train folder to load images and labels
    for class_name in os.listdir(train_data_dir):
        class_dir = os.path.join(train_data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            X_train.append(image)
            y_train.append(class_name)

    # Loop through the test folder to load test images and labels
    for class_name in os.listdir(test_data_dir):
        class_dir = os.path.join(test_data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            X_test.append(image)
            y_test.append(class_name)

    # Encode class labels into numerical values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Flatten the image data
    X_train_flat = np.array(X_train).reshape(len(X_train), -1)
    X_test_flat = np.array(X_test).reshape(len(X_test), -1)

    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train_flat, y_train_encoded)

    # Make predictions on the test data
    y_pred_encoded = rf_classifier.predict(X_test_flat)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)

    # Calculate metrics for the current device
    precision = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # Append the metrics for the current device to the respective lists
    all_precisions.append(precision)
    all_accuracies.append(accuracy)
    all_f1_scores.append(f1)
    all_recalls.append(recall)

    # Print metrics for the current device
    print(f"Metrics for {device_name}:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print("\n")

# Calculate the average metrics across all devices
avg_precision = sum(all_precisions) / len(all_precisions)
avg_accuracy = sum(all_accuracies) / len(all_accuracies)
avg_f1_score = sum(all_f1_scores) / len(all_f1_scores)
avg_recall = sum(all_recalls) / len(all_recalls)

# Print the average metrics
print("Average Precision: {:.2f}%".format(avg_precision * 100))
print("Average Accuracy: {:.2f}%".format(avg_accuracy * 100))
print("Average F1 Score: {:.2f}%".format(avg_f1_score * 100))
print("Average Recall: {:.2f}%".format(avg_recall * 100))
