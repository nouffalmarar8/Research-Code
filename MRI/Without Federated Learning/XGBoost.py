import os
import numpy as np
import cv2
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Define a function to load and preprocess images
def load_images_and_labels(data_dir):
    X = []
    y = []
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (224, 224))
            X.append(image)
            y.append(class_name)
    return X, y

# Define a function to train and evaluate a model for a device
def train_and_evaluate_device(train_data_dir, test_data_dir, device_number):
    print(f"Device {device_number} Report:")
    start_time = time.time()

    # Load training and test data
    X_train, y_train = load_images_and_labels(train_data_dir)
    X_test, y_test = load_images_and_labels(test_data_dir)

    # Encode class labels into numerical values
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Flatten the image data
    X_train_flat = np.array(X_train).reshape(len(X_train), -1)
    X_test_flat = np.array(X_test).reshape(len(X_test), -1)

    # Create an XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), random_state=42)

    # Train the classifier using the training data
    xgb_classifier.fit(X_train_flat, y_train_encoded)

    # Make predictions on the test data
    y_pred_encoded = xgb_classifier.predict(X_test_flat)

    # Decode the predicted labels from numerical values to class names
    y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred_decoded))

    accuracy = accuracy_score(y_test, y_pred_decoded)
    precision = precision_score(y_test, y_pred_decoded, average='weighted')
    recall = recall_score(y_test, y_pred_decoded, average='weighted')
    f1 = f1_score(y_test, y_pred_decoded, average='weighted')

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    return accuracy, precision, recall, f1  # Return the metrics

# Define the paths to your data folders for each device
devices = [
    {'train_data_dir': '../train1', 'test_data_dir': '../test1'},
    {'train_data_dir': '../train2', 'test_data_dir': '../test2'},
    {'train_data_dir': '../train3', 'test_data_dir': '../test3'},
    {'train_data_dir': '../train4', 'test_data_dir': '../test4'},
    {'train_data_dir': '../train5', 'test_data_dir': '../test5'}
]

# Lists to store individual device metrics
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Train and evaluate models for each device
for i, device in enumerate(devices):
    accuracy, precision, recall, f1 = train_and_evaluate_device(device['train_data_dir'], device['test_data_dir'], i + 1)

    # Store individual device metrics
    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

# Calculate the average metrics
avg_accuracy = np.mean(accuracy_list)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)

# Print the average metrics
print("Average Metrics Across Devices:")
print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
print(f"Average Precision: {avg_precision * 100:.2f}%")
print(f"Average Recall: {avg_recall * 100:.2f}%")
print(f"Average F1 Score: {avg_f1 * 100:.2f}%")
