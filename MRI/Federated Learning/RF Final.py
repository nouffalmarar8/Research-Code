import os
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time

# Define the paths to your train and test data folders for all devices
device_data_dirs = [
    ('Device 1', '../train1', '../test1'),
    ('Device 2', '../train2', '../test2'),
    ('Device 3', '../train3', '../test3'),
    ('Device 4', '../train4', '../test4'),
    ('Device 5', '../train5', '../test5')
]

# Initialize lists to store evaluation metrics for all devices
recalls = []
accuracies = []
precisions = []
f1_scores = []

# Initialize an empty list to store trained models
models = []

# Simulate federated learning for all devices
for device_name, train_data_dir, test_data_dir in device_data_dirs:
    print(f"{device_name} Report:")
    start_time = time.time()

    # Initialize empty lists to store image data and labels for training and testing
    X = []
    y = []

    # Function to load and preprocess images
    def load_images_and_labels(data_dir):
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                image = cv2.imread(image_path)
                image = cv2.resize(image, (224, 224))
                X.append(image)
                y.append(class_name)

    # Load training and test data
    load_images_and_labels(train_data_dir)
    load_images_and_labels(test_data_dir)

    # Encode class labels into numerical values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Flatten the image data
    X_flat = np.array(X).reshape(len(X), -1)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y_encoded, test_size=0.2, random_state=42)

    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier using the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_classifier.predict(X_test)

    # Decode the predicted labels from numerical values to class names
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    # Evaluate the model
    print("Classification Report:")
    report = classification_report(label_encoder.inverse_transform(y_test), y_pred_decoded, output_dict=True)
    recalls.append(report['weighted avg']['recall'])
    accuracies.append(accuracy_score(label_encoder.inverse_transform(y_test), y_pred_decoded))
    precisions.append(precision_score(label_encoder.inverse_transform(y_test), y_pred_decoded, average='weighted'))
    f1_scores.append(f1_score(label_encoder.inverse_transform(y_test), y_pred_decoded, average='weighted'))

    # Store the trained model
    models.append(rf_classifier)

    # Print the classification report for each device
    print(classification_report(label_encoder.inverse_transform(y_test), y_pred_decoded))
    print(f"Accuracy: {accuracies[-1] * 100:.2f}%")
    print(f"F1 Score: {f1_scores[-1] * 100:.2f}%")
    print(f"Precision: {precisions[-1] * 100:.2f}%")
    print(f"Recall: {recalls[-1] * 100:.2f}%")

# Calculate the average evaluation metrics for all devices
avg_recall = np.mean(recalls)
avg_accuracy = np.mean(accuracies)
avg_precision = np.mean(precisions)
avg_f1_score = np.mean(f1_scores)

# Print the average evaluation metrics
print("Average Results After Federated Learning:")
print(f"Average Recall: {avg_recall * 100:.2f}%")
print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
print(f"Average Precision: {avg_precision * 100:.2f}%")
print(f"Average F1 Score: {avg_f1_score * 100:.2f}%")
