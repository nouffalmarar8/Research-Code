import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Define the paths to your train and test data folders for each device
device_data = [
    ('C:/Users/pc_al/PycharmProjects/MRI/train1', 'test1'),
    ('train2', 'test2'),
    ('train3', 'test3'),
    ('train4', 'test4'),
    ('train5', 'test5')
]

# Initialize empty lists to store results for each device
results = []

# Simulated global model
global_model = None

# Define the number of communication rounds (iterations)
num_rounds = 10

# Initialize label encoder
label_encoder = LabelEncoder()

def load_data(train_data_dir, test_data_dir):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # Load training data
    train_images = [file for file in os.listdir(train_data_dir) if file.lower().endswith('.jpg')]
    print(f"Training data in {train_data_dir}: {train_images}")

    for image_file in train_images:
        image_path = os.path.join(train_data_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            # Preprocess and resize the image if needed
            # You may need to perform additional preprocessing depending on your data
            X_train.append(image)
            label = image_file.split("_")[0]  # Assuming labels are part of the file names
            y_train.append(label)

    # Load testing data
    test_images = [file for file in os.listdir(test_data_dir) if file.lower().endswith('.jpg')]
    print(f"Testing data in {test_data_dir}: {test_images}")

    for image_file in test_images:
        image_path = os.path.join(test_data_dir, image_file)
        image = cv2.imread(image_path)
        if image is not None:
            # Preprocess and resize the image if needed
            # You may need to perform additional preprocessing depending on your data
            X_test.append(image)
            label = image_file.split("_")[0]  # Assuming labels are part of the file names
            y_test.append(label)

    # Convert lists to NumPy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test

for round in range(num_rounds):
    print(f"Round {round + 1} of Federated Learning")

    # Initialize the global model at the start of each round
    if global_model is None:
        global_model = RandomForestClassifier(n_estimators=100, random_state=42)

    for train_data_dir, test_data_dir in device_data:
        # Load data for each device
        X_train, y_train, X_test, y_test = load_data(train_data_dir, test_data_dir)

        if X_train.shape[0] > 0 and X_test.shape[0] > 0:
            # Flatten and normalize image data if needed
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            # Train and evaluate the local model
            local_model = RandomForestClassifier(n_estimators=100, random_state=42)
            local_model.fit(X_train, y_train)
            y_pred = local_model.predict(X_test)

            # Print results for the local model
            classification_rep = classification_report(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            print(f"Classification Report for {train_data_dir} (Local Model):")
            print(classification_rep)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")

            # Update the global model using the local model (in a real federated learning system, this step would involve model aggregation)
            global_model = local_model
        else:
            print(f"No data available for {train_data_dir}.")

    # Evaluate the global model after all devices have trained their local models
    for train_data_dir, test_data_dir in device_data:
        X_train, y_train, X_test, y_test = load_data(train_data_dir, test_data_dir)

        if X_test.shape[0] > 0:
            # Flatten and normalize image data if needed
            X_test = X_test.reshape(X_test.shape[0], -1)

            y_pred = global_model.predict(X_test)

            # Print results for the global model
            classification_rep = classification_report(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')

            print(f"Classification Report for {train_data_dir} (Global Model):")
            print(classification_rep)
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"F1 Score: {f1 * 100:.2f}%")
            print(f"Precision: {precision * 100:.2f}%")
            print(f"Recall: {recall * 100:.2f}%")

            results.append((accuracy, f1, precision, recall))
        else:
            print(f"No data available for {train_data_dir}.")

# Calculate the average metrics for all devices after the simulation
if results:
    avg_accuracy = sum([result[0] for result in results]) / len(results)
    avg_f1 = sum([result[1] for result in results]) / len(results)
    avg_precision = sum([result[2] for result in results]) / len(results)
    avg_recall = sum([result[3] for result in results]) / len(results)

    print("Average Metrics for All Devices (Global Model):")
    print(f"Average Accuracy: {avg_accuracy * 100:.2f}%")
    print(f"Average F1 Score: {avg_f1 * 100:.2f}%")
    print(f"Average Precision: {avg_precision * 100:.2f}%")
    print(f"Average Recall: {avg_recall * 100:.2f}%")
else:
    print("No results available. Please check your data and file paths.")
