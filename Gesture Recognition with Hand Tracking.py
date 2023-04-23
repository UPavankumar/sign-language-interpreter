import cv2
import numpy as np
import tensorflow as tf
import urllib.request
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
import mediapipe as mp

# Download the BSL dataset
url = 'https://storage.googleapis.com/wandb_datasets/nih/dataset.zip'
urllib.request.urlretrieve(url, 'dataset.zip')

# Extract the dataset
import zipfile
with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
    zip_ref.extractall()

# Load the data and labels from the dataset
data = np.load('dataset/data.npy')
labels = np.load('dataset/labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the gesture recognition model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Loop through video frames
while True:
    ret, frame = cap.read()

    # Preprocess frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False
    results = hands.process(frame)
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Check if hand is detected
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])

        # Convert landmarks to 64x64 grayscale image
        img = np.zeros((480, 640), np.uint8)
        for landmark in landmarks:
            x = int(landmark[0] * 640)
            y = int(landmark[1] * 480)
            img = cv2.circle(img, (x, y), 5, 255, -1)
        img = cv2.resize(img, (64, 64))
        img = np.expand_dims(img, axis=-1)

        # Predict the gesture label
        pred_label = model.predict(np.array([img]))
        pred_label = np.argmax(pred_label)

        # Display the recognized gesture on screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, chr(pred_label + 65), (50, 50), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame on screen
        cv2.imshow('Frame', frame)

    # Exit loop on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break