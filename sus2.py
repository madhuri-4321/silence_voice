import cv2
import mediapipe as mp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Step 1: Load or train gesture classifier
def train_model():
    # Dummy training data (replace with actual gesture data)
    X = [  # landmark positions (flattened)
        np.random.rand(42),  # sample for 'hello'
        np.random.rand(42)   # sample for 'thank you'
    ]
    y = ['hello', 'thank you']

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X, y)

    with open('gesture_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def load_model():
    with open('gesture_model.pkl', 'rb') as f:
        return pickle.load(f)

# Step 2: Initialize MediaPipe and load model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

gesture_model = load_model()

# Step 3: Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])

            if len(landmarks) == 42:  # 21 points * (x, y)
                prediction = gesture_model.predict([landmarks])[0]
                cv2.putText(frame, f'{prediction}', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Sign to Text', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()