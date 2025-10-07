import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# -- configuration --
cascade_path = r"C:/Users/kp/AppData/Local/Programs/Python/Python310/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
model_path = "best_emotion_model.h5"   # path to your trained model
emotion_labels = ['neutral','happy','sad','surprise','fear','disgust','anger']  # adjust to your classes

# -- load detectors & model --
face_cap = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)
video_cap = cv2.VideoCapture(0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
history = deque(maxlen=8)   # smooth over last N frames

while True:
    ret, frame = video_cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48,48))

    for (x,y,w,h) in faces:
        # add padding for better context
        pad = int(0.2 * w)
        x1 = max(0, x-pad); y1 = max(0, y-pad)
        x2 = min(frame.shape[1], x+w+pad); y2 = min(frame.shape[0], y+h+pad)

        face_roi = gray[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_roi = clahe.apply(face_roi)               # contrast normalize
        face_roi = cv2.resize(face_roi, (48,48))
        face_roi = face_roi.astype('float32') / 255.0
        face_roi = np.expand_dims(face_roi, -1)        # (48,48,1)
        face_roi = np.expand_dims(face_roi, 0)         # (1,48,48,1)

        preds = model.predict(face_roi)[0]             # vector of length num_classes
        history.append(preds)
        avg_preds = np.mean(history, axis=0)
        idx = int(np.argmax(avg_preds))
        label = emotion_labels[idx]
        score = float(avg_preds[idx])

        # draw results
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        text = f"{label} {score:.2f}"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
