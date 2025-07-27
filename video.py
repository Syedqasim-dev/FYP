import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("mobilenet_emotion_model.h5")

# Labels (match your model training)
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']

def predict_emotions_from_video(video_path, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    emotion_predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_id % frame_skip != 0:
            continue

        # Convert to grayscale and resize
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (224, 224))  # model input size
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 1))
        face_rgb = np.repeat(face, 3, axis=-1)  # MobileNet expects 3 channels

        # Predict emotion
        prediction = model.predict(face_rgb)
        predicted_label = emotion_labels[np.argmax(prediction)]
        emotion_predictions.append(predicted_label)

    cap.release()

    # --- Calculate Emotion Distribution ---
    emotion_count = Counter(emotion_predictions)
    total = sum(emotion_count.values())
    emotion_distribution = {e: round(c / total, 2) for e, c in emotion_count.items()}
    dominant_emotion = emotion_count.most_common(1)[0][0]

    # --- Updated Empathy Scoring Logic ---
    positive_emotions = ["happy", "surprise"]
    neutral_emotions = ["contempt"]  # treating contempt as neutral
    negative_emotions = ["angry", "sadness", "disgust", "fear"]

    positive = sum([emotion_distribution.get(e, 0) for e in positive_emotions])
    neutral = sum([emotion_distribution.get(e, 0) for e in neutral_emotions])
    negative = sum([emotion_distribution.get(e, 0) for e in negative_emotions])

    facial_empathy_score = round((positive * 10) + (neutral * 5) - (negative * 5), 2)
    facial_empathy_score = max(0, min(10, facial_empathy_score))  # clamp between 0–10

    # --- Add Feedback Message ---
    if facial_empathy_score >= 8:
        feedback = "Highly empathetic interaction."
    elif facial_empathy_score >= 5:
        feedback = "Satisfactory empathy level."
    else:
        feedback = "Empathy appears low. Improvement recommended."

    # --- Final Result Dictionary ---
    result = {
        "session_id": video_path.split("/")[-1].split(".")[0],
        "video_file": video_path,
        "dominant_emotion": dominant_emotion,
        "emotion_distribution": emotion_distribution,
        "facial_empathy_score": facial_empathy_score,
        "notes": feedback
    }

    return result

# ▶️ Example usage:
output = predict_emotions_from_video(r"C:\Users\Qasim's Pc\Desktop\FYP Qasim\datasets\Video_test\WhatsApp Video 2.mp4")
print(output)
