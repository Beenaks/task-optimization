import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import tensorflow as tf
from keras.models import load_model
import threading
import time
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import load_model

# Load pre-trained emotion detection model
face_emotion_model = load_model('emotion_model.hdf5') # <- Make sure you have this model ready!

# Labels for the emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize face detector
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize emotion counter
emotion_counter = Counter()

# Text Sentiment Analysis (Simple)
def analyze_text_sentiment(text):
    text = text.lower()
    if any(word in text for word in ["happy", "great", "good", "awesome", "excited"]):
        return "Positive"
    elif any(word in text for word in ["sad", "bad", "upset", "angry", "terrible"]):
        return "Negative"
    else:
        return "Neutral"

# Audio Capture Thread
def capture_audio_emotion():
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source)
        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                sentiment = analyze_text_sentiment(text)
                print(f"Detected Speech Sentiment: {sentiment} | Text: {text}")
                break  # <--- STOP after first detection
            except sr.WaitTimeoutError:
                print("No speech detected, trying again...")
            except Exception as e:
                print(f"Speech recognition error: {e}")



# Live Plot Update
def update_plot():
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    while True:
        ax.clear()
        emotions = list(emotion_counter.keys())
        counts = list(emotion_counter.values())
        ax.bar(emotions, counts, color='skyblue')
        ax.set_xlabel('Emotions')
        ax.set_ylabel('Counts')
        ax.set_title('Real-Time Detected Emotions')
        plt.pause(1)

# Video Capture and Face Emotion Detection
def capture_video_emotion():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop face
                face = frame[y:y+h, x:x+w]
                if face.size != 0:
                    face = cv2.resize(face, (64, 64))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = face / 255.0
                    face = np.reshape(face, (1, 64, 64, 1))

                    # Predict emotion
                    emotion_prediction = face_emotion_model.predict(face)
                    max_index = int(np.argmax(emotion_prediction))
                    predicted_emotion = emotion_labels[max_index]

                    # Update counter
                    emotion_counter[predicted_emotion] += 1

                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, predicted_emotion, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting program...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Start Threads
if __name__ == "__main__":
    audio_thread = threading.Thread(target=capture_audio_emotion)
    video_thread = threading.Thread(target=capture_video_emotion)
    plot_thread = threading.Thread(target=update_plot)

    audio_thread.start()
    video_thread.start()
    plot_thread.start()

    audio_thread.join()
    video_thread.join()
    plot_thread.join()
