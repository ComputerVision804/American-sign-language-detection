import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ASLDetector:
    def __init__(self):
        self.model = load_model('model/asl_model.h5')
        with open('model/labels.txt', 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

    def process_frame(self, frame):
        img_size = 200  # Match your training size
        roi = cv2.resize(frame, (img_size, img_size))
        roi = np.expand_dims(roi, axis=0) / 255.0
        prediction = self.model.predict(roi, verbose=0)
        class_idx = np.argmax(prediction)
        label = self.labels[class_idx]
        cv2.putText(frame, f"Prediction: {label}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = ASLDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        processed_frame = detector.process_frame(frame)
        cv2.imshow('ASL Detection', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
