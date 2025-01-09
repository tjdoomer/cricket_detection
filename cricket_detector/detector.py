import cv2
import numpy as np
from sklearn.svm import SVC
import pickle
import os
from datetime import datetime
from sqlalchemy.orm import Session

class CricketDetector:
    def __init__(self, model_path='models/cricket_model.pkl', db_session=None):
        # Load the trained model and HOG descriptor
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.hog = cv2.HOGDescriptor()
        self.db_session = db_session
        self.current_count = 0
        self.confidence_threshold = 0.8
        
    def process_frame(self, frame):
        height, width = frame.shape[:2]
        detections = []
        
        # Process frame in sliding windows
        for y in range(0, height-64, 32):
            for x in range(0, width-64, 32):
                window = frame[y:y+64, x:x+64]
                if window.shape[:2] == (64, 64):
                    gray_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                    features = self.hog.compute(gray_window)
                    
                    if features is not None:
                        score = self.classifier.predict_proba([features.flatten()])[0][1]
                        if score > self.confidence_threshold:
                            detections.append({
                                'x': x,
                                'y': y,
                                'confidence': score
                            })
        
        # Update count and log to database
        self.current_count = len(detections)
        if self.db_session:
            self._log_count()
        
        # Draw detections
        for det in detections:
            x, y = det['x'], det['y']
            cv2.rectangle(frame, (x, y), (x+64, y+64), (0, 255, 0), 2)
            label = f"Cricket: {det['confidence']:.2f}"
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, self.current_count
    
    def _log_count(self):
        from models import CricketCount
        
        count_record = CricketCount(
            count=self.current_count,
            confidence=self.confidence_threshold
        )
        self.db_session.add(count_record)
        self.db_session.commit()