import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def train_cricket_detector(dataset_path='cricket_dataset'):
    # Load and prepare training data
    X = []  # images
    y = []  # labels
    
    # Load cricket images (positive samples)
    for img_name in os.listdir(dataset_path):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(dataset_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize to consistent size
                img = cv2.resize(img, (64, 64))
                X.append(img)
                y.append(1)  # 1 for cricket
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train a simple classifier (HOG + SVM)
    hog = cv2.HOGDescriptor()
    return hog

def process_video(video_path, detector):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for faster processing
        scaled_frame = cv2.resize(frame, (640, 480))
        
        # Apply detector
        # (Here you would implement your detection logic)
        # For now, let's just show the frame with some basic processing
        gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        # Show both original and processed frames
        cv2.imshow('Original', scaled_frame)
        cv2.imshow('Processing', edges)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Train the detector
detector = train_cricket_detector()

# Process each video
videos = ['cricket_1.MP4', 'cricket_2.MP4']
for video in videos:
    print(f"Processing {video}...")
    process_video(video, detector)
