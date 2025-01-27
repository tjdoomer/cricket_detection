import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_cricket_detector(dataset_path='cricket_dataset', video_path='cricket_1.MP4'):
    try:
        X = []
        y = []
        
        # Initialize HOG
        winSize = (64,64)
        blockSize = (16,16)
        blockStride = (8,8)
        cellSize = (8,8)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        
        print("Loading cricket images...")
        # Load positive samples (crickets)
        for img_name in os.listdir(dataset_path):
            if img_name.endswith('.jpg'):
                img_path = os.path.join(dataset_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    features = hog.compute(gray)
                    if features is not None:
                        X.append(features.flatten())
                        y.append(1)
        
        positive_count = len(X)
        print(f"Loaded {positive_count} positive samples")
        
        # Generate negative samples
        print(f"Generating negative samples from {video_path}...")
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        target_negatives = min(20, positive_count)
        negative_count = 0
        
        while negative_count < target_negatives:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video, resetting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            h, w = frame.shape[:2]
            x = np.random.randint(0, w-64)
            y_coord = np.random.randint(0, h-64)
            patch = frame[y_coord:y_coord+64, x:x+64]
            
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            features = hog.compute(gray_patch)
            
            if features is not None:
                X.append(features.flatten())
                y.append(0)
                negative_count += 1
                print(f"Generated negative sample {negative_count}/{target_negatives}")
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Cleaning up...")
    finally:
        if 'cap' in locals():
            cap.release()
        
        if len(X) > 0 and len(y) > 0:
            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)
            
            print(f"\nFinal dataset: {len(X)} total samples ({sum(y)} positive, {len(y)-sum(y)} negative)")
            
            # Train SVM classifier
            print("Training detector...")
            classifier = SVC(kernel='linear', probability=True)
            classifier.fit(X, y)
            
            return classifier, hog
        else:
            print("Not enough samples collected to train the classifier.")
            return None, None

def process_video(video_path, classifier, hog):
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        scaled_frame = cv2.resize(frame, (640, 480))
        
        for y in range(0, scaled_frame.shape[0]-64, 32):
            for x in range(0, scaled_frame.shape[1]-64, 32):
                window = scaled_frame[y:y+64, x:x+64]
                gray_window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
                features = hog.compute(gray_window)
                
                if features is not None:
                    score = classifier.predict_proba([features.flatten()])[0][1]
                    if score > 0.8:
                        cv2.rectangle(scaled_frame, (x, y), (x+64, y+64), (0, 255, 0), 2)
        
        cv2.imshow('Cricket Detection', scaled_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

# Main execution
if __name__ == "__main__":
    try:
        print("Starting cricket detector training...")
        classifier, hog = train_cricket_detector()
        
        if classifier is not None:
            # Process videos
            videos = ['cricket_1.MP4', 'cricket_2.MP4']
            for video in videos:
                print(f"Processing {video}...")
                process_video(video, classifier, hog)
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        cv2.destroyAllWindows()