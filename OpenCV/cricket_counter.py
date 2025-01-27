import cv2
import numpy as np
import os
import pickle
from sklearn.svm import SVC
import time

def load_model(model_path='cricket_model.pkl'):
    """
    Loads and validates the pre-trained cricket detection model from a pickle file.
    The model consists of an SVM classifier and HOG (Histogram of Oriented Gradients) parameters.
    """
    try:
        # Basic file validation
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found!")
            return None, None
            
        if os.path.getsize(model_path) == 0:
            print(f"Error: Model file {model_path} is empty!")
            return None, None
            
        # Load the pickled model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
            # Validate model structure
            if not isinstance(model_data, dict):
                print("Error: Invalid model file format!")
                return None, None
                
            if 'classifier' not in model_data or 'hog_params' not in model_data:
                print("Error: Model file missing required components!")
                return None, None
            
            # Reconstruct HOG descriptor from saved parameters
            params = model_data['hog_params']
            hog = cv2.HOGDescriptor(
                params['winSize'],      # Size of detection window
                params['blockSize'],    # Block size for normalization
                params['blockStride'],  # Overlap between blocks
                params['cellSize'],     # Size of cells for histogram computation
                params['nbins']         # Number of orientation bins
            )
            
            return model_data['classifier'], hog
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def count_crickets(video_path, classifier, hog, confidence_threshold=0.8):
    """
    Process video to detect and count crickets using computer vision and machine learning.
    Uses multi-scale detection, motion tracking, and non-maximum suppression for accurate counting.
    
    Args:
        video_path: Path to input video file
        classifier: Trained SVM classifier for cricket detection
        hog: HOG descriptor for feature extraction
        confidence_threshold: Minimum confidence score for detection (default: 0.8)
    """
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize tracking variables
    frame_counts = []          # Store cricket counts per frame
    processed_frames = 0       # Counter for processed frames
    start_time = time.time()   # For FPS calculation
    prev_frame = None         # Previous frame for motion detection
    prev_detections = []      # Previous detections for tracking
    
    # Motion detection parameters
    motion_threshold = 25  # Minimum pixel difference for motion
    min_movement = 5      # Minimum pixels moved to be considered cricket movement
    max_movement = 100    # Maximum pixels moved to be considered same cricket
    
    # Setup display window
    cv2.namedWindow('Cricket Counter', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cricket Counter', width, height)
    
    # Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frames += 1
        current_detections = []
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Motion detection using frame differencing
        if prev_frame is not None:
            frame_diff = cv2.absdiff(frame_gray, prev_frame)
            motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
            motion_mask = cv2.dilate(motion_mask, None, iterations=2)
        
        # Multi-scale detection for different cricket sizes
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        base_window_size = 64  # Base detection window size
        
        # Sliding window detection at multiple scales
        for scale in scales:
            window_size = int(base_window_size * scale)
            stride = int(window_size // 2)
            
            # Slide window across frame
            for y in range(0, height-window_size, stride):
                for x in range(0, width-window_size, stride):
                    # Extract and preprocess window
                    window = frame[y:y+window_size, x:x+window_size]
                    window_resized = cv2.resize(window, (64, 64))
                    gray_window = cv2.cvtColor(window_resized, cv2.COLOR_BGR2GRAY)
                    features = hog.compute(gray_window)
                    
                    if features is not None:
                        # Get classification confidence
                        score = classifier.predict_proba([features.flatten()])[0][1]
                        
                        # Incorporate motion detection
                        motion_score = 0
                        if prev_frame is not None:
                            region_motion = motion_mask[y:y+window_size, x:x+window_size]
                            motion_score = np.mean(region_motion) / 255.0
                        
                        # Combine classifier and motion scores
                        combined_score = score * (1 + motion_score)
                        
                        # Store high-confidence detections
                        if combined_score > confidence_threshold:
                            current_detections.append({
                                'x': x,
                                'y': y,
                                'width': window_size,
                                'height': window_size,
                                'score': combined_score,
                                'motion_score': motion_score
                            })
        
        # Track crickets between frames
        tracked_detections = []
        if prev_detections:
            for curr_det in current_detections:
                curr_center = (curr_det['x'] + curr_det['width']//2, 
                             curr_det['y'] + curr_det['height']//2)
                
                # Match with previous detections
                for prev_det in prev_detections:
                    prev_center = (prev_det['x'] + prev_det['width']//2, 
                                 prev_det['y'] + prev_det['height']//2)
                    
                    # Calculate movement distance
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                     (curr_center[1] - prev_center[1])**2)
                    
                    # Boost score for consistent movement
                    if min_movement <= distance <= max_movement:
                        curr_det['score'] *= 1.2
                        tracked_detections.append(curr_det)
                        break
        
        # Apply non-maximum suppression to remove overlapping detections
        final_detections = []
        current_detections.sort(key=lambda x: x['score'], reverse=True)
        
        while current_detections:
            best = current_detections[0]
            final_detections.append(best)
            # Remove overlapping detections
            current_detections = [
                d for d in current_detections[1:]
                if not (abs(d['x'] - best['x']) < best['width']//2 and
                       abs(d['y'] - best['y']) < best['height']//2)
            ]
        
        # Visualize detections
        for det in final_detections:
            # Color based on motion score (more green = more motion)
            color = (0, int(255 * min(det.get('motion_score', 0), 1)), 0)
            cv2.rectangle(frame, 
                        (det['x'], det['y']), 
                        (det['x'] + det['width'], det['y'] + det['height']), 
                        color, 
                        2)
            # Add confidence scores to visualization
            label = f"{det['score']:.2f}"
            if 'motion_score' in det:
                label += f" (m:{det['motion_score']:.2f})"
            cv2.putText(frame, 
                      label, 
                      (det['x'], det['y'] - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      color, 
                      2)
        
        # Update statistics
        frame_counts.append(len(final_detections))
        avg_count = np.mean(frame_counts)
        
        # Calculate density (assumes 1m x 1m viewing area)
        area = 1.0  # Adjust based on actual viewing area
        density = avg_count / area
        
        # Add statistics overlay
        stats_text = [
            f"Frame: {processed_frames}/{total_frames}",
            f"Current Count: {len(final_detections)}",
            f"Average Count: {avg_count:.1f}",
            f"Estimated Density: {density:.1f} crickets/m²"
        ]
        
        # Draw statistics on frame
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, 
                      text, 
                      (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7, 
                      (0, 255, 0), 
                      2)
            y_offset += 30
        
        # Print progress every 30 frames
        if processed_frames % 30 == 0:
            elapsed_time = time.time() - start_time
            fps = processed_frames / elapsed_time
            print(f"Processed {processed_frames}/{total_frames} frames ({fps:.1f} fps)")
        
        # Update tracking variables
        prev_frame = frame_gray
        prev_detections = final_detections
        
        # Display output
        cv2.imshow('Cricket Counter', frame)
        
        # Check for quit command
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final statistics
    if frame_counts:
        print("\nFinal Statistics:")
        print(f"Total Frames Processed: {processed_frames}")
        print(f"Average Cricket Count: {np.mean(frame_counts):.1f}")
        print(f"Maximum Count in Single Frame: {np.max(frame_counts)}")
        print(f"Minimum Count in Single Frame: {np.min(frame_counts)}")
        print(f"Estimated Population Density: {np.mean(frame_counts)/area:.1f} crickets/m²")
        
        return frame_counts
    return []

if __name__ == "__main__":
    # Main execution block
    classifier, hog = load_model()
    
    if classifier is not None and hog is not None:
        video_path = 'cricket_1.MP4'  # Input video path
        print(f"Processing video: {video_path}")
        
        # Process video and get frame-by-frame counts
        frame_counts = count_crickets(video_path, classifier, hog)
        
        # Optional: Generate plot of cricket counts over time
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(frame_counts)
            plt.title('Cricket Count Over Time')
            plt.xlabel('Frame Number')
            plt.ylabel('Number of Crickets')
            plt.grid(True)
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping plot generation.")