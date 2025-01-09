from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import threading
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, CricketCount
from detector import CricketDetector
from camera import Camera

app = Flask(__name__)
socketio = SocketIO(app)

# Database setup
engine = create_engine('sqlite:///cricket_data.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Initialize detector and camera
detector = CricketDetector(db_session=Session())
camera = Camera(use_picamera=False)  # Set to True for Raspberry Pi

def process_camera():
    while True:
        ret, frame = camera.get_frame()
        if ret:
            processed_frame, count = detector.process_frame(frame)
            # Encode frame for web streaming
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            # Emit frame and count via WebSocket
            socketio.emit('frame_update', {
                'frame': frame_bytes,
                'count': count
            })
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/counts')
def get_counts():
    session = Session()
    counts = session.query(CricketCount).order_by(CricketCount.timestamp.desc()).limit(100).all()
    session.close()
    return jsonify([count.to_dict() for count in counts])

if __name__ == '__main__':
    # Start camera processing in background thread
    camera_thread = threading.Thread(target=process_camera)
    camera_thread.daemon = True
    camera_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
