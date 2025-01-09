class Camera:
    def __init__(self, use_picamera=False):
        self.use_picamera = use_picamera
        if use_picamera:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration())
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(0)
    
    def get_frame(self):
        if self.use_picamera:
            frame = self.camera.capture_array()
            return True, frame
        else:
            return self.camera.read()
    
    def release(self):
        if not self.use_picamera:
            self.camera.release()
        else:
            self.camera.stop()