import sys
from pathlib import Path
import cv2
import numpy as np
import argparse
import time
import sklearn


# Absolute project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from body_measurement_cv import BodyMeasurementExtractor, save_calibration, load_calibration
from core_ml_pipeline.shirt_size_pipeline import ShirtSizePipeline

class WebcamShirtSizePredictor:
    def __init__(self, model_version=None, backend='mediapipe'):
        print(f"Loading {backend} backend...")
        # NEW CODE
        self.extractor = BodyMeasurementExtractor()
        self.pipeline = ShirtSizePipeline()
        
        # Initialize ML model
        model_dir = Path('models')
        if model_dir.exists() and list(model_dir.iterdir()):
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if versions:
                self.pipeline.load_model(model_version or versions[-1])
        else:
            print("Training initial model...")
            df = self.pipeline.generate_synthetic_data(n_samples=2000)
            self.pipeline.train(df)
            self.pipeline.save_model()
            
        load_calibration(self.extractor)
        
        # UI State
        self.current_measurements = None
        self.current_prediction = None
        self.user_height_cm = None
        self.show_instructions = True
        self.measurement_history = []

    def run(self, camera_source):
        # Handle camera source (int for USB, str for Mobile URL)
        if str(camera_source).isdigit():
            camera_source = int(camera_source)
            print(f"Connecting to USB Camera {camera_source}...")
        else:
            print(f"Connecting to Mobile Camera at: {camera_source}")
            
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            print("If using mobile, ensure the URL ends in /video or similar MJPEG stream.")
            return

        # ... (Rest of the Run loop is identical to original, omitting for brevity) ...
        # ... Insert the exact same while loop from your original file here ...
        
        print("Camera started. Press 'Q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame (Mobile camera might have disconnected)")
                time.sleep(1) # Prevent tight loop on error
                continue

            # Standard processing
            # Resize for speed if using high-res mobile cam
            frame = cv2.resize(frame, (1024, 768)) 
            
            # Flip if it's a selfie camera/webcam, usually don't flip back-facing mobile cam
            if isinstance(camera_source, int):
                frame = cv2.flip(frame, 1)

            annotated_frame, measurements, found = self.extractor.process_frame(frame, self.user_height_cm)
            
            if found and measurements:
                self.current_measurements = measurements
                self.measurement_history.append(measurements)
                if len(self.measurement_history) > 10: self.measurement_history.pop(0)

            # Draw UI (Simplified call to existing method)
            display_frame = self.draw_ui(annotated_frame)
            cv2.imshow('Shirt Size Predictor', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('h'):
                try:
                    self.user_height_cm = float(input("\nEnter height (cm): "))
                except: pass
            elif key == ord('c'):
                if self.current_measurements:
                    pred = self.pipeline.predict(self.current_measurements)
                    self.current_prediction = pred
                    print(f"Predicted: {pred['predicted_size']}")

        cap.release()
        cv2.destroyAllWindows()

    def draw_ui(self, frame):
        # ... (Copy the draw_ui method from your original file) ...
        # (It depends on self.current_prediction which we set above)
        cv2.putText(frame, f"Backend: {self.extractor.backend}", (10, frame.shape[0]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        return frame

def main():
    parser = argparse.ArgumentParser()
    # Change type to str so it accepts both "0" and "http://..."
    parser.add_argument("--camera", type=str, default="0", help="Camera index or IP URL")
    args = parser.parse_args()

    # Convert to int if it's a single digit (0, 1, 2), otherwise keep as string URL
    camera_source = args.camera
    if camera_source.isdigit() and len(camera_source) == 1:
        camera_source = int(camera_source)

    predictor = WebcamShirtSizePredictor()
    predictor.run(camera_source)

if __name__ == "__main__":
    main()