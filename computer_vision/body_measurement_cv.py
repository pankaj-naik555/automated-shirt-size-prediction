"""
Computer Vision Module for Body Measurement Extraction
FIXED VERSION - Proper skeleton visualization and pose detection
Updated to use MoveNet SinglePose Lightning TFLite model.
"""

import cv2
import numpy as np
import tensorflow.lite as tflite
from pathlib import Path
import json

class BodyMeasurementExtractor:
    """Extract body measurements using MoveNet TFLite Pose Estimation"""
    
    def __init__(self, model_path=None):
        # Backend identifier for UI
        self.backend = "MoveNet (TFLite)" 
        
        # Use a raw string to fix the '\P' SyntaxWarning
        if model_path is None:
            model_path = r"C:\PROJECT_2025\machine_learning2025\Automated shirt size predication\computer_vision\MoveNet-Python-Example-main\tflite\lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite"

        # Initialize TFLite Interpreter
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_size = 192  # MoveNet Lightning standard input size

        # Calibration state
        self.pixels_per_cm = None
        self.reference_height_cm = None
        self.calibration_method = 'default'
        
        # MoveNet keypoint names for reference
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
     
    def calibrate_with_reference_object(self, frame, object_type='credit_card'):
        """
        Calibrate using standard objects
        
        Reference sizes:
        - Credit card: 8.56 cm × 5.398 cm
        - A4 paper: 21.0 cm × 29.7 cm
        - Standard door: 203.2 cm height
        """
        reference_sizes = {
            'credit_card': 8.56,  # width in cm
            'a4_paper': 21.0,     # width in cm
            'door': 203.2         # height in cm
        }
        
        # Detect object using contour detection or ArUco markers
        detected_pixels = self.detect_reference_object(frame, object_type)
        
        if detected_pixels:
            self.pixels_per_cm = detected_pixels / reference_sizes[object_type]
            self.calibration_method = f'reference_{object_type}'
            return True
        return False
    
    def detect_reference_object(self, frame, object_type):
        """Detect reference object in frame"""
        if object_type == 'credit_card':
            # Use ArUco marker on credit card for precision
            # Or use template matching
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ... ArUco detection code ...
            return detected_pixels

    def process_frame(self, frame, estimated_height_cm=None):
        """Processes frame to find landmarks and calculate measurements"""
        h, w = frame.shape[:2]
        
        # 1. Preprocess for MoveNet
        input_img = cv2.resize(frame, (self.input_size, self.input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.reshape(1, self.input_size, self.input_size, 3).astype(np.uint8)

        # 2. Run Inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_img)
        self.interpreter.invoke()
        
        # Get Keypoints: [1, 1, 17, 3] -> [y, x, score]
        keypoints = np.squeeze(self.interpreter.get_tensor(self.output_details[0]['index']))

        # 3. Check if pose is detected (require key points with good confidence)
        # Check nose, shoulders, and at least one ankle
        pose_detected = (
            keypoints[0][2] > 0.3 and  # nose
            keypoints[5][2] > 0.3 and  # left shoulder
            keypoints[6][2] > 0.3 and  # right shoulder
            (keypoints[15][2] > 0.3 or keypoints[16][2] > 0.3)  # at least one ankle
        )
        
        if not pose_detected:
            # No reliable pose detected - return original frame
            # Add "NO POSE DETECTED" text
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, 'NO POSE DETECTED', 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(annotated_frame, 'Stand in full view of camera', 
                       (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated_frame, None, False

        # 4. Calibration Logic
        # MoveNet indices: 0:nose, 15:L_ankle, 16:R_ankle
        nose_y = keypoints[0][0] * h
        
        # Use average of both ankles if available
        if keypoints[15][2] > 0.3 and keypoints[16][2] > 0.3:
            avg_ankle_y = ((keypoints[15][0] + keypoints[16][0]) / 2) * h
        elif keypoints[15][2] > 0.3:
            avg_ankle_y = keypoints[15][0] * h
        else:
            avg_ankle_y = keypoints[16][0] * h
            
        pixel_height = abs(avg_ankle_y - (nose_y - (0.05 * h)))  # Adjust for head top
        
        # Update calibration if height is provided
        if estimated_height_cm:
            self.pixels_per_cm = pixel_height / estimated_height_cm
            self.reference_height_cm = estimated_height_cm
            self.calibration_method = 'manual_height'
        elif not self.pixels_per_cm:
            # Default fallback calibration
            self.pixels_per_cm = pixel_height / 170.0
            self.calibration_method = 'default'

        # 5. Extract Measurements in CM
        measurements = self.calculate_shirt_measurements(keypoints, h, w)
        
        # 6. Draw Skeleton for feedback
        annotated_frame = self.draw_skeleton(frame.copy(), keypoints, h, w)
        
        # 7. Add measurement info on frame
        annotated_frame = self.add_measurement_overlay(annotated_frame, measurements, keypoints)

        return annotated_frame, measurements, True

    def calculate_shirt_measurements(self, kp, h, w):
        """Calculates specific metrics required by ShirtSizePipeline"""
        def get_dist(p1_idx, p2_idx):
            """Calculate distance between two keypoints in centimeters"""
            y1, x1 = kp[p1_idx][0] * h, kp[p1_idx][1] * w
            y2, x2 = kp[p2_idx][0] * h, kp[p2_idx][1] * w
            pixel_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            return pixel_dist / self.pixels_per_cm

        # Calculate basic measurements
        shoulder_width = get_dist(5, 6)
        hip_width = get_dist(11, 12)
        
        
        # Calculate height from keypoints
        nose_y = kp[0][0] * h
        if kp[15][2] > 0.3 and kp[16][2] > 0.3:
            avg_ankle_y = ((kp[15][0] + kp[16][0]) / 2) * h
        elif kp[15][2] > 0.3:
            avg_ankle_y = kp[15][0] * h
        else:
            avg_ankle_y = kp[16][0] * h
            
        pixel_height = abs(avg_ankle_y - (nose_y - (0.08 * h)))
        estimated_height = pixel_height / self.pixels_per_cm
        
        # Use reference height if available
        final_height = self.reference_height_cm if self.reference_height_cm else estimated_height
        
        # Improved circumference estimation
        chest_circumference = shoulder_width * 3.2
        waist_circumference = hip_width * 3.6
        
        # Calculate arm length
        arm_length = get_dist(5, 7) + get_dist(7, 9)
        
        # Weight estimation
        estimated_bmi = 22 + (shoulder_width - 45) * 0.3
        estimated_weight = estimated_bmi * ((final_height / 100) ** 2)
        
        measurements_cm = {
            'height_cm': final_height,
            'shoulder_width_cm': shoulder_width,
            'chest_cm': chest_circumference,
            'waist_cm': waist_circumference,
            'arm_length_cm': arm_length,
            'weight_kg': estimated_weight,
            'pose_quality': self._calculate_pose_quality(kp),
            'calibration_method': self.calibration_method
        }

        return measurements_cm

    def _calculate_pose_quality(self, kp):
        """Calculate pose quality score based on keypoint confidence"""
        key_indices = [0, 5, 6, 7, 9, 11, 12, 15, 16]
        confidences = [kp[i][2] for i in key_indices if i < len(kp)]
        
        if not confidences:
            return 0.0
        
        return float(np.mean(confidences))

    def draw_skeleton(self, frame, kp, h=None, w=None):
        """Draw skeleton overlay on frame with MoveNet visualization"""
        if h is None or w is None:
            h, w = frame.shape[:2]
        
        # Define skeleton connections
        connections = [
            # Head
            (0, 1), (0, 2),  # nose to eyes
            (1, 3), (2, 4),  # eyes to ears
            # Torso
            (5, 6),   # shoulders
            (5, 11),  # left shoulder to hip
            (6, 12),  # right shoulder to hip
            (11, 12), # hips
            # Arms
            (5, 7),   # left shoulder to elbow
            (7, 9),   # left elbow to wrist
            (6, 8),   # right shoulder to elbow
            (8, 10),  # right elbow to wrist
            # Legs
            (11, 13), # left hip to knee
            (13, 15), # left knee to ankle
            (12, 14), # right hip to knee
            (14, 16)  # right knee to ankle
        ]
        
        # Draw connections (skeleton lines)
        for p1, p2 in connections:
            if kp[p1][2] > 0.3 and kp[p2][2] > 0.3:
                pt1 = (int(kp[p1][1] * w), int(kp[p1][0] * h))
                pt2 = (int(kp[p2][1] * w), int(kp[p2][0] * h))
                
                # Draw line with gradient effect (thicker = more confident)
                confidence = (kp[p1][2] + kp[p2][2]) / 2
                thickness = int(2 + confidence * 2)
                cv2.line(frame, pt1, pt2, (0, 255, 0), thickness)
        
        # Draw keypoints (circles)
        for i in range(17):
            if kp[i][2] > 0.3:
                center = (int(kp[i][1] * w), int(kp[i][0] * h))
                confidence = kp[i][2]
                
                # Color based on confidence (green = high, yellow = medium, red = low)
                if confidence > 0.7:
                    color = (0, 255, 0)  # Green
                elif confidence > 0.5:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 165, 255)  # Orange
                
                # Draw filled circle for keypoint
                cv2.circle(frame, center, 5, color, -1)
                # Draw outline
                cv2.circle(frame, center, 5, (255, 255, 255), 1)
        
        return frame

    def add_measurement_overlay(self, frame, measurements, keypoints):
        """Add measurement text overlay on frame"""
        h, w = frame.shape[:2]
        # Add semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Add "POSE DETECTED" indicator
        cv2.putText(frame, 'POSE DETECTED', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add key measurements
        y_offset = 60
        line_height = 20
        
        text_lines = [
            f"Height: {measurements['height_cm']:.1f} cm",
            f"Chest: {measurements['chest_cm']:.1f} cm",
            f"Waist: {measurements['waist_cm']:.1f} cm",
            f"Shoulder: {measurements['shoulder_width_cm']:.1f} cm",
            f"Quality: {measurements['pose_quality']:.2f}"
        ]
        
        for i, text in enumerate(text_lines):
            cv2.putText(frame, text, (20, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add calibration indicator
        calib_text = f"Cal: {self.calibration_method}"
        cv2.putText(frame, calib_text, (20, y_offset + len(text_lines) * line_height + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        return frame

    def cleanup(self):
        """Cleanup resources"""
        pass  # TFLite interpreter handles its own memory


def save_calibration(extractor, filepath="calibration.json"):
    """Save calibration data to file"""
    data = {
        "pixels_per_cm": extractor.pixels_per_cm,
        "reference_height_cm": extractor.reference_height_cm,
        "calibration_method": extractor.calibration_method
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {filepath}")


def load_calibration(extractor, filepath="calibration.json"):
    """Load calibration data from file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            extractor.pixels_per_cm = data.get("pixels_per_cm")
            extractor.reference_height_cm = data.get("reference_height_cm")
            extractor.calibration_method = data.get("calibration_method", "default")
        print(f"Calibration loaded from {filepath}")
        return True
    except FileNotFoundError:
        print(f"Calibration file not found: {filepath}")
        return False
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return False