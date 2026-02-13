"""
Enhanced Flask Backend for Shirt Size Prediction System
Includes real-time webcam streaming, database integration, and REST API

FIXED VERSION - Corrected imports for your project structure
"""

from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from pathlib import Path
import sys
import json
import base64
from datetime import datetime
import os

# Get the directory where app.py is located
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent  # Go up one level if needed

# Add paths for imports
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import from current directory first
try:
    from database import DatabaseManager, init_database
except ImportError:
    print("Warning: database.py not found in current directory")
    print("You'll need to copy database.py to the same folder as app.py")
    DatabaseManager = None
    init_database = None

try:
    from body_measurement_cv import BodyMeasurementExtractor
except ImportError:
    print("Error: body_measurement_cv.py not found!")
    print(f"Looking in: {APP_DIR}")
    sys.exit(1)

try:
    from core_ml_pipeline.shirt_size_pipeline import ShirtSizePipeline
except ImportError:
    print("Error: shirt_size_pipeline.py not found!")
    print(f"Looking in: {APP_DIR}")
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='frontend',
            static_folder='frontend/static')
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
if DatabaseManager:
    db = DatabaseManager()
else:
    db = None
    print("WARNING: Running without database support")

extractor = BodyMeasurementExtractor()
pipeline = ShirtSizePipeline()

# Load or train ML model
MODEL_DIR = Path('models')
if MODEL_DIR.exists() and list(MODEL_DIR.iterdir()):
    versions = sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir()])
    if versions:
        pipeline.load_model(versions[-1])
        print(f"✓ Loaded model version: {versions[-1]}")
else:
    print("Training new model...")
    df = pipeline.generate_synthetic_data(n_samples=2000)
    pipeline.train(df)
    pipeline.save_model()
    print("✓ Model trained and saved")

# Global variables for webcam streaming
active_cameras = {}
current_measurements = {}


# ==================== HTML Routes ====================

@app.route('/')
def index():
    """Serve main web interface"""
    return render_template('index.html')


# ==================== API Routes - Person Management ====================

@app.route('/api/person/create', methods=['POST'])
def create_person():
    """Create a new person"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'name' not in data:
            return jsonify({'error': 'Name is required'}), 400
        
        # Check if email already exists
        if data.get('email'):
            existing = db.get_person_by_email(data['email'])
            if existing:
                return jsonify({'error': 'Email already registered'}), 400
        
        person = db.create_person(
            name=data['name'],
            email=data.get('email'),
            phone=data.get('phone'),
            actual_shirt_size=data.get('actual_shirt_size')
        )
        
        return jsonify({
            'success': True,
            'person': person.to_dict()
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/person/<int:person_id>', methods=['GET'])
def get_person(person_id):
    """Get person details"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        person = db.get_person(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        
        return jsonify({
            'success': True,
            'person': person.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/person/all', methods=['GET'])
def get_all_persons():
    """Get all persons"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        limit = request.args.get('limit', 100, type=int)
        persons = db.get_all_persons(limit=limit)
        
        return jsonify({
            'success': True,
            'persons': [p.to_dict() for p in persons],
            'count': len(persons)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/person/<int:person_id>', methods=['PUT'])
def update_person(person_id):
    """Update person information"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        person = db.update_person(person_id, **data)
        
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        
        return jsonify({
            'success': True,
            'person': person.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/person/<int:person_id>', methods=['DELETE'])
def delete_person(person_id):
    """Delete person and all measurements"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        success = db.delete_person(person_id)
        
        if not success:
            return jsonify({'error': 'Person not found'}), 404
        
        return jsonify({
            'success': True,
            'message': 'Person deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== API Routes - Measurements ====================

@app.route('/api/measurement/save', methods=['POST'])
def save_measurement():
    """Save measurement data"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'person_id' not in data or 'measurements' not in data:
            return jsonify({'error': 'person_id and measurements required'}), 400
        
        person_id = data['person_id']
        measurements = data['measurements']
        
        # Verify person exists
        person = db.get_person(person_id)
        if not person:
            return jsonify({'error': 'Person not found'}), 404
        
        # Make prediction
        prediction = pipeline.predict(measurements)
        
        # Save to database
        measurement_record = db.save_measurement(person_id, measurements, prediction)
        
        return jsonify({
            'success': True,
            'measurement': measurement_record.to_dict(),
            'prediction': prediction
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/measurement/history/<int:person_id>', methods=['GET'])
def get_measurement_history(person_id):
    """Get measurement history for a person"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        limit = request.args.get('limit', 50, type=int)
        measurements = db.get_person_measurements(person_id, limit=limit)
        
        return jsonify({
            'success': True,
            'measurements': [m.to_dict() for m in measurements],
            'count': len(measurements)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/measurement/<int:measurement_id>', methods=['GET'])
def get_measurement(measurement_id):
    """Get specific measurement"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        measurement = db.get_measurement(measurement_id)
        if not measurement:
            return jsonify({'error': 'Measurement not found'}), 404
        
        return jsonify({
            'success': True,
            'measurement': measurement.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== API Routes - Prediction ====================

@app.route('/api/predict', methods=['POST'])
def predict_size():
    """Predict shirt size from measurements"""
    try:
        data = request.get_json()
        
        if not data or 'measurements' not in data:
            return jsonify({'error': 'measurements required'}), 400
        
        measurements = data['measurements']
        
        # Validate required fields
        required_fields = ['height_cm', 'weight_kg', 'chest_cm', 
                          'waist_cm', 'shoulder_width_cm', 'arm_length_cm']
        missing = [f for f in required_fields if f not in measurements]
        
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        # Make prediction
        prediction = pipeline.predict(measurements)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== API Routes - Calibration ====================

@app.route('/api/calibrate', methods=['POST'])
def calibrate_camera():
    """Save camera calibration data"""
    if not db:
        # Still allow calibration without database
        try:
            data = request.get_json()
            extractor.pixels_per_cm = data.get('pixels_per_cm')
            if data.get('reference_height_cm'):
                extractor.reference_height_cm = data['reference_height_cm']
            return jsonify({'success': True, 'message': 'Calibration applied (not saved)'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    try:
        data = request.get_json()
        
        required = ['camera_id', 'pixels_per_cm']
        missing = [f for f in required if f not in data]
        
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        calibration = db.save_calibration(
            camera_id=data['camera_id'],
            pixels_per_cm=data['pixels_per_cm'],
            reference_height_cm=data.get('reference_height_cm'),
            method=data.get('method', 'manual')
        )
        
        # Update extractor calibration
        extractor.pixels_per_cm = data['pixels_per_cm']
        if data.get('reference_height_cm'):
            extractor.reference_height_cm = data['reference_height_cm']
        
        return jsonify({
            'success': True,
            'calibration': calibration.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/calibration/<camera_id>', methods=['GET'])
def get_calibration(camera_id):
    """Get calibration for specific camera"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        calibration = db.get_calibration(camera_id)
        
        if not calibration:
            return jsonify({'error': 'Calibration not found'}), 404
        
        return jsonify({
            'success': True,
            'calibration': calibration.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== API Routes - Statistics ====================

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get overall system statistics"""
    if not db:
        return jsonify({
            'success': True,
            'statistics': {
                'total_persons': 0,
                'total_measurements': 0,
                'size_distribution': {},
                'avg_confidence': 0
            }
        }), 200
    
    try:
        stats = db.get_statistics()
        
        return jsonify({
            'success': True,
            'statistics': stats
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/person/<int:person_id>/size-history', methods=['GET'])
def get_size_history(person_id):
    """Get size prediction history for analytics"""
    if not db:
        return jsonify({'error': 'Database not available'}), 500
    
    try:
        history = db.get_person_size_history(person_id)
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==================== Health & Info Routes ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline.model is not None,
        'database_connected': db is not None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    if pipeline.model is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    return jsonify({
        'success': True,
        'model_info': {
            'feature_columns': pipeline.feature_columns_extended,
            'size_classes': pipeline.label_encoder.classes_.tolist(),
            'model_type': type(pipeline.model).__name__,
            'backend': extractor.backend
        }
    }), 200


# ==================== WebSocket Events for Real-time Streaming ====================
# CHANGE THIS LINE:

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    ping_timeout=60,   # Added: waits 60s before timing out
    ping_interval=25   # Added: checks connection every 25s
) 

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('start_camera')
def handle_start_camera(data):
    """Start camera streaming"""
    try:
        camera_source = data.get('camera_source', 0)
        person_id = data.get('person_id')
        
        # Convert string to int for USB cameras
        if isinstance(camera_source, str) and camera_source.isdigit():
            camera_source = int(camera_source)
        
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            emit('camera_error', {'error': 'Could not open camera'})
            return
        
        active_cameras[request.sid] = {
            'cap': cap,
            'person_id': person_id,
            'camera_source': camera_source
        }
        
        emit('camera_started', {'status': 'Camera started successfully'})
        
    except Exception as e:
        emit('camera_error', {'error': str(e)})


@socketio.on('stop_camera')
def handle_stop_camera():
    """Stop camera streaming"""
    try:
        if request.sid in active_cameras:
            active_cameras[request.sid]['cap'].release()
            del active_cameras[request.sid]
        
        emit('camera_stopped', {'status': 'Camera stopped'})
        
    except Exception as e:
        emit('camera_error', {'error': str(e)})


@socketio.on('request_frame')
def handle_request_frame(data):
    """Process and return camera frame with measurements"""
    try:
        if request.sid not in active_cameras:
            emit('frame_error', {'error': 'Camera not started'})
            return
        
        cap = active_cameras[request.sid]['cap']
        ret, frame = cap.read()
        
        if not ret:
            emit('frame_error', {'error': 'Failed to capture frame'})
            return
        
        # Resize for processing
        frame = cv2.resize(frame, (960, 960))
        
        # Get height from data if provided
        estimated_height = data.get('estimated_height_cm')
        
        # Process frame
        annotated_frame, measurements, found = extractor.process_frame(
            frame, 
            estimated_height_cm=estimated_height
        )
        
        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Send frame and measurements
        emit('frame_data', {
            'frame': frame_base64,
            'measurements': measurements if found else None,
            'pose_detected': found,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Store current measurements
        if found:
            current_measurements[request.sid] = measurements
        
    except Exception as e:
        emit('frame_error', {'error': str(e)})


@socketio.on('capture_and_predict')
def handle_capture_and_predict(data):
    """Capture current frame, predict size, and save to database"""
    try:
        if request.sid not in current_measurements:
            emit('prediction_error', {'error': 'No measurements available'})
            return
        
        measurements = current_measurements[request.sid]
        person_id = data.get('person_id')
        
        if not person_id:
            emit('prediction_error', {'error': 'person_id required'})
            return
        
        # Make prediction
        prediction = pipeline.predict(measurements)
        
        # Save to database if available
        if db:
            try:
                measurement_record = db.save_measurement(person_id, measurements, prediction)
                emit('prediction_result', {
                    'success': True,
                    'measurement': measurement_record.to_dict(),
                    'prediction': prediction
                })
            except Exception as db_error:
                print(f"Database error: {db_error}")
                emit('prediction_result', {
                    'success': True,
                    'prediction': prediction,
                    'warning': 'Prediction made but not saved to database'
                })
        else:
            emit('prediction_result', {
                'success': True,
                'prediction': prediction,
                'warning': 'Database not available - prediction not saved'
            })
        
    except Exception as e:
        emit('prediction_error', {'error': str(e)})


# ==================== Main ====================

if __name__ == '__main__':
    # Initialize database if available
    if init_database and db:
        init_database()
    
    # Run server
    print("="*60)
    print("🎽 Shirt Size Prediction Server")
    print("="*60)
    print(f"Backend: {extractor.backend}")
    print(f"Model loaded: {pipeline.model is not None}")
    print(f"Database: {'Connected' if db else 'Not Available'}")
    print("Server running on http://0.0.0.0:5000")
    print("="*60)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)