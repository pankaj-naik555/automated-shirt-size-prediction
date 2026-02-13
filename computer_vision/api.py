"""
REST API for Shirt Size Prediction Service

Endpoints:
- POST /predict: Predict shirt size from measurements
- GET /health: Health check
- GET /model/info: Get model information
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from core_ml_pipeline.shirt_size_pipeline import ShirtSizePipeline
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Initialize pipeline
pipeline = ShirtSizePipeline()

# Load or train model on startup
MODEL_DIR = Path('models')
if MODEL_DIR.exists() and list(MODEL_DIR.iterdir()):
    # Load latest model
    versions = sorted([d.name for d in MODEL_DIR.iterdir() if d.is_dir()])
    if versions:
        latest_version = versions[-1]
        pipeline.load_model(latest_version)
        print(f"Loaded model version: {latest_version}")
else:
    # Train new model
    print("No existing model found. Training new model...")
    df = pipeline.generate_synthetic_data(n_samples=2000)
    pipeline.train(df)
    pipeline.save_model()
    print("Model trained and saved")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline.model is not None
    }), 200


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if pipeline.model is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    return jsonify({
        'feature_columns': pipeline.feature_columns_extended,
        'size_classes': pipeline.label_encoder.classes_.tolist(),
        'model_type': type(pipeline.model).__name__
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict shirt size from measurements
    
    Request body:
    {
        "height_cm": 175,
        "weight_kg": 70,
        "chest_cm": 100,
        "waist_cm": 90,
        "shoulder_width_cm": 48,
        "arm_length_cm": 68
    }
    
    Or for batch prediction:
    {
        "measurements": [
            {"height_cm": 175, ...},
            {"height_cm": 180, ...}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if it's a batch request
        if 'measurements' in data:
            measurements = data['measurements']
            if not isinstance(measurements, list):
                return jsonify({'error': 'measurements must be a list'}), 400
        else:
            measurements = data
        
        # Validate required fields
        required_fields = [
            'height_cm', 'weight_kg', 'chest_cm', 
            'waist_cm', 'shoulder_width_cm', 'arm_length_cm'
        ]
        
        if isinstance(measurements, list):
            for i, m in enumerate(measurements):
                missing = [f for f in required_fields if f not in m]
                if missing:
                    return jsonify({
                        'error': f'Missing required fields in measurement {i}: {missing}'
                    }), 400
        else:
            missing = [f for f in required_fields if f not in measurements]
            if missing:
                return jsonify({
                    'error': f'Missing required fields: {missing}'
                }), 400
        
        # Make prediction
        result = pipeline.predict(measurements)
        
        return jsonify({
            'success': True,
            'predictions': result if isinstance(result, list) else [result]
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint (alternative format)
    
    Request body:
    [
        {"height_cm": 175, ...},
        {"height_cm": 180, ...}
    ]
    """
    try:
        measurements = request.get_json()
        
        if not isinstance(measurements, list):
            return jsonify({'error': 'Request body must be a list of measurements'}), 400
        
        result = pipeline.predict(measurements)
        
        return jsonify({
            'success': True,
            'predictions': result,
            'count': len(result)
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)