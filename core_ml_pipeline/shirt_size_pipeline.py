"""
Automated Shirt Size Prediction Pipeline
COMPLETE & FIXED VERSION
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from datetime import datetime
from pathlib import Path

class ShirtSizePipeline:
    """End-to-end pipeline for shirt size prediction"""
    
    def __init__(self, model_path='models'):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
        # Base features expected from input
        self.feature_columns = [
            'height_cm', 'weight_kg', 'chest_cm', 
            'waist_cm', 'shoulder_width_cm', 'arm_length_cm'
        ]
        
    # =========================================================================
    # 1. Synthetic Data Generation (The missing part)
    # =========================================================================
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data for shirt size prediction"""
        print("Generating synthetic data...")
        np.random.seed(42)
        data = []
        
        size_specs = {
            'XS': {'height': (150, 165), 'weight': (45, 55), 'chest': (80, 88), 
                   'waist': (65, 75), 'shoulder': (38, 42), 'arm': (58, 62)},
            'S': {'height': (160, 170), 'weight': (55, 65), 'chest': (88, 96),
                  'waist': (75, 85), 'shoulder': (42, 46), 'arm': (62, 66)},
            'M': {'height': (168, 178), 'weight': (65, 75), 'chest': (96, 104),
                  'waist': (85, 95), 'shoulder': (46, 50), 'arm': (66, 70)},
            'L': {'height': (175, 185), 'weight': (75, 85), 'chest': (104, 112),
                  'waist': (95, 105), 'shoulder': (50, 54), 'arm': (70, 74)},
            'XL': {'height': (180, 190), 'weight': (85, 95), 'chest': (112, 120),
                   'waist': (105, 115), 'shoulder': (54, 58), 'arm': (74, 78)},
            'XXL': {'height': (185, 200), 'weight': (95, 110), 'chest': (120, 130),
                    'waist': (115, 130), 'shoulder': (58, 62), 'arm': (78, 82)}
        }
        
        samples_per_size = n_samples // len(size_specs)
        
        for size, specs in size_specs.items():
            for _ in range(samples_per_size):
                data.append({
                    'height_cm': np.random.uniform(*specs['height']) + np.random.normal(0, 1),
                    'weight_kg': np.random.uniform(*specs['weight']) + np.random.normal(0, 2),
                    'chest_cm': np.random.uniform(*specs['chest']) + np.random.normal(0, 2),
                    'waist_cm': np.random.uniform(*specs['waist']) + np.random.normal(0, 2),
                    'shoulder_width_cm': np.random.uniform(*specs['shoulder']) + np.random.normal(0, 1),
                    'arm_length_cm': np.random.uniform(*specs['arm']) + np.random.normal(0, 1),
                    'size': size
                })
        
        return pd.DataFrame(data)

    # =========================================================================
    # 2. Robust Preprocessing (Fixes "Unseen Features" error)
    # =========================================================================
    def preprocess_data(self, df):
        """Clean data and add features, ignoring extra metadata"""
        # FIX: Only keep known columns to ignore 'pose_quality', 'calibration', etc.
        df_processed = df[self.feature_columns].copy()
        
        # Feature Engineering
        df_processed['bmi'] = df_processed['weight_kg'] / (df_processed['height_cm'] / 100) ** 2
        df_processed['chest_to_waist'] = df_processed['chest_cm'] / df_processed['waist_cm']
        df_processed['build_index'] = (df_processed['chest_cm'] * df_processed['shoulder_width_cm']) / df_processed['height_cm']
        
        # Simplified BMI category to avoid One-Hot Encoding mismatch
        df_processed['is_overweight'] = (df_processed['bmi'] > 25).astype(int)
        
        # Store sorted column list for consistency
        self.feature_columns_extended = sorted(df_processed.columns.tolist())
        return df_processed[self.feature_columns_extended]

    # =========================================================================
    # 3. Data Preparation (Fixes "prepare_data missing" error)
    # =========================================================================
    def prepare_data(self, df):
        """Prepares data for training"""
        X = self.preprocess_data(df)
        y = self.label_encoder.fit_transform(df['size'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit scaler ONLY on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    # =========================================================================
    # 4. Fast Training (Fixes slow training time)
    # =========================================================================
    def train(self, df, fast_mode=True):
        """Train the model (Fast Mode enabled by default)"""
        print("Preparing data...")
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        if fast_mode:
            print("Running FAST training (skipping grid search)...")
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15, 
                random_state=42, 
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
        else:
            print("Running FULL GridSearch (may take time)...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'max_features': ['sqrt']
            }
            grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            self.model = grid.best_estimator_
            print(f"Best Params: {grid.best_params_}")
            
        accuracy = self.model.score(X_test, y_test)
        print(f"Training complete. Accuracy: {accuracy:.2%}")
        return self.model

    # =========================================================================
    # 5. Prediction Logic
    # =========================================================================
    def predict(self, measurements):
        """Predict using the trained model"""
        if self.model is None:
            raise ValueError("Model not loaded! Please train or load a model first.")
            
        # Handle dict or list input
        df = pd.DataFrame([measurements] if isinstance(measurements, dict) else measurements)
        
        # Preprocess and Scale
        X = self.preprocess_data(df)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        preds = self.model.predict(X_scaled)
        probs = self.model.predict_proba(X_scaled)
        
        # Format result
        size_label = self.label_encoder.inverse_transform(preds)[0]
        confidence = float(np.max(probs))
        
        return {
            'predicted_size': size_label,
            'confidence': confidence,
            'probabilities': {
                cls: float(prob) 
                for cls, prob in zip(self.label_encoder.classes_, probs[0])
            }
        }

    # =========================================================================
    # 6. Save/Load Logic
    # =========================================================================
    def save_model(self, version=None):
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_path = self.model_path / version
        save_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path / 'model.joblib')
        joblib.dump(self.scaler, save_path / 'scaler.joblib')
        joblib.dump(self.label_encoder, save_path / 'label_encoder.joblib')
        
        # Save metadata including column names
        metadata = {
            'feature_columns': self.feature_columns,
            'extended_columns': getattr(self, 'feature_columns_extended', [])
        }
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        print(f"Model saved to {save_path}")
        return version

    def load_model(self, version):
        load_path = self.model_path / version
        if not load_path.exists():
            raise FileNotFoundError(f"Model {version} not found")
            
        self.model = joblib.load(load_path / 'model.joblib')
        self.scaler = joblib.load(load_path / 'scaler.joblib')
        self.label_encoder = joblib.load(load_path / 'label_encoder.joblib')
        
        # Restore column names
        meta_path = load_path / 'metadata.json'
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                data = json.load(f)
                self.feature_columns_extended = data.get('extended_columns', [])
        
        print(f"Model {version} loaded.")