"""
Database Module for Shirt Size Prediction System
Uses SQLAlchemy ORM for database operations
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()


class Person(Base):
    """Model for storing person information"""
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    phone = Column(String(20))
    actual_shirt_size = Column(String(10))  # User's known size (optional)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    measurements = relationship("Measurement", back_populates="person", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'actual_shirt_size': self.actual_shirt_size,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'measurement_count': len(self.measurements)
        }


class Measurement(Base):
    """Model for storing body measurements"""
    __tablename__ = 'measurements'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey('persons.id'), nullable=False)
    
    # Body measurements in cm
    height_cm = Column(Float)
    weight_kg = Column(Float)
    chest_cm = Column(Float)
    waist_cm = Column(Float)
    shoulder_width_cm = Column(Float)
    arm_length_cm = Column(Float)
    
    # Derived measurements
    bmi = Column(Float)
    chest_to_waist_ratio = Column(Float)
    
    # Prediction results
    predicted_size = Column(String(10))
    confidence = Column(Float)
    all_probabilities = Column(String(500))  # JSON string
    
    # Quality metrics
    pose_detection_quality = Column(Float)  # 0-1 score
    calibration_method = Column(String(50))  # 'manual', 'auto', 'reference_object'
    
    # Metadata
    measurement_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(String(500))
    
    # Relationship
    person = relationship("Person", back_populates="measurements")
    
    def to_dict(self):
        return {
            'id': self.id,
            'person_id': self.person_id,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'chest_cm': self.chest_cm,
            'waist_cm': self.waist_cm,
            'shoulder_width_cm': self.shoulder_width_cm,
            'arm_length_cm': self.arm_length_cm,
            'bmi': self.bmi,
            'predicted_size': self.predicted_size,
            'confidence': self.confidence,
            'all_probabilities': json.loads(self.all_probabilities) if self.all_probabilities else {},
            'measurement_date': self.measurement_date.isoformat() if self.measurement_date else None,
            'pose_quality': self.pose_detection_quality,
            'notes': self.notes
        }


class CalibrationData(Base):
    """Model for storing camera calibration data"""
    __tablename__ = 'calibration_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(50), unique=True)  # e.g., 'usb_0', 'ip_192.168.1.100'
    pixels_per_cm = Column(Float)
    reference_height_cm = Column(Float)
    reference_method = Column(String(50))  # 'manual', 'credit_card', 'a4_paper'
    calibration_date = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Integer, default=1)  # 1 = active, 0 = inactive
    
    def to_dict(self):
        return {
            'id': self.id,
            'camera_id': self.camera_id,
            'pixels_per_cm': self.pixels_per_cm,
            'reference_height_cm': self.reference_height_cm,
            'reference_method': self.reference_method,
            'calibration_date': self.calibration_date.isoformat() if self.calibration_date else None,
            'is_active': bool(self.is_active)
        }


class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, database_uri='sqlite:///database/measurements.db'):
        self.engine = create_engine(database_uri, echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    # ==================== Person Operations ====================
    
    def create_person(self, name, email=None, phone=None, actual_shirt_size=None):
        """Create a new person record"""
        person = Person(
            name=name,
            email=email,
            phone=phone,
            actual_shirt_size=actual_shirt_size
        )
        self.session.add(person)
        self.session.commit()
        return person
    
    def get_person(self, person_id):
        """Get person by ID"""
        return self.session.query(Person).filter_by(id=person_id).first()
    
    def get_person_by_email(self, email):
        """Get person by email"""
        return self.session.query(Person).filter_by(email=email).first()
    
    def get_all_persons(self, limit=100):
        """Get all persons"""
        return self.session.query(Person).order_by(Person.created_at.desc()).limit(limit).all()
    
    def update_person(self, person_id, **kwargs):
        """Update person information"""
        person = self.get_person(person_id)
        if person:
            for key, value in kwargs.items():
                if hasattr(person, key):
                    setattr(person, key, value)
            person.updated_at = datetime.utcnow()
            self.session.commit()
        return person
    
    def delete_person(self, person_id):
        """Delete person and all measurements"""
        person = self.get_person(person_id)
        if person:
            self.session.delete(person)
            self.session.commit()
            return True
        return False
    
    # ==================== Measurement Operations ====================
    
    def save_measurement(self, person_id, measurements, prediction_result):
        """Save measurement data and prediction"""
        
        # Calculate derived metrics
        bmi = measurements.get('weight_kg', 0) / ((measurements.get('height_cm', 170) / 100) ** 2)
        chest_to_waist = measurements.get('chest_cm', 0) / measurements.get('waist_cm', 1)
        
        measurement = Measurement(
            person_id=person_id,
            height_cm=measurements.get('height_cm'),
            weight_kg=measurements.get('weight_kg'),
            chest_cm=measurements.get('chest_cm'),
            waist_cm=measurements.get('waist_cm'),
            shoulder_width_cm=measurements.get('shoulder_width_cm'),
            arm_length_cm=measurements.get('arm_length_cm'),
            bmi=bmi,
            chest_to_waist_ratio=chest_to_waist,
            predicted_size=prediction_result.get('predicted_size'),
            confidence=prediction_result.get('confidence'),
            all_probabilities=json.dumps(prediction_result.get('all_probabilities', {})),
            pose_detection_quality=measurements.get('pose_quality', 0.0),
            calibration_method=measurements.get('calibration_method', 'auto')
        )
        
        self.session.add(measurement)
        self.session.commit()
        return measurement
    
    def get_measurement(self, measurement_id):
        """Get measurement by ID"""
        return self.session.query(Measurement).filter_by(id=measurement_id).first()
    
    def get_person_measurements(self, person_id, limit=50):
        """Get all measurements for a person"""
        return self.session.query(Measurement)\
            .filter_by(person_id=person_id)\
            .order_by(Measurement.measurement_date.desc())\
            .limit(limit)\
            .all()
    
    def get_latest_measurement(self, person_id):
        """Get the most recent measurement for a person"""
        return self.session.query(Measurement)\
            .filter_by(person_id=person_id)\
            .order_by(Measurement.measurement_date.desc())\
            .first()
    
    def delete_measurement(self, measurement_id):
        """Delete a measurement"""
        measurement = self.get_measurement(measurement_id)
        if measurement:
            self.session.delete(measurement)
            self.session.commit()
            return True
        return False
    
    # ==================== Calibration Operations ====================
    
    def save_calibration(self, camera_id, pixels_per_cm, reference_height_cm=None, method='manual'):
        """Save or update calibration data"""
        calibration = self.session.query(CalibrationData).filter_by(camera_id=camera_id).first()
        
        if calibration:
            # Update existing
            calibration.pixels_per_cm = pixels_per_cm
            calibration.reference_height_cm = reference_height_cm
            calibration.reference_method = method
            calibration.calibration_date = datetime.utcnow()
        else:
            # Create new
            calibration = CalibrationData(
                camera_id=camera_id,
                pixels_per_cm=pixels_per_cm,
                reference_height_cm=reference_height_cm,
                reference_method=method
            )
            self.session.add(calibration)
        
        self.session.commit()
        return calibration
    
    def get_calibration(self, camera_id):
        """Get calibration data for a camera"""
        return self.session.query(CalibrationData)\
            .filter_by(camera_id=camera_id, is_active=1)\
            .first()
    
    def get_all_calibrations(self):
        """Get all active calibrations"""
        return self.session.query(CalibrationData)\
            .filter_by(is_active=1)\
            .order_by(CalibrationData.calibration_date.desc())\
            .all()
    
    # ==================== Analytics ====================
    
    def get_person_size_history(self, person_id):
        """Get size prediction history for analytics"""
        measurements = self.get_person_measurements(person_id)
        
        history = []
        for m in measurements:
            history.append({
                'date': m.measurement_date.isoformat() if m.measurement_date else None,
                'predicted_size': m.predicted_size,
                'confidence': m.confidence,
                'measurements': {
                    'height': m.height_cm,
                    'weight': m.weight_kg,
                    'chest': m.chest_cm,
                    'waist': m.waist_cm
                }
            })
        
        return history
    
    def get_statistics(self):
        """Get overall system statistics"""
        total_persons = self.session.query(Person).count()
        total_measurements = self.session.query(Measurement).count()
        
        # Size distribution
        from sqlalchemy import func
        size_distribution = self.session.query(
            Measurement.predicted_size,
            func.count(Measurement.id)
        ).group_by(Measurement.predicted_size).all()
        
        return {
            'total_persons': total_persons,
            'total_measurements': total_measurements,
            'size_distribution': dict(size_distribution),
            'avg_confidence': self.session.query(func.avg(Measurement.confidence)).scalar() or 0
        }
    
    def close(self):
        """Close database connection"""
        self.session.close()


# Utility functions
def init_database(database_uri='sqlite:///database/measurements.db'):
    """Initialize database with tables"""
    import os
    os.makedirs('database', exist_ok=True)
    
    engine = create_engine(database_uri, echo=True)
    Base.metadata.create_all(engine)
    print(f"Database initialized at {database_uri}")


if __name__ == "__main__":
    # Test database creation
    print("Initializing database...")
    init_database()
    
    # Test operations
    db = DatabaseManager()
    
    # Create test person
    person = db.create_person(
        name="John Doe",
        email="john@example.com",
        actual_shirt_size="M"
    )
    print(f"Created person: {person.to_dict()}")
    
    # Save test measurement
    test_measurements = {
        'height_cm': 175,
        'weight_kg': 70,
        'chest_cm': 100,
        'waist_cm': 90,
        'shoulder_width_cm': 48,
        'arm_length_cm': 68
    }
    
    test_prediction = {
        'predicted_size': 'M',
        'confidence': 0.92,
        'all_probabilities': {'XS': 0.01, 'S': 0.05, 'M': 0.92, 'L': 0.02}
    }
    
    measurement = db.save_measurement(person.id, test_measurements, test_prediction)
    print(f"Saved measurement: {measurement.to_dict()}")
    
    # Get statistics
    stats = db.get_statistics()
    print(f"Statistics: {stats}")
    
    db.close()
    print("Database test completed!")