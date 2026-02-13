"""
Test script for the shirt size prediction pipeline
Demonstrates functionality and validates predictions
"""

import pandas as pd
from shirt_size_pipeline import ShirtSizePipeline


def test_pipeline():
    """Test the complete pipeline"""
    print("="*60)
    print("SHIRT SIZE PREDICTION PIPELINE TEST")
    print("="*60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = ShirtSizePipeline()
    print("✓ Pipeline initialized")
    
    # Generate data
    print("\n2. Generating synthetic training data...")
    df = pipeline.generate_synthetic_data(n_samples=1000)
    print(f"✓ Generated {len(df)} samples")
    print(f"\nSize distribution:")
    print(df['size'].value_counts().sort_index())
    
    # Train model
    print("\n3. Training model...")
    results = pipeline.train(df, model_type='random_forest')
    print(f"✓ Model trained with accuracy: {results['accuracy']:.4f}")
    
    # Save model
    print("\n4. Saving model...")
    model_dir = pipeline.save_model()
    print(f"✓ Model saved to: {model_dir}")
    
    # Test predictions
    print("\n5. Testing predictions...")
    print("="*60)
    
    test_cases = [
        {
            'name': 'Extra Small',
            'measurements': {
                'height_cm': 160,
                'weight_kg': 50,
                'chest_cm': 84,
                'waist_cm': 70,
                'shoulder_width_cm': 40,
                'arm_length_cm': 60
            },
            'expected': 'XS'
        },
        {
            'name': 'Small',
            'measurements': {
                'height_cm': 165,
                'weight_kg': 60,
                'chest_cm': 92,
                'waist_cm': 80,
                'shoulder_width_cm': 44,
                'arm_length_cm': 64
            },
            'expected': 'S'
        },
        {
            'name': 'Medium',
            'measurements': {
                'height_cm': 173,
                'weight_kg': 70,
                'chest_cm': 100,
                'waist_cm': 90,
                'shoulder_width_cm': 48,
                'arm_length_cm': 68
            },
            'expected': 'M'
        },
        {
            'name': 'Large',
            'measurements': {
                'height_cm': 180,
                'weight_kg': 80,
                'chest_cm': 108,
                'waist_cm': 100,
                'shoulder_width_cm': 52,
                'arm_length_cm': 72
            },
            'expected': 'L'
        },
        {
            'name': 'Extra Large',
            'measurements': {
                'height_cm': 185,
                'weight_kg': 90,
                'chest_cm': 116,
                'waist_cm': 110,
                'shoulder_width_cm': 56,
                'arm_length_cm': 76
            },
            'expected': 'XL'
        },
        {
            'name': 'Double Extra Large',
            'measurements': {
                'height_cm': 190,
                'weight_kg': 105,
                'chest_cm': 125,
                'waist_cm': 125,
                'shoulder_width_cm': 60,
                'arm_length_cm': 80
            },
            'expected': 'XXL'
        }
    ]
    
    correct = 0
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Measurements:")
        for key, value in test_case['measurements'].items():
            print(f"  {key}: {value}")
        
        result = pipeline.predict(test_case['measurements'])
        predicted = result['predicted_size']
        confidence = result['confidence']
        
        is_correct = predicted == test_case['expected']
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Expected: {test_case['expected']}, Predicted: {predicted} (confidence: {confidence:.2%})")
        
        # Show top 3 predictions
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        print("Top 3 predictions:")
        for size, prob in sorted_probs[:3]:
            print(f"  {size}: {prob:.2%}")
    
    accuracy = correct / len(test_cases)
    print("\n" + "="*60)
    print(f"Test Results: {correct}/{len(test_cases)} correct ({accuracy:.1%})")
    print("="*60)
    
    # Test batch prediction
    print("\n6. Testing batch prediction...")
    batch_measurements = [tc['measurements'] for tc in test_cases[:3]]
    batch_results = pipeline.predict(batch_measurements)
    print(f"✓ Batch prediction successful: {len(batch_results)} predictions made")
    for i, result in enumerate(batch_results):
        print(f"  {i+1}. {result['predicted_size']} (confidence: {result['confidence']:.2%})")
    
    # Test edge cases
    print("\n7. Testing edge cases...")
    
    edge_cases = [
        {
            'name': 'Between M and L',
            'measurements': {
                'height_cm': 177,
                'weight_kg': 75,
                'chest_cm': 102,
                'waist_cm': 92,
                'shoulder_width_cm': 49,
                'arm_length_cm': 69
            }
        },
        {
            'name': 'Very tall but light (athletic)',
            'measurements': {
                'height_cm': 188,
                'weight_kg': 78,
                'chest_cm': 100,
                'waist_cm': 85,
                'shoulder_width_cm': 50,
                'arm_length_cm': 74
            }
        },
        {
            'name': 'Shorter but heavier (muscular)',
            'measurements': {
                'height_cm': 170,
                'weight_kg': 85,
                'chest_cm': 110,
                'waist_cm': 95,
                'shoulder_width_cm': 52,
                'arm_length_cm': 67
            }
        }
    ]
    
    for edge_case in edge_cases:
        print(f"\n{edge_case['name']}:")
        result = pipeline.predict(edge_case['measurements'])
        print(f"  Predicted: {result['predicted_size']} (confidence: {result['confidence']:.2%})")
        
        # Show uncertainty
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        top_2_diff = sorted_probs[0][1] - sorted_probs[1][1]
        if top_2_diff < 0.2:
            print(f"  ⚠ Low confidence - close call between {sorted_probs[0][0]} and {sorted_probs[1][0]}")
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*60)


if __name__ == "__main__":
    test_pipeline()