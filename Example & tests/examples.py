"""
Simple example usage of the Shirt Size Prediction Pipeline

This script shows the most common use cases:
1. Quick prediction with a trained model
2. Training a new model
3. Making multiple predictions
"""

from shirt_size_pipeline import ShirtSizePipeline


def quick_prediction_example():
    """Example: Quick prediction with existing model"""
    print("="*60)
    print("EXAMPLE 1: Quick Prediction")
    print("="*60)
    
    # Initialize and train a model
    pipeline = ShirtSizePipeline()
    df = pipeline.generate_synthetic_data(n_samples=1000)
    pipeline.train(df)
    
    # Make a prediction
    my_measurements = {
        'height_cm': 175,
        'weight_kg': 70,
        'chest_cm': 100,
        'waist_cm': 90,
        'shoulder_width_cm': 48,
        'arm_length_cm': 68
    }
    
    result = pipeline.predict(my_measurements)
    
    print(f"\nYour measurements:")
    for key, value in my_measurements.items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommended shirt size: {result['predicted_size']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    print("\nProbability for each size:")
    for size in ['XS', 'S', 'M', 'L', 'XL', 'XXL']:
        prob = result['all_probabilities'][size]
        bar = '█' * int(prob * 50)
        print(f"  {size:3s}: {bar} {prob:.1%}")


def training_custom_data_example():
    """Example: Training with your own data"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Training with Custom Data")
    print("="*60)
    
    import pandas as pd
    
    # Create sample custom data
    custom_data = pd.DataFrame([
        {'height_cm': 160, 'weight_kg': 55, 'chest_cm': 86, 'waist_cm': 72, 
         'shoulder_width_cm': 40, 'arm_length_cm': 60, 'size': 'S'},
        {'height_cm': 175, 'weight_kg': 75, 'chest_cm': 102, 'waist_cm': 92, 
         'shoulder_width_cm': 49, 'arm_length_cm': 69, 'size': 'L'},
        {'height_cm': 188, 'weight_kg': 95, 'chest_cm': 118, 'waist_cm': 108, 
         'shoulder_width_cm': 57, 'arm_length_cm': 77, 'size': 'XL'},
    ])
    
    print("\nCustom training data:")
    print(custom_data[['height_cm', 'weight_kg', 'size']])
    
    # In real use, you'd have more data!
    # For demo, we'll use synthetic data with a few custom samples
    pipeline = ShirtSizePipeline()
    synthetic_data = pipeline.generate_synthetic_data(n_samples=500)
    combined_data = pd.concat([synthetic_data, custom_data], ignore_index=True)
    
    print(f"\nTraining on {len(combined_data)} samples...")
    pipeline.train(combined_data)
    
    # Save for later use
    model_path = pipeline.save_model(version='my_model_v1')
    print(f"Model saved to: {model_path}")


def batch_prediction_example():
    """Example: Predicting for multiple people at once"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Predictions")
    print("="*60)
    
    # Initialize and train
    pipeline = ShirtSizePipeline()
    df = pipeline.generate_synthetic_data(n_samples=1000)
    pipeline.train(df, model_type='random_forest')
    
    # Batch of measurements
    team_measurements = [
        {
            'name': 'Alice',
            'height_cm': 165,
            'weight_kg': 58,
            'chest_cm': 90,
            'waist_cm': 78,
            'shoulder_width_cm': 42,
            'arm_length_cm': 63
        },
        {
            'name': 'Bob',
            'height_cm': 180,
            'weight_kg': 82,
            'chest_cm': 106,
            'waist_cm': 98,
            'shoulder_width_cm': 51,
            'arm_length_cm': 71
        },
        {
            'name': 'Charlie',
            'height_cm': 172,
            'weight_kg': 68,
            'chest_cm': 98,
            'waist_cm': 88,
            'shoulder_width_cm': 47,
            'arm_length_cm': 67
        }
    ]
    
    # Extract just measurements for prediction
    measurements_only = [
        {k: v for k, v in person.items() if k != 'name'}
        for person in team_measurements
    ]
    
    results = pipeline.predict(measurements_only)
    
    print("\nShirt size recommendations:")
    print("-" * 60)
    for person, result in zip(team_measurements, results):
        print(f"{person['name']:10s} | Size: {result['predicted_size']:3s} | "
              f"Confidence: {result['confidence']:.0%} | "
              f"Height: {person['height_cm']}cm, Weight: {person['weight_kg']}kg")


def size_chart_example():
    """Example: Creating a size recommendation chart"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Size Chart Generation")
    print("="*60)
    
    pipeline = ShirtSizePipeline()
    df = pipeline.generate_synthetic_data(n_samples=1000)
    pipeline.train(df)
    
    print("\nSize recommendations by height and weight:")
    print("(Assuming average proportions)")
    print()
    print("Weight →")
    print("Height ↓  ", end="")
    
    weights = [55, 65, 75, 85, 95, 105]
    heights = [160, 165, 170, 175, 180, 185, 190]
    
    # Print header
    for w in weights:
        print(f"{w}kg ", end="")
    print()
    
    # Print size chart
    for height in heights:
        print(f"{height}cm:    ", end="")
        for weight in weights:
            # Estimate other measurements based on height and weight
            chest = 80 + (height - 160) * 0.5 + (weight - 55) * 0.7
            waist = 70 + (height - 160) * 0.4 + (weight - 55) * 0.8
            shoulder = 38 + (height - 160) * 0.12
            arm = 58 + (height - 160) * 0.12
            
            measurements = {
                'height_cm': height,
                'weight_kg': weight,
                'chest_cm': chest,
                'waist_cm': waist,
                'shoulder_width_cm': shoulder,
                'arm_length_cm': arm
            }
            
            result = pipeline.predict(measurements)
            print(f"{result['predicted_size']:3s}  ", end="")
        print()


if __name__ == "__main__":
    # Run all examples
    quick_prediction_example()
    training_custom_data_example()
    batch_prediction_example()
    size_chart_example()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)