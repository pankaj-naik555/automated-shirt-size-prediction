"""
Demo: Shirt Size Prediction from Static Images
Test the computer vision pipeline without a webcam
"""
import sys
from pathlib import Path

# Absolute project root path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

print("PROJECT_ROOT =", PROJECT_ROOT)  # DEBUG
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from body_measurement_cv import BodyMeasurementExtractor
from core_ml_pipeline.shirt_size_pipeline import ShirtSizePipeline
import argparse
from pathlib import Path
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'Core ml pipeline'))


def create_test_image():
    """Create a simple test image with a person silhouette"""
    # Create blank image
    img = np.ones((600, 400, 3), dtype=np.uint8) * 255
    
    # Draw a simple person silhouette for testing
    # Head
    cv2.circle(img, (200, 100), 30, (100, 100, 100), -1)
    
    # Body
    cv2.rectangle(img, (170, 130), (230, 300), (100, 100, 100), -1)
    
    # Arms
    cv2.rectangle(img, (130, 150), (170, 250), (100, 100, 100), -1)
    cv2.rectangle(img, (230, 150), (270, 250), (100, 100, 100), -1)
    
    # Legs
    cv2.rectangle(img, (170, 300), (200, 500), (100, 100, 100), -1)
    cv2.rectangle(img, (200, 300), (230, 500), (100, 100, 100), -1)
    
    return img


def demo_from_image(image_path=None, known_height_cm=None):
    """
    Demonstrate shirt size prediction from an image
    
    Parameters:
    -----------
    image_path : str, optional
        Path to image file. If None, uses generated test image
    known_height_cm : float, optional
        Known height for calibration
    """
    print("="*60)
    print("SHIRT SIZE PREDICTION FROM IMAGE DEMO")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    extractor = BodyMeasurementExtractor()
    pipeline = ShirtSizePipeline()
    
    # Load or train model
    model_dir = Path('models')
    if model_dir.exists() and list(model_dir.iterdir()):
        versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
        if versions:
            pipeline.load_model(versions[-1])
            print(f"   ✓ Loaded model: {versions[-1]}")
    else:
        print("   Training new model...")
        df = pipeline.generate_synthetic_data(n_samples=1000)
        pipeline.train(df)
        pipeline.save_model()
        print("   ✓ Model trained")
    
    # Load image
    print("\n2. Loading image...")
    if image_path and Path(image_path).exists():
        image = cv2.imread(image_path)
        print(f"   ✓ Loaded image: {image_path}")
    else:
        if image_path:
            print(f"   ⚠ Image not found: {image_path}")
            print("   Using test image instead")
        image = create_test_image()
        print("   ✓ Created test image")
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Process image
    print("\n3. Detecting body pose...")
    annotated_frame, measurements, found = extractor.process_frame(
        image,
        estimated_height_cm=known_height_cm
    )
    
    if found:
        print("   ✓ Body pose detected!")
        print("\n4. Extracted measurements:")
        print("   " + "-"*50)
        for key, value in measurements.items():
            print(f"   {key:20s}: {value:6.1f}")
        print("   " + "-"*50)
        
        # Make prediction
        print("\n5. Predicting shirt size...")
        result = pipeline.predict(measurements)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\n   Predicted Size: {result['predicted_size']}")
        print(f"   Confidence: {result['confidence']:.1%}")
        
        print("\n   Size Probability Distribution:")
        print("   " + "-"*50)
        for size, prob in sorted(result['all_probabilities'].items()):
            bar = '█' * int(prob * 40)
            print(f"   {size:3s} | {bar:40s} | {prob:6.1%}")
        print("   " + "-"*50)
        
        # Save annotated image
        output_path = 'output_annotated.jpg'
        cv2.imwrite(output_path, annotated_frame)
        print(f"\n   ✓ Saved annotated image to: {output_path}")
        
        # Display image (if display available)
        try:
            cv2.imshow('Pose Detection', annotated_frame)
            print("\n   Press any key to close the image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("   (Display not available - image saved to file)")
        
    else:
        print("   ✗ No body pose detected in image")
        print("\n   Tips:")
        print("   - Ensure person is clearly visible")
        print("   - Person should face the camera")
        print("   - Full body should be in frame")
        print("   - Use good lighting")
        
        # Still show the image
        try:
            cv2.imshow('Input Image', image)
            print("\n   Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass
    
    # Cleanup
    extractor.cleanup()
    print("\n" + "="*60)
    print("DEMO COMPLETED")
    print("="*60)


def demo_batch_processing(image_folder, known_height_cm=None):
    """
    Process multiple images in a folder
    
    Parameters:
    -----------
    image_folder : str
        Path to folder containing images
    known_height_cm : float, optional
        Known height for calibration (applied to all)
    """
    print("="*60)
    print("BATCH IMAGE PROCESSING DEMO")
    print("="*60)
    
    folder = Path(image_folder)
    if not folder.exists():
        print(f"Folder not found: {image_folder}")
        return
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(folder.glob(ext))
        image_files.extend(folder.glob(ext.upper()))
    
    if not image_files:
        print(f"No images found in: {image_folder}")
        return
    
    print(f"\nFound {len(image_files)} images")
    
    # Initialize
    extractor = BodyMeasurementExtractor()
    pipeline = ShirtSizePipeline()
    
    # Load model
    model_dir = Path('models')
    if model_dir.exists() and list(model_dir.iterdir()):
        versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
        if versions:
            pipeline.load_model(versions[-1])
    else:
        df = pipeline.generate_synthetic_data(n_samples=1000)
        pipeline.train(df)
        pipeline.save_model()
    
    # Process each image
    results = []
    
    print("\nProcessing images...")
    print("-" * 60)
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {img_path.name}")
        
        image = cv2.imread(str(img_path))
        annotated_frame, measurements, found = extractor.process_frame(
            image,
            estimated_height_cm=known_height_cm
        )
        
        if found:
            prediction = pipeline.predict(measurements)
            results.append({
                'file': img_path.name,
                'size': prediction['predicted_size'],
                'confidence': prediction['confidence'],
                'measurements': measurements
            })
            print(f"  ✓ Size: {prediction['predicted_size']} (confidence: {prediction['confidence']:.1%})")
            
            # Save annotated image
            output_path = folder / f"annotated_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_frame)
        else:
            print("  ✗ No pose detected")
            results.append({
                'file': img_path.name,
                'size': None,
                'confidence': 0,
                'measurements': None
            })
    
    # Summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r['size'] is not None)
    print(f"\nProcessed: {len(results)} images")
    print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
    
    if successful > 0:
        print("\nResults:")
        print("-" * 60)
        print(f"{'File':<30} {'Size':<8} {'Confidence':<12}")
        print("-" * 60)
        for r in results:
            if r['size']:
                print(f"{r['file']:<30} {r['size']:<8} {r['confidence']:<12.1%}")
            else:
                print(f"{r['file']:<30} {'N/A':<8} {'N/A':<12}")
    
    extractor.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description='Demo: Shirt size prediction from images'
    )
    parser.add_argument(
        '--image', type=str,
        help='Path to image file'
    )
    parser.add_argument(
        '--folder', type=str,
        help='Path to folder containing multiple images'
    )
    parser.add_argument(
        '--height', type=float,
        help='Known height in cm for calibration'
    )
    
    args = parser.parse_args()
    
    if args.folder:
        demo_batch_processing(args.folder, args.height)
    else:
        demo_from_image(args.image, args.height)


if __name__ == "__main__":
    main()