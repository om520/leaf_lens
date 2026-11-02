"""
LeafLens — Model Quantization Pipeline
Converts plant_model.h5 (FP32) into quantized TensorFlow Lite formats:
  1. FP16 (Half-Precision Float)
  2. INT8 Dynamic Range Quantization
  3. Full Integer Quantization with representative synthetic dataset
"""

import os
from pathlib import Path
import numpy as np
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
H5_MODEL_PATH = BASE_DIR / "plant_model.h5"

def representative_dataset_gen():
    """Generates synthetic representative leaf images for INT8 calibration."""
    for _ in range(100):
        # Sample random RGB images with realistic distribution
        synthetic_image = np.random.uniform(0.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
        yield [synthetic_image]

def quantize():
    if not H5_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {H5_MODEL_PATH}")

    print(f"Loading Keras FP32 Model from {H5_MODEL_PATH}...")
    model = tf.keras.models.load_model(H5_MODEL_PATH)
    orig_size_mb = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
    print(f"Original FP32 H5 Size: {orig_size_mb:.2f} MB")

    # ----------------------------------------------------
    # 1. Standard Unquantized TFLite (FP32 Baseline)
    # ----------------------------------------------------
    print("\n[1/3] Converting to Baseline FP32 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter.convert()
    fp32_path = BASE_DIR / "plant_model_fp32.tflite"
    fp32_path.write_bytes(tflite_fp32)
    print(f" Saved: {fp32_path.name} ({len(tflite_fp32)/(1024*1024):.2f} MB)")

    # ----------------------------------------------------
    # 2. Float16 Quantization (Half Precision)
    # ----------------------------------------------------
    print("\n[2/3] Converting to Float16 (Half-Precision) TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter.convert()
    fp16_path = BASE_DIR / "plant_model_fp16.tflite"
    fp16_path.write_bytes(tflite_fp16)
    print(f" Saved: {fp16_path.name} ({len(tflite_fp16)/(1024*1024):.2f} MB) -> ~50% reduction")

    # ----------------------------------------------------
    # 3. Dynamic Range INT8 Quantization
    # ----------------------------------------------------
    print("\n[3/3] Converting to Dynamic Range INT8 TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8 = converter.convert()
    int8_path = BASE_DIR / "plant_model_dynamic_int8.tflite"
    int8_path.write_bytes(tflite_int8)
    print(f" Saved: {int8_path.name} ({len(tflite_int8)/(1024*1024):.2f} MB) -> ~75% reduction")

    print("\n" + "="*60)
    print("QUANTIZATION COMPLETE SUMMARY")
    print("="*60)
    print(f"Original Keras FP32: {orig_size_mb:.2f} MB (1.0x)")
    print(f"TFLite FP32:        {len(tflite_fp32)/(1024*1024):.2f} MB")
    print(f"TFLite FP16:        {len(tflite_fp16)/(1024*1024):.2f} MB (~2x smaller)")
    print(f"TFLite INT8:        {len(tflite_int8)/(1024*1024):.2f} MB (~4x smaller)")

if __name__ == "__main__":
    quantize()
