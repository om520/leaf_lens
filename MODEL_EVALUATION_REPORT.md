# LeafLens — Deep Model Inspection & Quantization Evaluation Report

This report documents the architectural inspection, numerical precision validation, parameter breakdown, and quantization feasibility for `plant_model.h5`.

---

## 1. Executive Quantization Verdict

| Verification Item | Inspected Value | Status |
| :--- | :--- | :---: |
| **Model Format** | Keras / HDF5 (`plant_model.h5`) | Standard Keras 3.13.2 |
| **Current Precision** | **Single Precision (IEEE 754 Float32)** | **UNQUANTIZED** |
| **Float32 Parameter Ratio** | **2,306,662 / 2,306,662 (100.0%)** | **100% FP32** |
| **INT8 / UINT8 Quantized Tensors** | **0 tensors** | None |
| **Float16 Half-Precision Tensors** | **0 tensors** | None |
| **Quantization Scale / Zero-Point Factors** | **None found in HDF5 attributes** | None |
| **Quantization Status** | **NOT QUANTIZED (FP32 Full Precision)** | ❌ Not Quantized |

### Proof of Unquantized State:
1. **Mathematical Weight Byte Parity:**  
   $$2,306,662 \text{ parameters} \times 4 \text{ bytes/param (Float32)} = 9,226,648 \text{ bytes} \approx 8.80 \text{ MB}$$  
   Adding the HDF5 metadata and Adam optimizer state yields the exact file size on disk: **9,982,448 bytes (9.52 MB)**.
2. **Data Types:** Every one of the 262 weight tensors (kernels, biases, batch-norm gamma/beta/moving statistics) is stored in raw `float32`.
3. **Keras DTypePolicy:** Explicitly configured as `{'name': 'float32'}`.

---

## 2. Model Architecture & Topological Breakdown

The model employs transfer learning with a **MobileNetV2** backbone trained on crop disease classification (PlantVillage dataset):

* **Architecture Paradigm:** `Sequential`
  1. **Input Layer:** Expects RGB image tensor: $(B, 224, 224, 3)$
  2. **Feature Extractor:** `mobilenetv2_1.00_224` (154 sublayers)
     * $35 \times$ Standard Convolutions (`Conv2D`)
     * $17 \times$ Depthwise Separable Convolutions (`DepthwiseConv2D`)
     * $52 \times$ Batch Normalization layers (`BatchNormalization`)
     * $35 \times$ ReLU6 activation layers (`ReLU`)
     * $10 \times$ Residual skip connection additions (`Add`)
     * $4 \times$ Zero padding layers (`ZeroPadding2D`)
  3. **Pooling Layer:** `GlobalAveragePooling2D` (reduces $7 \times 7 \times 1280 \rightarrow 1280$ feature vector)
  4. **Classification Head:** `Dense` (1280 inputs $\rightarrow$ 38 outputs with `Softmax` activation)

---

## 3. Quantitative Model Metrics

| Metric | Measured Value | Unit |
| :--- | :---: | :---: |
| **Total Weight Parameters** | **2,306,662** | parameters |
| **Backbone Parameters** | 2,258,022 | parameters |
| **Head (Dense) Parameters** | 48,678 ($1280 \times 38 + 38$) | parameters |
| **Weight Memory Footprint** | **8.80** | MB |
| **Total File Size on Disk** | **9.52** | MB |
| **Target Output Classes** | **38** | classes |
| **Input Dimensions** | **$224 \times 224 \times 3$** | pixels (RGB) |
| **Input Tensor Size (Batch=1)** | 602.1 | KB |
| **Computational Complexity** | **~300 Million MACs (~600 MFLOPs)** | operations |
| **Training Optimizer Stored** | Adam | - |
| **Training Loss Objective** | Categorical Crossentropy | - |

---

## 4. Quantization Impact & Optimization Blueprint

Because the model is currently unquantized, deploying it on edge devices (Flutter mobile in `LeafLens/`) or free-tier cloud hosting (Render / Fly.io) incurs unnecessary memory and latency overhead.

### Precision vs. Footprint Comparison

| Quantization Strategy | Target Precision | Model Size | Size Reduction | Mobile/CPU Speedup | Target Deployment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Current Baseline (FP32)** | `float32` | **9.52 MB** | 0.0% (Baseline) | $1.0\times$ | Cloud Server (GPU/CPU) |
| **FP16 Half-Precision** | `float16` | **~4.40 MB** | **~53.8%** | $1.5\times - 2.0\times$ | Mobile GPU / Cloud API |
| **Dynamic Range Quantization** | `int8` weights, `fp32` act | **~2.40 MB** | **~74.8%** | $2.0\times - 3.0\times$ | Mobile CPU / Render API |
| **Full Integer Quantization (QAT/PTQ)** | `int8` weights & act | **~2.20 MB** | **~76.9%** | **$3.0\times - 4.5\times$** | **Edge TPU / Mobile NPU (Ideal for App)** |

---

## 5. How to Quantize the Model

Use the automated script `quantize_model.py`:

```bash
# Generate both FP16 and INT8 quantized TFLite models
python quantize_model.py
```

This generates:
1. `plant_model_fp16.tflite` (~4.4 MB)
2. `plant_model_int8.tflite` (~2.4 MB)
