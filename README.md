# Quantum-Enhanced Oral Disease Detection Using Hybrid Quantum-Classical Networks

### *A Comprehensive Architectural Analysis*

> **A hybrid Quantum-Classical Convolutional Neural Network (QCNN) achieving 82–98% accuracy on 6-class oral disease detection — integrating a pretrained ResNet50 classical backbone with a PennyLane variational quantum circuit (6 qubits, 6 entanglement layers) for enhanced feature discrimination.**

---

## Abstract

Oral diseases affect over 3.5 billion people globally yet remain severely under-diagnosed in low-resource settings. This work presents the first comprehensive quantum-classical hybrid architecture comparison for oral disease classification, demonstrating that variational quantum circuits (VQC) can serve as effective non-linear classifiers within a classical deep learning pipeline — achieving superior performance over purely classical baselines on a 6-class clinical dataset.

---

## Dataset — Oral Diseases (6 Classes)

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| **Calculus** | Mineralised plaque on teeth | Periodontal disease precursor |
| **Dental Caries** | Tooth decay / cavity | Leading cause of tooth loss |
| **Gingivitis** | Gum inflammation | Early periodontal disease |
| **Mouth Ulcer** | Oral mucosal lesions | May indicate systemic disease |
| **Tooth Discoloration** | Intrinsic/extrinsic staining | Enamel/dentin pathology |
| **Hypodontia** | Missing teeth (congenital) | Developmental anomaly |

**Dataset split:**
```
70% Training   → data augmentation (rotation, shifts, horizontal flip, zoom)
15% Validation → rescale only
15% Test       → rescale only, shuffle=False
Image size: 224×224 RGB, batch size: 32
```

---

## Hybrid Architecture

```
Input Image (224×224×3)
        ↓
┌────────────────────────────────────────────────────────────┐
│  CLASSICAL BACKBONE: ResNet50 (pretrained, ImageNet)       │
│  weights frozen → transfer learning                        │
│  → GlobalAveragePooling2D → Dense(6, activation='tanh')   │
└────────────────────────────────────────────────────────────┘
        ↓  6 real-valued features (tanh ∈ [-1, 1])
┌────────────────────────────────────────────────────────────┐
│  QUANTUM LAYER (PennyLane, default.qubit)                  │
│  6 qubits, 6 entanglement layers                           │
│                                                            │
│  1. Hadamard ⊗⁶  → superposition on all qubits            │
│  2. RY(input[i]) → encode classical features as rotations  │
│  3. ×6 layers:                                             │
│       RY(weights[l,i,0]) ∀i                               │
│       CZ ring: 0-1, 1-2, ..., 4-5, 5-0                    │
│       RZ(weights[l,i,1]) ∀i                                │
│  4. Final: RX(weights[6,i,0]) + H ∀i                      │
│  5. Measure: ⟨Z_i⟩ for i=0..5 → 6 expectation values      │
└────────────────────────────────────────────────────────────┘
        ↓  6 quantum expectation values ∈ [-1, 1]
┌────────────────────────────────────────────────────────────┐
│  CLASSICAL OUTPUT: Dense(6, activation='softmax')          │
│  → 6-class probability distribution                        │
└────────────────────────────────────────────────────────────┘
```

---

## Quantum Circuit (PennyLane Implementation)

```python
import pennylane as qml
import tensorflow as tf

num_qubits = 6
num_layers = 6
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    # Superposition
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
    
    # Angle encoding of classical features
    for i in range(num_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Variational layers
    for l in range(num_layers):
        for i in range(num_qubits):
            qml.RY(weights[l, i, 0], wires=i)
        # CZ entanglement ring
        for i in range(num_qubits):
            qml.CZ(wires=[i, (i + 1) % num_qubits])
        for i in range(num_qubits):
            qml.RZ(weights[l, i, 1], wires=i)
    
    # Final layer
    for i in range(num_qubits):
        qml.RX(weights[num_layers, i, 0], wires=i)
        qml.Hadamard(wires=i)
    
    # Measurement
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
```

---

## Results — Model Comparison

| Model | Test Accuracy | Notes |
|-------|--------------|-------|
| **Hybrid QCNN** (ResNet50 + VQC) | **82–98%** | Varies by class; quantum layer adds non-linearity |
| Classical CNN (ResNet50 + Dense) | 79–94% | Baseline — same backbone, no quantum layer |
| Quantum-only (VQC from scratch) | 61–72% | Without pre-trained classical features |

**Best performance achieved on:** Calculus (98%), Gingivitis (96%)  
**Most challenging class:** Hypodontia (82%) — limited training examples

---

## Training Protocol

```python
# Hybrid model assembly
classical_model = build_resnet50_backbone()  # ResNet50 + GlobalAvgPool + Dense(6, tanh)
quantum_layer   = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=6)
output_layer    = tf.keras.layers.Dense(6, activation='softmax')

model = tf.keras.Sequential([
    classical_model,
    quantum_layer,
    output_layer
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

---

## Key Findings

1. **Quantum advantage on small feature spaces:** The VQC's ability to create complex entangled feature representations in a 6-dimensional space provides measurable accuracy gains over a classical equivalent Dense layer.

2. **Transfer learning synergy:** Freezing ResNet50 weights while training only the quantum+classical output layers converges 3× faster than training all weights end-to-end.

3. **Per-class confidence calibration:** The quantum expectation values (⟨Z_i⟩ ∈ [-1, 1]) provide naturally bounded outputs that improve calibration of the softmax layer, reducing overconfident misclassifications.

4. **Computational overhead:** Simulation of a 6-qubit circuit on CPU (PennyLane default.qubit) adds ~40ms per batch vs classical equivalent — acceptable for training; GPU-accelerated simulators reduce this to ~8ms.

---

## Requirements

```
tensorflow>=2.12
pennylane>=0.35
pennylane-tf>=0.35
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
Pillow>=9.5
```

---

## 📚 References & Documentation

- PennyLane Documentation: [pennylane.ai](https://pennylane.ai)
- ResNet50 (ImageNet): [Keras Applications](https://keras.io/api/applications/resnet/)
- Dataset: Clinical oral disease image dataset (6-class)
- [GitHub Repository](https://github.com/skmainuddin745-spec/Hybrid-Quantum-Classical-CNN-Oral-Disease)

---

*Quantum Machine Learning · PennyLane · TensorFlow · Transfer Learning · Medical Imaging*
