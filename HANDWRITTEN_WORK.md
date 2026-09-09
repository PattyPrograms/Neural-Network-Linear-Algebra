# Handwritten Derivation Notes

The original project included 11 handwritten pages used to manually verify the Python/Jupyter calculations. The work covered the same 3 → 4 → 1 neural-network example shown in the notebook.

## What the handwritten work verifies

### 1. Network architecture and dimensions
- Input vector with 3 Iris features
- First-layer weight matrix with shape 4 × 3
- Hidden-layer bias vector with shape 4 × 1
- Second-layer weight matrix with shape 1 × 4
- Scalar output bias
- Dimension checks for every matrix-vector product

### 2. Forward pass
The handwritten calculations evaluate:

- `a1 = W1x + b1`
- `z1 = tanh(a1)`
- `a2 = W2z1 + b2`
- `ŷ = a2`
- `E = 1/2(ŷ - y)^2`

The numerical results are checked against the Jupyter notebook to confirm the matrix arithmetic and nonlinear activation calculations.

### 3. Backpropagation
The handwritten derivation applies the chain rule to calculate:

- `∂E/∂a2`
- `∂E/∂W2`
- `∂E/∂b2`
- the hidden-layer error signal
- the derivative of `tanh`, `1 - z²`
- `∂E/∂a1`
- `∂E/∂W1`
- `∂E/∂b1`

### 4. Explicit chain-rule verification
A specific first-layer weight derivative is expanded as a product of individual chain-rule factors and compared with the corresponding entry in the computed gradient matrix.

## Purpose

The handwritten work was not separate from the code; it was the mathematical derivation that the Python notebook was designed to verify. Together, the two parts demonstrate how the linear algebra and calculus behind a neural network map directly to NumPy operations.

> The original handwritten pages are retained with the project materials. This repository currently provides this structured transcription alongside the cleaned notebook so the derivation is readable directly on GitHub.
