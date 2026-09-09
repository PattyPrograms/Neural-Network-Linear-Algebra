# Neural Network Linear Algebra Project

An applied linear algebra project that connects matrix operations, vector dimensions, activation functions, loss functions, and the chain rule to the forward and backward passes of a small neural network.

The project uses **Python in Jupyter Notebook** to verify detailed handwritten calculations for a neural network regression example based on the Iris dataset.

## Project Overview

The network uses a **3 → 4 → 1 architecture**:

- **3 input features:** sepal length, sepal width, and petal width
- **4 hidden neurons**
- **1 continuous output:** predicted petal length
- **Hidden activation:** hyperbolic tangent (`tanh`)
- **Output activation:** linear
- **Loss:** \(\frac{1}{2}(\hat{y}-y)^2\)

The notebook uses the first Iris data point for the numerical verification:

- Sepal length: 5.1
- Sepal width: 3.5
- Petal width: 0.2
- Target petal length: 1.4

A fixed random seed (`20`) is used so the initialized weights are reproducible.

## What I Worked Through

### Forward Pass

I computed and verified:

1. Hidden-layer pre-activations

   \[
   \mathbf{a}^{(1)} = W^{(1)}\mathbf{x} + \mathbf{b}^{(1)}
   \]

2. Hidden-layer activations

   \[
   \mathbf{z}^{(1)} = \tanh(\mathbf{a}^{(1)})
   \]

3. Output-layer prediction

   \[
   a^{(2)} = W^{(2)}\mathbf{z}^{(1)} + b^{(2)}
   \]

4. Squared-error loss

   \[
   E = \frac{1}{2}(\hat{y}-y)^2
   \]

### Backpropagation

I derived and verified gradients using the chain rule, including:

- \(\frac{\partial E}{\partial a^{(2)}}\)
- \(\frac{\partial E}{\partial W^{(2)}}\)
- \(\frac{\partial E}{\partial b^{(2)}}\)
- propagation of the output error back to the hidden layer
- derivative of `tanh`
- \(\frac{\partial E}{\partial W^{(1)}}\)
- \(\frac{\partial E}{\partial \mathbf{b}^{(1)}}\)

The handwritten work shows the matrix arithmetic, dimensional checks, derivative steps, and numerical verification used alongside the notebook.

## Numerical Results

For the selected weights and first Iris observation:

- Hidden pre-activations: approximately `[1.8434, 1.3413, 0.8852, -2.9651]`
- Hidden activations: approximately `[0.9511, 0.8720, 0.7090, -0.9947]`
- Prediction: approximately `-0.9782`
- Target: `1.4`
- Loss: approximately `2.8278`

The goal of the project is not to train a production model, but to understand and verify the **linear algebra and calculus underlying neural-network computation**.

## Technologies

- Python
- Jupyter Notebook
- NumPy
- scikit-learn
- Linear algebra
- Calculus / chain rule
- Neural-network fundamentals

## Repository Structure

```text
.
├── FinalLinearProjectCode.ipynb
├── handwritten-work.pdf
├── HANDWRITTEN_WORK.md
├── requirements.txt
├── .gitignore
└── README.md
```

## Handwritten Work

The full handwritten derivations are included in **[handwritten-work.pdf](handwritten-work.pdf)**.

These 11 pages document the manual calculations used to verify the Python results, including:

- network architecture and dimension verification
- weight initialization
- hidden-layer matrix-vector multiplication
- activation calculations
- output and loss calculations
- chain-rule derivations
- `tanh` derivative work
- gradients for the output and hidden layers
- final comparison between manual and computed values

`HANDWRITTEN_WORK.md` provides a concise written guide to what is covered in the derivation pages.

## Running the Notebook

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then open `FinalLinearProjectCode.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab and run the cells in order.

## Skills Demonstrated

This project demonstrates the ability to connect mathematical theory with code by:

- translating matrix and vector equations into NumPy operations
- checking mathematical dimensions against NumPy array shapes
- implementing a neural-network forward pass
- applying nonlinear activation functions
- computing prediction loss
- deriving backpropagation gradients using the chain rule
- validating handwritten calculations computationally
