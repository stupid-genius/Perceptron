# Perceptron
A full, multi-layer perceptron implemented in JavaScript. Uses dual numbers for auto-diff to perform backpropagation.

Inspired by the excellent video series by [Welch Labs on YouTube](https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLpjq_3E7axFoWk4xx5Vnsy0o9hEqnCE2Q&index=8).

## Features
- **Automatic Differentiation:** Uses a Dual Number implementation for exact gradients through a computation graph.
- **Multi-Layer Support:** Flexible architecture defined by simple schema arrays (e.g., `[2, 16, 1]`).
- **Layer Indexing:** Direct access to individual layers via Proxy (e.g., `p[0].activation = ...`).
- **Built-in Activations:** RELU, SIGMOID, TANH, STEP, and IDENTITY.
- **Built-in Loss Functions:** MSE, MAE, HUBER, and CROSS_ENTROPY.

## Example
```javascript
const { Perceptron } = require('./Perceptron.js');

// 1. Initialize: 2 inputs, 3 hidden neurons, 1 output
const p = new Perceptron([2, 3, 1]);

// 2. Configure layers (optional)
// You can set activation globally:
p.activation = Perceptron.RELU;

// Or fine-tune per-layer using the Proxy:
p[1].activation = Perceptron.SIGMOID;

// 3. Forward pass
const output = p.forward(0.5, -0.2);
console.log(`Prediction: ${output[0][0].real}`);

// 4. Training step
p.backward([1.0]); // Backpropagate error relative to target
p.update(0.1);     // Apply gradients with learning rate
```

TODO:
- add metrics and visualization tools
- add attention mechanism
