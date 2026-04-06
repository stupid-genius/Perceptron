const Logger = require('log-ng');
const path = require('node:path');
const { DualNumber, DualMatrix } = require('./Dual.js');

const logger = new Logger(path.basename(__filename));

/**
 * Layer constructor
 *
 * @param {number} numInputs - Number of input features
 * @param {number} numOutputs - Number of neurons in this layer
 */
function Layer(numInputs, numOutputs){
	if(!new.target){
		return new Layer(...arguments);
	}

	const weights = DualMatrix(numOutputs, numInputs, new Float64Array(numOutputs * numInputs).map(() => {
		return Math.random() * 2 - 1;
	}));
	const bias = DualMatrix(numOutputs, 1, new Float64Array(numOutputs).map(() => {
		return Math.random() * 2 - 1;
	}));

	Object.defineProperties(this, {
		/**
		 * The activation function
		 * @type {function}
		 */
		activation: {
			value: Perceptron.SIGMOID,
			writable: true
		},
		/**
		 * The bias vector getter/setter
		 * @param {Float64Array|Array<number>|...number} [data] - New bias data to set
		 * @returns {Float64Array|this} The bias data if no arguments, or 'this' for chaining
		 */
		bias: {
			value: function(){
				if(arguments.length === 0){
					return bias.real.data;
				}
				if(arguments.length === 1 && (arguments[0] instanceof Float64Array || Array.isArray(arguments[0]))){
					bias.real.data.set(arguments[0]);
				}else{
					bias.real.data.set(arguments);
				}
				return this;
			}
		},
		/**
		 * Forward pass through the layer
		 * @param {DualMatrix} inputVector - The input column vector (numInputs x 1)
		 * @returns {DualMatrix} The output column vector (numOutputs x 1)
		 */
		forward: {
			value: function(inputVector){
				return weights.multiply(inputVector).add(bias).map(this.activation);
			}
		},
		numInputs: {
			value: numInputs
		},
		numOutputs: {
			value: numOutputs
		},
		/**
		 * Update weights and biases using accumulated gradients
		 * @param {number} learningRate
		 * @returns {this}
		 */
		update: {
			value: function(learningRate){
				for(let i = 0; i < weights.real.data.length; ++i){
					weights.real.data[i] -= learningRate * weights.grad.data[i];
				}
				weights.grad.data.fill(0);

				for(let i = 0; i < bias.real.data.length; ++i){
					bias.real.data[i] -= learningRate * bias.grad.data[i];
				}
				bias.grad.data.fill(0);

				return this;
			}
		},
		/**
		 * The weights matrix getter/setter
		 * @param {Float64Array|Array<number>|...number} [data] - New weights data to set
		 * @returns {Float64Array|this} The weights data if no arguments, or 'this' for chaining
		 */
		weights: {
			value: function(){
				if(arguments.length === 0){
					return weights.real.data;
				}
				if(arguments.length === 1 && (arguments[0] instanceof Float64Array || Array.isArray(arguments[0]))){
					weights.real.data.set(arguments[0]);
				}else{
					weights.real.data.set(arguments);
				}
				return this;
			}
		}
	});
}

/**
 * Perceptron constructor (General MLP implementation)
 *
 * @param {Array<number>} [schema=[2, 1]] - Array defining the architecture [numInputs, ...hidden, numOutputs]
 *
 * @example
 * // SLP (2 inputs, 1 output)
 * const p = new Perceptron([2, 1]);
 *
 * // MLP (2 inputs, 16 hidden, 1 output)
 * const p = new Perceptron([2, 16, 1]);
 */
function Perceptron(schema = [2, 1]){
	if(!new.target){
		return new Perceptron(...arguments);
	}

	const layers = [];
	for(let i = 0; i < schema.length - 1; ++i){
		layers.push(new Layer(schema[i], schema[i + 1]));
	}

	let outputs = null;

	Object.defineProperties(this, {
		/**
		 * The activation function for all layers
		 * This function must operate on DualNumber instances
		 *
		 * @type {function}
		 * @example
		 * p.activation = (x) => {
		 *  return x.max(0); // ReLU
		 * };
		 */
		activation: {
			get: () => {
				return layers[0].activation;
			},
			set: (fn) => {
				for(const layer of layers){
					layer.activation = fn;
				}
			}
		},
		/**
		 * Backward pass: calculate loss and trigger backprop
		 * @param {Array|number} yTarget - Target values
		 * @returns {this}
		 */
		backward: {
			value: function(yTarget){
				if(!outputs){
					throw new Error('Must call forward before backward');
				}
				const lossNode = this.loss(outputs, yTarget);
				lossNode.backprop();
				return this;
			}
		},
		/**
		 * Forward pass through all layers
		 * @param {...number} args - Input features
		 * @returns {DualMatrix} The output column vector (numOutputs x 1)
		 */
		forward: {

			value: function(...args){
				const inputData = new Float64Array(schema[0]);
				for(let i = 0; i < schema[0]; ++i){
					inputData[i] = args[i] ?? 0;
				}
				let current = DualMatrix(schema[0], 1, inputData);

				for(const layer of layers){
					current = layer.forward(current);
				}

				outputs = current;
				return outputs;
			}
		},
		/**
		 * The loss function
		 * This function must operate on DualMatrix (yPred) and raw target values (yTarget)
		 *
		 * @type {function}
		 * @example
		 * p.loss = (yPred, yTarget) => {
		 *  let totalLoss = DualNumber(0);
		 *  for(let i = 0; i < yPred.dimensions[0]; i++){
		 *    const diff = yPred[i][0].sub(yTarget[i]);
		 *    totalLoss = totalLoss.add(diff.mul(diff).mul(0.5));
		 *  }
		 *  return totalLoss;
		 * };
		 */
		loss: {
			value: Perceptron.MSE,
			writable: true
		},
		/**
		 * Update all layers' weights and biases
		 * @param {number} learningRate
		 * @returns {this}
		 */
		update: {
			value: function(learningRate){
				for(const layer of layers){
					layer.update(learningRate);
				}
				return this;
			}
		},
		/**
		 * Get or set all weights and biases as a single flat array
		 * @param {Float64Array|Array<number>|...number} [data] - New weights/biases to set
		 * @returns {Float64Array|this} All weights/biases if no arguments, or 'this' for chaining
		 */
		weights: {
			value: function(){
				if(arguments.length === 0){
					let totalSize = 0;
					for(const l of layers){
						totalSize += l.weights().length + l.bias().length;
					}
					const all = new Float64Array(totalSize);
					let offset = 0;
					for(const l of layers){
						all.set(l.weights(), offset);
						offset += l.weights().length;
						all.set(l.bias(), offset);
						offset += l.bias().length;
					}
					return all;
				}

				let allData;
				if(arguments.length === 1 && (arguments[0] instanceof Float64Array || Array.isArray(arguments[0]))){
					allData = arguments[0];
				}else{
					allData = Array.from(arguments);
				}

				let offset = 0;
				for(const l of layers){
					const wSize = l.numInputs * l.numOutputs;
					const bSize = l.numOutputs;
					l.weights(allData.slice(offset, offset + wSize));
					offset += wSize;
					l.bias(allData.slice(offset, offset + bSize));
					offset += bSize;
				}
				return this;
			}
		}
	});

	return new Proxy(this, {
		get(target, prop, receiver){
			if(typeof prop === 'string'){
				const idx = Number(prop);
				if(!Number.isNaN(idx) && idx >= 0 && idx < layers.length){
					return layers[idx];
				}
				if(prop === 'length'){
					return layers.length;
				}
			}
			return Reflect.get(target, prop, receiver);
		}
	});
}
Object.defineProperties(Perceptron, {
	IDENTITY: {
		value: { f: x => x, df: _x => 1 }
	},
	STEP: {
		value: { f: x => x >= 0 ? 1 : 0, df: _x => 0 }
	},
	RELU: {
		value: { f: x => x >= 0 ? x : 0, df: x => x > 0 ? 1 : 0 }
	},
	SIGMOID: {
		value: {
			f: x => 1 / (1 + Math.exp(-x)),
			df: x => {
				const s = 1 / (1 + Math.exp(-x));
				return s * (1 - s);
			}
		}
	},
	TANH: {
		value: {
			f: x => Math.tanh(x),
			df: x => {
				const t = Math.tanh(x);
				return 1 - t * t;
			}
		}
	},
	SOFTMAX: {
		value: function(x){
			const max = Math.max(...x.real.data);
			const exps = x.real.data.map(v => Math.exp(v - max));

			const sumExps = exps.reduce((a, b) => a + b, 0);
			const out = DualMatrix(x.dimensions[0], 1, new Float64Array(exps.map(e => e / sumExps)));
			out.backward = () => {
				for(let i = 0; i < x.dimensions[0]; ++i){
					let gradSum = 0;
					for(let j = 0; j < x.dimensions[0]; ++j){
						const s_i = out[i][0].real;
						const s_j = out[j][0].real;
						const gradContribution = (i === j ? s_i * (1 - s_i) : -s_i * s_j) * out.grad.data[j];
						gradSum += gradContribution;
					}
					x.grad.data[i] += gradSum;
				}
			};
			out.parents.push(x);
			return out;
		}
	}
});
Object.defineProperties(Perceptron, {
	MSE: {
		value: (yPred, yTarget) => {
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);
				const diff = pred.sub(target);
				totalLoss = totalLoss.add(diff.mul(diff).mul(0.5));
			}
			return totalLoss.div(numOutputs);
		}
	},
	MAE: {
		value: (yPred, yTarget) => {
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);
				totalLoss = totalLoss.add(pred.sub(target).abs());
			}
			return totalLoss.div(numOutputs);
		}
	},
	HUBER: {
		value: (yPred, yTarget, delta = 1.0) => {
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);
				const diff = pred.sub(target);
				const absDiff = diff.abs();
				const quadratic = absDiff.min(delta);
				const linear = absDiff.sub(quadratic);
				const loss = quadratic.mul(quadratic).mul(0.5).add(linear.mul(delta));
				totalLoss = totalLoss.add(loss);
			}
			return totalLoss.div(numOutputs);
		}
	},
	CROSS_ENTROPY: {
		/**
		 * Binary Cross Entropy Loss
		 * Expects yPred to be probabilities in range [0, 1] (e.g. from SIGMOID)
		 */
		value: (yPred, yTarget) => {
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			const one = DualNumber(1, 0);
			const epsilon = 1e-7;
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);

				const targetDN = typeof target === 'number' ? DualNumber(target) : target;

				// Clamping prediction to [epsilon, 1 - epsilon] for stability
				const safePred = pred.max(epsilon).min(1 - epsilon);
				const logYPred = safePred.log();
				const logOneMinusYPred = one.sub(safePred).log();
				const loss = targetDN.mul(logYPred).add(one.sub(targetDN).mul(logOneMinusYPred)).mul(-1);
				totalLoss = totalLoss.add(loss);
			}
			return totalLoss.div(numOutputs);
		}
	}
});

module.exports = {
	Perceptron,
	Layer
};
