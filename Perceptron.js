const Logger = require('log-ng');
const path = require('node:path');
const DualNumber = require('./Dual.js');
// const Matrix = require('./Matrix.js');

const logger = new Logger(path.basename(__filename));

/**
 * Perceptron constructor
 *
 * @example
 * // inference w/pre-trained weights
 * perceptron.weights(0.4, 0.6, -0.2).activation(fn).forward(x1, x2);
 *
 * // train
 * perceptron.activation(fn).loss(fn);
 * const trainingData = [];
 * for(const [x1, x2, yTarget] of trainingData){
 *   perceptron.forward(x1, x2);
 *   perceptron.backward(yTarget);
 * }
 * perceptron.update(0.01);
 */
function Perceptron(numInputs = 2){
	if(!new.target) {
		return new Perceptron(...arguments);
	}

	const inputs = [];
	const weights = Array(numInputs + 1).fill(0).map(() => DualNumber(Math.random() * 2 - 1));
	const outputs = [];

	Object.defineProperties(this, {
		/**
		 * Set the activation function
		 * This function must operate on DualNumber instances
		 *
		 * @param {function} fn - Activation function that takes a DualNumber and returns a DualNumber
		 * @example
		 * perceptron.activation((x) => {
		 *  const dnX = DualNumber(x, 1);
		 *  const dnOut = dnX.real < 0 ? DualNumber(0, 0) : dnX;
		 *  return dnOut;
		 * });
		 */
		activation: {
			value: Perceptron.RELU,
			writable: true
		},
		backward: {
			value: function(yTarget){
				const loss = this.loss(outputs[0], yTarget);
				loss.backprop();
				return this;
			}
		},
		forward: {
			value: function(){
				for(let i = 0; i < arguments.length; ++i){
					inputs[i] = DualNumber(arguments[i]);
				}
				inputs[arguments.length] = DualNumber(1); // bias input
				let sum = DualNumber(0);
				for(let i = 0; i < inputs.length; ++i){
					sum = sum.add(weights[i].mul(inputs[i]));
				}

				const output = this.activation(sum);
				outputs[0] = output;
				return output;
			}
		},
		/**
		 * Set the loss function
		 * This function must operate on DualNumber instances
		 *
		 * @param {function} fn - Loss function that takes (yPred: DualNumber, yTarget: number) and returns a DualNumber
		 * @example
		 * perceptron.loss((yPred, yTarget) => {
		 *  const diff = yPred.sub(DualNumber(yTarget, 0));
		 *  return diff.mul(diff).mul(DualNumber(0.5, 0));
		 * });
		 */
		loss: {
			value: Perceptron.MSE,
			writable: true
		},
		update: {
			value: function(learningRate){
				for(let i = 0; i < weights.length; ++i){
					weights[i].real -= learningRate * weights[i].grad;
					weights[i].grad = 0; // reset so we can accumulate again
				}
				return this;
			}
		},
		weights: {
			value: function(){
				if(arguments.length === 0){
					return weights.map(w => w.real);
				}

				for(let i = 0; i < arguments.length; ++i){
					weights[i] = DualNumber(arguments[i]);
				}
				return this;
			}
		}
	});
}
Object.defineProperties(Perceptron, {
	IDENTITY: {
		value: function(x){
			return x;
		}
	},
	STEP: {
		value: function(x){
			const out = DualNumber(x.real >= 0 ? 1 : 0);
			out.backward = () => {};
			out.parents.push(x);
			return out;
		}
	},
	RELU: {
		value: function(x){
			const out = DualNumber(x.real >= 0 ? x.real : 0);
			out.backward = () => {
				x.grad += (x.real > 0 ? 1 : 0) * out.grad;
			};
			out.parents.push(x);
			return out;
		}
	},
	SIGMOID: {
		value: function(x){
			const s = 1 / (1 + Math.exp(-x.real));
			const out = DualNumber(s);
			out.backward = () => {
				x.grad += s * (1 - s) * out.grad;
			};
			out.parents.push(x);
			return out;
		}
	},
	TANH: {
		value: function(x){
			const t = Math.tanh(x.real);
			const out = DualNumber(t);
			out.backward = () => {
				x.grad += (1 - t * t) * out.grad;
			};
			out.parents.push(x);
			return out;
		}
	}
});
Object.defineProperties(Perceptron, {
	MSE: {
		value: (yPred, yTarget) => {
			const diff = yPred.sub(yTarget);
			return diff.mul(diff).mul(0.5);
		}
	}
});

module.exports = Perceptron;
