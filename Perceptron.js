const Logger = require('log-ng');
const path = require('node:path');
const { DualNumber, DualMatrix } = require('./Dual.js');

const logger = new Logger(path.basename(__filename));

/**
 * Perceptron constructor
 *
 * @param {number} [numInputs=2] - Number of input features
 * @param {number} [numOutputs=1] - Number of output neurons
 *
 * @example
 * // inference w/pre-trained weights
 * const p = new Perceptron(2, 1);
 * p.weights(0.4, 0.6, -0.2);
 * p.forward(1, 1);
 *
 * // train
 * const p = new Perceptron(2, 2);
 * p.activation = Perceptron.SIGMOID;
 * p.loss = Perceptron.MSE;
 * const trainingData = [[1, 1, [0.8, 0.2]]];
 * for(const [x1, x2, yTargets] of trainingData){
 *   p.forward(x1, x2);
 *   p.backward(yTargets);
 *   p.update(0.01);
 * }
 */
function Perceptron(numInputs = 2, numOutputs = 1){
	if(!new.target) {
		return new Perceptron(...arguments);
	}

	const weights = DualMatrix(numOutputs, numInputs + 1, new Float64Array(numOutputs * (numInputs + 1)).map(() => Math.random() * 2 - 1));
	let outputs = [];

	Object.defineProperties(this, {
		/**
		 * The activation function
		 * This function must operate on DualNumber instances
		 *
		 * @type {function}
		 * @example
		 * p.activation = (x) => {
		 *  return x.max(0); // ReLU
		 * };
		 */
		activation: {
			value: Perceptron.SIGMOID,
			writable: true
		},
		backward: {
			value: function(yTarget){
				const loss = this.loss(outputs, yTarget);
				loss.backprop();
				return this;
			}
		},
		forward: {
			value: function(){
				const inputData = new Float64Array(numInputs + 1);
				for(let i = 0; i < numInputs; ++i){
					inputData[i] = arguments[i] ?? 0;
				}
				inputData[numInputs] = 1; // bias input

				const inputVector = DualMatrix(numInputs + 1, 1, inputData);
				const sums = weights.multiply(inputVector);

				outputs = sums.map(this.activation);
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
		update: {
			value: function(learningRate){
				for(let i = 0; i < weights.real.data.length; ++i){
					weights.real.data[i] -= learningRate * weights.grad.data[i];
				}
				weights.zeroGrads();
				return this;
			}
		},
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
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);
				const diff = pred.sub(target);
				totalLoss = totalLoss.add(diff.mul(diff).mul(0.5));
			}
			return totalLoss;
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
			return totalLoss;
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
				const loss = diff.abs().clip(0, delta).mul(0.5).add(diff.abs().sub(delta).max(0).mul(delta));
				totalLoss = totalLoss.add(loss);
			}
			return totalLoss;
		}
	},
	CROSS_ENTROPY: {
		value: (yPred, yTarget) => {
			let totalLoss = DualNumber(0);
			const numOutputs = yPred.dimensions[0];
			const one = DualNumber(1, 0);
			for(let i = 0; i < numOutputs; ++i){
				const pred = yPred[i][0];
				const target = Array.isArray(yTarget) ? yTarget[i] : (typeof yTarget === 'number' ? yTarget : yTarget.data[i]);

				const targetDN = typeof target === 'number' ? DualNumber(target) : target;
				// Note: log(0) is -Infinity. If pred is 0 or 1, this will crash or return NaN.
				// Maybe add a small epsilon (e.g., 1e-7) to pred to prevent this.
				const logYPred = pred.log();
				const logOneMinusYPred = one.sub(pred).log();
				const loss = targetDN.mul(logYPred).add(one.sub(targetDN).mul(logOneMinusYPred)).mul(-1);
				totalLoss = totalLoss.add(loss);
			}
			return totalLoss;
		}
	}
});

module.exports = Perceptron;
