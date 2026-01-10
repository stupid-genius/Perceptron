const Logger = require('log-ng');
const path = require('node:path');
// const Matrix = require('./Matrix.js');

const logger = new Logger(path.basename(__filename));

/**
 * Perceptron constructor
 *
 * @example
 * // predict w/pre-trained weights
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
function Perceptron(){
	if(!new.target) {
		return new Perceptron(...arguments);
	}

	const inputs = [];
	const weights = Array(3).fill(0).map(() => Math.random() * 2 - 1);
	const gradients = Array(weights.length).fill(0);
	let activationFn = (x) => x; // identity by default
	let lossFn = (yPred, yTarget) => 0.5 * Math.pow(yPred - yTarget, 2); // MSE by default

	Object.defineProperties(this, {
		activation: {
			value: function(fn){
				activationFn = fn;
				return this;
			}
		},
		backward: {
			// hardcoded gradient for now
			value: function(yTarget){
				const yPred = this.forward(inputs[0], inputs[1]);

				const dLoss_dYPred = yPred - yTarget; // d(MSE)/d(yPred)
				const dYPred_dSum = 1; // derivative of activation (identity) is 1
				const dSum_dWeights = inputs; // partial derivatives w.r.t. weights

				const newGradients = weights.map((_, i) => dLoss_dYPred * dYPred_dSum * dSum_dWeights[i]);

				for(let i = 0; i < weights.length; ++i){
					gradients[i] = newGradients[i];
				}

				return this;
			}
		},
		forward: {
			value: function(x1, x2){
				inputs[0] = x1;
				inputs[1] = x2;
				inputs[2] = 1; // bias input
				let sum = 0;
				for(let i = 0; i < inputs.length; ++i){
					sum += inputs[i] * weights[i];
				}

				return activationFn(sum);
			}
		},
		loss: {
			// what a beast; just hardcoding for now
			// will likely need to integrate a CAS to make this actually work
			value: function(fn){
				lossFn = fn;
				return this;
			}
		},
		update: {
			value: function(learningRate){
				for(let i = 0; i < weights.length; ++i){
					weights[i] -= learningRate * gradients[i];
				}
				return this;
			}
		},
		weights: {
			value: function(){
				for(let i = 0; i < arguments.length; ++i){
					weights[i] = arguments[i];
				}
				return this;
			}
		}
	});
}

module.exports = Perceptron;
