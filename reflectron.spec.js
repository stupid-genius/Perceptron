const { assert } = require('chai');
const Logger = require('log-ng');
const path = require('node:path');
const { DualMatrix } = require('./Dual.js');
const { Reflectron } = require('./Reflectron.js');
const { Perceptron } = require('./Perceptron.js');

const logger = new Logger(path.basename(__filename));

describe('Reflectron (Attention Demo)', function(){
	it('should perform a forward pass', function(){
		const dim = 4;
		const r = new Reflectron(dim);

		const x = DualMatrix(dim, 1, [1, 0, 1, 0]);

		const output = r.forward(x);

		logger.info(`Input vector:\n${x.real.toString()}`);
		logger.info(`Reflectron output:\n${output.real.toString()}`);

		assert.equal(output.dimensions[0], dim);
		assert.equal(output.dimensions[1], 1);
		assert.notDeepEqual(output.real.data, x.real.data);
	});

	it('should allow gradients to flow back to internal projections', function(){
		const dim = 2;
		const r = new Reflectron(dim);
		const x = DualMatrix(dim, 1, [1, 1]);

		const output = r.forward(x);

		const target = [2, 2];
		const loss = Perceptron.MSE(output, target);

		logger.info(`Initial Loss: ${loss.real.toFixed(4)}`);

		loss.backprop();

		// Verify that gradients were calculated by checking if weights move after update
		const weightsBefore = r.weights().slice();
		const initialLossValue = loss.real;

		r.update(0.1);
		const weightsAfter = r.weights();

		assert.notDeepEqual(weightsBefore, weightsAfter, 'Weights should have moved after backprop + update');

		const secondOutput = r.forward(x);
		const secondLossValue = Perceptron.MSE(secondOutput, target).real;

		logger.info(`Initial Loss: ${initialLossValue.toFixed(6)}, Loss after update: ${secondLossValue.toFixed(6)}`);

		assert.ok(secondLossValue < initialLossValue, 'Loss should decrease after one update step');
		assert.notDeepEqual(output.real.data, secondOutput.real.data, 'Output should have moved');
	});

	it('should learn a simple scaling transformation', function(){
		this.timeout(5000);
		const dim = 3;
		const r = new Reflectron(dim);

		const input = [1, 0.5, -1];
		const target = [2, 1, -2]; // We want it to learn to double the input

		let initialLoss = 0;

		// Training loop
		for(let i = 0; i < 100; i++){
			const x = DualMatrix(dim, 1, input);
			const y = r.forward(x);
			const loss = Perceptron.MSE(y, target);

			if(i === 0) initialLoss = loss.real;

			loss.backprop();
			r.update(0.01);
			y.zeroGrads();
		}
		const finalOutput = r.forward(DualMatrix(dim, 1, input));
		const finalLoss = Perceptron.MSE(finalOutput, target).real;

		logger.info(`Scaling Task: Initial Loss: ${initialLoss.toFixed(6)}, Final Loss: ${finalLoss.toFixed(6)}`);
		logger.info(`Final Output:\n${finalOutput.real.toString()}`);

		assert.ok(finalLoss < initialLoss, 'Reflectron failed to converge on scaling task');
	});
});

