const tf = require('@tensorflow/tfjs');
const { assert } = require('chai');
const Logger = require('log-ng');
const path = require('node:path');

const logger = new Logger(path.basename(__filename));

describe('TensorFlow.js', function(){
	this.timeout(5e3);

	before(async () => {
		await tf.setBackend('cpu');
		await tf.ready();
	});

	it('should solve XOR using the Layers API', async function(){
		// Sequntial model, joined end-to-end; similar to Perceptron Layers
		const model = tf.sequential();

		// Hidden Layer: 4 neurons, input shape [2]
		model.add(tf.layers.dense({
			units: 4,
			inputShape: [2],
			activation: 'sigmoid',
			// glorotUniform (Xavier) helps SGD start in a more 'balanced' place
			kernelInitializer: 'glorotUniform'
		}));

		// Output Layer: 1 neuron
		model.add(tf.layers.dense({
			units: 1,
			activation: 'sigmoid',
			kernelInitializer: 'glorotUniform'
		}));

		// 'sgd' is Stochastic Gradient Descent (what Perceptron uses)
		model.compile({
			optimizer: tf.train.sgd(0.2),
			// optimizer: tf.train.adam(0.1),
			loss: 'meanSquaredError'
		});

		const xs = tf.tensor2d([
			[0, 0],
			[0, 1],
			[1, 0],
			[1, 1]
		]);

		const ys = tf.tensor2d([
			[0],
			[1],
			[1],
			[0]
		]);

		// Train the model; fit() handles the loops
		await model.fit(xs, ys, {
			epochs: 5e3,
			verbose: 1
		});

		const results = model.predict(xs);
		const data = await results.data();

		logger.info(`TF XOR Results: [0,0]->${data[0].toFixed(3)}, [0,1]->${data[1].toFixed(3)}, [1,0]->${data[2].toFixed(3)}, [1,1]->${data[3].toFixed(3)}`);

		assert.approximately(data[0], 0, 0.15, 'TF should solve XOR [0,0]');
		assert.approximately(data[1], 1, 0.15, 'TF should solve XOR [0,1]');
		assert.approximately(data[2], 1, 0.15, 'TF should solve XOR [1,0]');
		assert.approximately(data[3], 0, 0.15, 'TF should solve XOR [1,1]');

		xs.dispose();
		ys.dispose();
		results.dispose();
		model.dispose();
	});

	it('should solve XOR using the low-level Ops API', async function(){
		const epochs = 5e3;

		// Hidden layer: [2, 4], Output layer: [4, 1]
		const w1 = tf.variable(tf.randomNormal([2, 4]));
		const b1 = tf.variable(tf.zeros([4]));
		const w2 = tf.variable(tf.randomNormal([4, 1]));
		const b2 = tf.variable(tf.zeros([1]));

		const optimizer = tf.train.sgd(0.5);

		const xs = tf.tensor2d([[0, 0], [0, 1], [1, 0], [1, 1]]);
		const ys = tf.tensor2d([[0], [1], [1], [0]]);

		const predict = (x) => {
			return tf.tidy(() => {
				const l1 = x.matMul(w1).add(b1).sigmoid();
				return l1.matMul(w2).add(b2).sigmoid();
			});
		};

		for(let i = 0; i < epochs; ++i){
			optimizer.minimize(() => {
				const pred = predict(xs);
				return tf.losses.meanSquaredError(ys, pred);
			});
		}

		const results = predict(xs);
		const data = await results.data();

		logger.info(`TF Ops XOR Results: [0,0]->${data[0].toFixed(3)}, [0,1]->${data[1].toFixed(3)}, [1,0]->${data[2].toFixed(3)}, [1,1]->${data[3].toFixed(3)}`);

		assert.approximately(data[0], 0, 0.15);
		assert.approximately(data[1], 1, 0.15);
		assert.approximately(data[2], 1, 0.15);
		assert.approximately(data[3], 0, 0.15);

		tf.dispose([w1, b1, w2, b2, xs, ys, results]);
	});
});
