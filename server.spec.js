const {assert} = require('chai');
const Logger = require('log-ng');
const path = require('node:path');
const generateData = require('./datagen.js');
const Matrix = require('./Matrix.js');
const Perceptron = require('./Perceptron.js');

const logger = new Logger(path.basename(__filename));

describe('Matrix', function(){
	it('should support 2D indexing and mutation', function(){
		const A = Matrix(2, 3);
		assert.equal(A[0][0], 0);
		assert.equal(A[1][2], 0);

		A[0][1] = 5;
		A[1][0] = -3;
		assert.equal(A[0][1], 5);
		assert.equal(A[1][0], -3);
	});

	it('should add two matrices', function(){
		const A = Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = Matrix(2, 2, new Float64Array([5, 6, 7, 8]));
		const C = A.add(B);

		assert.deepEqual(C.data, new Float64Array([6, 8, 10, 12]));
	});

	it('should multiply two matrices', function(){
		const A = Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
		const B = Matrix(3, 2, new Float64Array([7, 8, 9, 10, 11, 12]));
		const C = A.multiply(B);

		assert.deepEqual(C.data, new Float64Array([58, 64, 139, 154]));
	});

	it('should scale a matrix by a scalar', function(){
		const A = Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = A.scalar(2);

		assert.deepEqual(B.data, new Float64Array([2, 4, 6, 8]));
	});

	it('should transpose a matrix', function(){
		const A = Matrix(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
		const B = A.transpose();

		assert.deepEqual(B.data, new Float64Array([1, 4, 2, 5, 3, 6]));
		assert.deepEqual(B.dimensions, [3, 2]);
	});

	it('should create an identity matrix', function(){
		const I = Matrix.identity(3);

		const expected = new Float64Array([
			1, 0, 0,
			0, 1, 0,
			0, 0, 1
		]);
		assert.deepEqual(I.data, expected);
	});

	it('should compute the determinant', function(){
		const A = Matrix(2, 2, new Float64Array([1, 2, 3, 4]));
		assert.equal(A.determinant(), -2);

		const B = Matrix(3, 3, new Float64Array([6, 1, 1, 4, -2, 5, 2, 8, 7]));
		assert.equal(B.determinant(), -306);
	});

	it('should compute the inverse of a matrix', function(){
		// Test 2x2 matrix
		const A = Matrix(2, 2, new Float64Array([4, 7, 2, 6]));
		const AInv = A.inverse();
		const I2 = A.multiply(AInv);

		// Check that A * A^-1 ≈ I (within floating point tolerance)
		assert.approximately(I2[0][0], 1, 1e-10);
		assert.approximately(I2[0][1], 0, 1e-10);
		assert.approximately(I2[1][0], 0, 1e-10);
		assert.approximately(I2[1][1], 1, 1e-10);

		// Test 3x3 matrix
		const B = Matrix(3, 3, new Float64Array([
			3, 0, 2,
			2, 0, -2,
			0, 1, 1
		]));
		const BInv = B.inverse();
		const I3 = B.multiply(BInv);

		// Verify B * B^-1 ≈ I
		for(let i = 0; i < 3; ++i){
			for(let j = 0; j < 3; ++j){
				const expected = i === j ? 1 : 0;
				assert.approximately(I3[i][j], expected, 1e-10);
			}
		}

		// Test 1x1 matrix
		const C = Matrix(1, 1, new Float64Array([5]));
		const CInv = C.inverse();
		assert.approximately(CInv[0][0], 0.2, 1e-10);

		// Test that singular matrix throws error
		const singular = Matrix(2, 2, new Float64Array([1, 2, 2, 4]));
		assert.throws(() => singular.inverse(), /singular/);

		// Test that non-square matrix throws error
		const nonSquare = Matrix(2, 3);
		assert.throws(() => nonSquare.inverse(), /only defined for square/);
	});

	it('should throw on dimension mismatch for add', function(){
		const A = Matrix(2, 2);
		const B = Matrix(3, 2);

		assert.throws(() => A.add(B), /dimension mismatch/);
	});

	it('should throw on dimension mismatch for multiply', function(){
		const A = Matrix(2, 2);
		const B = Matrix(3, 3);

		assert.throws(() => A.multiply(B), /dimension mismatch/);
	});

	it.skip('should measure large matrix multiplication time', function(){
		this.timeout(10000);

		const size = 500;
		const A = Matrix(size, size, new Float64Array(size * size).fill(1));
		const B = Matrix(size, size, new Float64Array(size * size).fill(1));

		const start = performance.now();
		const C = A.multiply(B);
		const end = performance.now();

		const elapsedMs = end - start;
		logger.info(`Multiplying two ${size}x${size} matrices took ${elapsedMs.toFixed(2)} ms`);

		assert.equal(C.dimensions[0], size);
		assert.equal(C.dimensions[1], size);
		assert.equal(C[0][0], size);
	});
});

describe('Perceptron', function(){
	it('should initialize weights with length 3', function(){
		const p = new Perceptron();
		assert.equal(p.weights().length, 3);
		logger.debug('Perceptron created without crash.');
	});

	it('should forward propagate correctly with identity activation', function(){
		const p = new Perceptron();
		p.weights(0.5, 0.5, 0.1);
		p.activation(x => x);
		const y = p.forward(1, 1);
		// sum = 1*0.5 + 1*0.5 + 1*0.1 = 1.1
		logger.debug(`Forward output: ${y}`);
		assert.strictEqual(y, 1.1);
	});

	it('should compute backward gradients correctly for identity/MSE', function(){
		const p = new Perceptron();
		p.weights(0.5, 0.5, 0.1);
		p.activation(x => x);

		const yPred = p.forward(1, 1);

		const yTarget = 2;
		p.backward(yTarget);

		// Expected gradients: dL/dw_i = (yPred - yTarget) * x_i
		// const expectedGradients = [(1.1 - 2) * 1, (1.1 - 2) * 1, (1.1 - 2) * 1]; // [-0.9, -0.9, -0.9]

		// Gradients are private, so we can simulate by doing a manual update and checking effect
		p.update(0.1);
		const newY = p.forward(1, 1);
		logger.debug(`Backward/update moved output from ${yPred} to ${newY}`);
		assert(Math.abs(newY - yTarget) < Math.abs(yPred - yTarget));
	});

	it('should allow setting activation function', function(){
		const p = new Perceptron();
		p.activation(x => x * 2);
		const y = p.forward(1, 1);
		logger.debug(`Custom activation output: ${y}`);
		assert.strictEqual(typeof y, 'number');
	});

	it('should allow setting weights explicitly', function(){
		const p = new Perceptron();
		p.weights(0.1, 0.2, 0.3);
		const y = p.forward(0, 0);
		assert.strictEqual(y, 0.3);
	});
});

describe('Perceptron training + inference', function(){
	it('should reduce error on test data after training', function(){
		this.timeout(5000);
		const { training, test } = generateData();

		const p = new Perceptron();

		const learningRate = 0.01;
		const epochs = 1e4;

		function mse(data){
			let sum = 0;
			for(const [x1, x2, yTarget] of data){
				const yPred = p.forward(x1, x2);
				const err = yPred - yTarget;
				sum += err * err;
			}
			return sum / data.length;
		}

		const initialError = mse(test);

		for(let epoch = 0; epoch < epochs; epoch++){
			for(const [x1, x2, yTarget] of training){
				p.forward(x1, x2);
				p.backward(yTarget);
			}
			p.update(learningRate);
		}

		const finalError = mse(test);

		logger.info(`Initial MSE: ${initialError.toFixed(4)}, Final MSE: ${finalError.toFixed(4)}`);
		assert(finalError < initialError, `Expected final error < initial error (${finalError} >= ${initialError})`);
	});

	it('should use trained weights in a new perceptron instance', function(){
		this.timeout(5000);
		let { training, test } = generateData();
		const p1 = new Perceptron();

		const learningRate = 0.1;
		const epochs = 1e4;

		function mse(p, data){
			let sum = 0;
			for(const [x1, x2, yTarget] of data){
				const yPred = p.forward(x1, x2);
				const err = yPred - yTarget;
				sum += err * err;
			}
			return sum / data.length;
		}

		for(let epoch = 0; epoch < epochs; epoch++){
			for(const [x1, x2, yTarget] of training){
				p1.forward(x1, x2);
				p1.backward(yTarget);
				p1.update(learningRate);
			}
		}

		const trainedError = mse(p1, test);
		const trainedWeights = p1.weights();

		const p2 = new Perceptron();
		p2.weights(...trainedWeights);

		({ training, test } = generateData());
		const error = mse(p2, test);

		logger.info(`Transferred model test MSE: ${error.toFixed(4)}`);
		assert(error <= trainedError, `MSE too high: ${error} > ${trainedError}`);
	});
});

