const {assert} = require('chai');
const Logger = require('log-ng');
const path = require('node:path');
const {
	generate2x1Data,
	generate3x2Data,
} = require('./datagen.js');
const {DualNumber, DualMatrix} = require('./Dual.js');
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

	it('should provide a fancy string representation via toString', function(){
		const A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
		const strA = A.toString();
		logger.info(`Fancy 3x3 matrix:\n${strA}`);
		assert.ok(strA.includes('┌ 1.00  2.00  3.00 ┐'));
		assert.ok(strA.includes('│ 4.00  5.00  6.00 │'));
		assert.ok(strA.includes('└ 7.00  8.00  9.00 ┘'));

		const B = Matrix(1, 3, [1, 2, 3]);
		const strB = B.toString();
		logger.info(`Fancy 1x3 matrix:\n${strB}`);
		assert.equal(strB, '[ 1.00  2.00  3.00 ]');
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

	it('should handle 1x1 matrix arithmetic', function(){
		const A = Matrix(1, 1, [5]);
		const B = Matrix(1, 1, [10]);
		assert.equal(A.add(B)[0][0], 15);
		assert.equal(A.multiply(B)[0][0], 50);
	});

	it('should return 0 for determinant of a singular matrix', function(){
		const A = Matrix(2, 2, [1, 2, 2, 4]);
		assert.equal(A.determinant(), 0);
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

const testScale = 100;
describe('DualNumber', function(){
	it('should perform addition correctly', function(){
		for(let a = -testScale; a <= testScale; ++a){
			for(let b = -testScale; b <= testScale; b += 10){
				const dn1 = DualNumber(a, 1);
				const dn2 = DualNumber(b, 0);
				const dn3 = dn1.add(dn2);
				const dn4 = Math.ceil(Math.random() * testScale);

				assert.equal(dn3.real, a + b);
				assert.equal(dn3.dual, 1);
				assert.equal(dn1.add(dn2).add(dn4).real, a + b + dn4);
			}
		}
	});

	it('should perform subtraction correctly', function(){
		for(let a = -testScale; a <= testScale; ++a){
			for(let b = -testScale; b <= testScale; b += 10){
				const dn1 = DualNumber(a, 1);
				const dn2 = DualNumber(b, 0);
				const dn3 = dn1.sub(dn2);
				const dn4 = Math.ceil(Math.random() * testScale);

				assert.equal(dn3.real, a - b);
				assert.equal(dn3.dual, 1);
				assert.equal(dn1.sub(dn2).sub(dn4).real, a - b - dn4);
			}
		}
	});

	it('should perform multiplication correctly', function(){
		for(let a = -testScale; a <= testScale; a += 5){
			for(let b = -testScale; b <= testScale; b += 10){
				const dn1 = DualNumber(a, 1);
				const dn2 = DualNumber(b, 0);
				const dn3 = dn1.mul(dn2);
				const dn4 = Math.ceil(Math.random() * testScale);

				assert.equal(dn3.real, a * b);
				assert.equal(dn3.dual, b);
				assert.equal(dn1.mul(dn2).mul(dn4).real, a * b * dn4);
			}
		}
	});

	it('should perform division correctly', function(){
		for(let a = -testScale; a <= testScale; a += 5){
			for(let b = -testScale; b <= testScale; b += 10){
				if(b === 0) continue;
				const dn1 = DualNumber(a, 1);
				const dn2 = DualNumber(b, 0);
				const dn3 = dn1.div(dn2);
				const dn4 = Math.ceil(Math.random() * testScale);

				assert.equal(dn3.real, a / b);
				assert.equal(dn3.dual, 1 / b);
				assert.equal(dn1.div(dn2).div(dn4).real, (a / b) / dn4);
			}
		}
	});

	it('should compute derivatives via dual part', function(){
		for(let x = -testScale; x <= testScale; x += 5){
			const dn = DualNumber(x, 1);

			// f(x) = x^2
			const f1 = dn.mul(dn);
			assert.equal(f1.real, x * x);
			assert.equal(f1.dual, 2 * x);

			// f(x) = x^3 + 2x + 1
			const f2 = dn.mul(dn).mul(dn).add(dn.mul(2)).add(1);
			assert.equal(f2.real, x * x * x + 2 * x + 1);
			assert.equal(f2.dual, 3 * x * x + 2);
		}
	});

	it('should compute gradients via backward propagation', function(){
		for(let x = -testScale; x <= testScale; x += 5){
			const dn = DualNumber(x, 0);

			// f(x) = x^2
			const f = dn.mul(dn);

			f.backprop();

			assert.equal(f.real, x * x);
			assert.equal(dn.grad, 2 * x);
		}
	});

	it('should handle the "Diamond Problem" (reusing a variable in the graph)', function(){
		const x = DualNumber(1);
		const a = x.mul(2);
		const b = x.mul(3);
		const y = a.add(b); // y = 2x + 3x = 5x

		y.backprop();

		assert.equal(x.grad, 5);
	});
});

describe('DualMatrix', function(){
	it('should perform matrix addition correctly (forward and backward)', function(){
		const A = DualMatrix(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = DualMatrix(2, 2, new Float64Array([5, 6, 7, 8]));
		const C = A.add(B);

		assert.deepEqual(C.real.data, new Float64Array([6, 8, 10, 12]));

		C.backprop();

		assert.deepEqual(A.grad.data, new Float64Array([1, 1, 1, 1]));
		assert.deepEqual(B.grad.data, new Float64Array([1, 1, 1, 1]));
	});

	it('should perform matrix multiplication correctly (forward and backward)', function(){
		// A (2x2) * B (2x1) = C (2x1)
		// [1 2]   [5]   [1*5 + 2*6]   [17]
		// [3 4] * [6] = [3*5 + 4*6] = [39]
		const A = DualMatrix(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = DualMatrix(2, 1, new Float64Array([5, 6]));
		const C = A.multiply(B);

		assert.deepEqual(C.real.data, new Float64Array([17, 39]));

		// Backprop
		// gradA = gradC * B^T = [1] * [5 6] = [5 6]
		//                       [1]           [5 6]
		// gradB = A^T * gradC = [1 3] * [1] = [4]
		//                       [2 4]   [1]   [6]
		C.backprop();

		assert.deepEqual(A.grad.data, new Float64Array([5, 6, 5, 6]));
		assert.deepEqual(B.grad.data, new Float64Array([4, 6]));
	});

	it('should handle reused matrices in a graph (Diamond Problem)', function(){
		const A = DualMatrix(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = A.add(A);

		B.backprop();

		assert.deepEqual(A.grad.data, new Float64Array([2, 2, 2, 2]));
	});

	it('should support element-wise map (e.g., for activations)', function(){
		const A = DualMatrix(1, 2, new Float64Array([1, -2]));
		const B = A.map(x => x.real > 0 ? x : DualNumber(0));

		assert.deepEqual(B.real.data, new Float64Array([1, 0]));

		B.backprop();

		assert.deepEqual(A.grad.data, new Float64Array([1, 0]));
	});

	it('should provide a string representation via toString', function(){
		const A = DualMatrix(1, 1, [5]);
		const str = A.toString();
		logger.info(`DualMatrix string representation: ${str}`);
		assert.ok(str.includes('DualMatrix(1x1)'));
		assert.ok(str.includes('Real:'));
		assert.ok(str.includes('5'));
	});

	it('should correctly reset gradients in a mixed graph (Matrix + Scalar)', function(){
		const A = DualMatrix(1, 2, [1, 2]);
		const B = DualMatrix(2, 1, [3, 4]);
		const C = A.multiply(B);

		const scalarOut = C[0][0].add(10);

		scalarOut.backprop();

		assert.notEqual(A.grad.data[0], 0);
		assert.notEqual(scalarOut.grad, 0);

		scalarOut.zeroGrads();

		assert.equal(A.grad.data[0], 0);
		assert.equal(A.grad.data[1], 0);
		assert.equal(scalarOut.grad, 0);
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
		p.activation = x => x;
		const y = p.forward(1, 1)[0][0];
		// sum = 1*0.5 + 1*0.5 + 1*0.1 = 1.1
		logger.debug(`Forward output: ${y}`);
		assert.strictEqual(y.real, 1.1);
	});

	it('should compute backward gradients correctly for identity/MSE', function(){
		const p = new Perceptron();
		p.actiation = Perceptron.IDENTITY;
		p.weights(0.5, 0.5, 0.1);
		p.activation = x => x;

		const yPred = p.forward(1, 1)[0][0];

		const yTarget = 2;
		p.backward(yTarget);

		// Gradients are private, so we can simulate by doing a manual update and checking effect
		p.update(0.1);
		const newY = p.forward(1, 1)[0][0];
		logger.debug(`Backward/update moved output from ${yPred} to ${newY}`);
		assert(Math.abs(newY.real - yTarget) < Math.abs(yPred.real - yTarget));
	});

	it('should allow setting activation function', function(){
		const p = new Perceptron();
		p.activation = x => x.mul(2);
		const y = p.forward(1, 1)[0][0];
		logger.debug(`Custom activation output: ${y}`);
		assert.ok(y instanceof DualNumber);
	});

	it('should allow setting loss function', function(){
		const p = new Perceptron();
		p.loss = (yPred, yTarget) => {
			const diff = yPred.sub(DualNumber(yTarget, 0));
			return diff.mul(diff).mul(DualNumber(0.5, 0));
		};
		const y = p.forward(1, 1)[0][0];
		const loss = p.loss(y, 3);
		logger.debug(`Custom loss output: ${loss}`);
		assert.ok(loss instanceof DualNumber);
	});

	it('should allow manually setting weights', function(){
		const p = new Perceptron();
		p.activation = Perceptron.RELU;

		p.weights(0.1, 0.2, 0.3);
		const y = p.forward(0, 0)[0][0];
		assert.strictEqual(y.real, 0.3);
	});

	it('should correctly calculate number of weights', function(){
		const numInputs = 3;
		const numOutputs = 2;

		const p = new Perceptron(numInputs, numOutputs);
		assert.equal(p.weights().length, (numInputs + 1) * numOutputs);
	});

	it('should handle arbitrary number of inputs', function(){
		const numInputs = 5;
		const p = new Perceptron(numInputs);
		p.activation = Perceptron.RELU;

		const inputs = [1, 2, 3, 4, 5];
		const weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // including bias
		p.weights(...weights);

		const expectedOutput = inputs.reduce((sum, x, i) => sum + x * weights[i], weights[weights.length - 1]);

		const y = p.forward(...inputs)[0][0];
		logger.debug(`Output with ${numInputs} inputs: ${y}`);
		assert.approximately(y.real, expectedOutput, 1e-10);
	});

	it('should handle arbitrary number of outputs', function(){
		const numInputs = 2;
		const numOutputs = 3;
		const p = new Perceptron(numInputs, numOutputs);
		p.activation = Perceptron.IDENTITY;

		const inputs = [1, 2];
		const weights = [
			0.1, 0.2, 0.3,
			0.4, 0.5, 0.6,
			0.7, 0.8, 0.9
		];
		p.weights(...weights);

		const y = p.forward(...inputs);
		assert.equal(y.dimensions[0], numOutputs);

		assert.approximately(y[0][0].real, 0.1 * 1 + 0.2 * 2 + 0.3, 1e-10);
		assert.approximately(y[1][0].real, 0.4 * 1 + 0.5 * 2 + 0.6, 1e-10);
		assert.approximately(y[2][0].real, 0.7 * 1 + 0.8 * 2 + 0.9, 1e-10);
	});

	it('should allow switching between built-in loss functions', function(){
		const p = new Perceptron(2, 1);
		p.activation = Perceptron.IDENTITY;
		p.forward(1, 1);

		// Switch to MAE
		p.loss = Perceptron.MAE;
		p.backward([2]); // target 2, pred ~random

		// Confirm it runs without crash
		p.update(0.1);

		// Switch to Huber
		p.loss = Perceptron.HUBER;
		p.forward(1, 1);
		p.backward([2]);
		p.update(0.1);
	});
});

describe('Perceptron training + inference', function(){
	before(function(){
		this.numSamples = 1e3;
		this.learningRate = 0.01;
		this.epochs = 50;
	});

	it('should verify that generated data is normalized to [0, 1]', function(){
		const noise = 0.05;
		const { training } = generate3x2Data(100, 1.0, true, noise);
		for(const [x1, x2, x3, [y1, y2]] of training){
			assert.ok(x1 >= 0 && x1 <= 1, `x1 out of range: ${x1}`);
			assert.ok(x2 >= 0 && x2 <= 1, `x2 out of range: ${x2}`);
			assert.ok(x3 >= 0 && x3 <= 1, `x3 out of range: ${x3}`);
			// Allow small overflow due to noise
			assert.ok(y1 >= -noise && y1 <= 1 + noise, `y1 out of range: ${y1}`);
			assert.ok(y2 >= -noise && y2 <= 1 + noise, `y2 out of range: ${y2}`);
		}
	});

	it('should reduce error on test data after training', function(){
		this.timeout(7e3);
		const { training, test } = generate2x1Data(this.numSamples);

		const p = new Perceptron();
		p.activation = Perceptron.IDENTITY;

		function mse(data){
			let sum = 0;
			for(const [x1, x2, yTarget] of data){
				const yPred = p.forward(x1, x2)[0][0];
				const err = yPred.real - yTarget;
				sum += err * err;
			}
			return sum / data.length;
		}

		const initialError = mse(test);

		for(let epoch = 0; epoch < this.epochs; epoch++){
			for(const [x1, x2, yTarget] of training){
				p.forward(x1, x2);
				p.backward(yTarget);
				p.update(this.learningRate);
			}
		}

		const finalError = mse(test);

		logger.info(`Initial MSE: ${initialError.toFixed(4)}, Final MSE: ${finalError.toFixed(4)}`);
		assert(finalError < initialError, `Expected final error < initial error (${finalError} >= ${initialError})`);
	});

	it('should learn both Grade and Fatigue simultaneously', function(){
		this.timeout(7e3);
		const { training, test } = generate3x2Data(this.numSamples);
		const p = new Perceptron(3, 2);
		p.activation = Perceptron.IDENTITY;

		function mse(data){
			let sum = 0;
			for(const [x1, x2, x3, yTargets] of data){
				const yPred = p.forward(x1, x2, x3);
				const err1 = yPred[0][0].real - yTargets[0];
				const err2 = yPred[1][0].real - yTargets[1];
				sum += (err1 * err1 + err2 * err2) / 2;
			}
			return sum / data.length;
		}

		const initialError = mse(test);

		for(let epoch = 0; epoch < this.epochs; epoch++){
			for(const [x1, x2, x3, yTargets] of training){
				p.forward(x1, x2, x3);
				p.backward(yTargets);
				p.update(this.learningRate);
			}
		}

		const finalError = mse(test);

		logger.info(`Multi-output (3x2) Initial MSE: ${initialError.toFixed(4)}, Final MSE: ${finalError.toFixed(4)}`);
		assert(finalError < initialError, `Expected final error < initial error (${finalError} >= ${initialError})`);
	});

	it('should use trained weights in a new perceptron instance', function(){
		this.timeout(7e3);
		let { training, test } = generate2x1Data(this.numSamples);
		const p1 = new Perceptron();

		function mse(p, data){
			let sum = 0;
			for(const [x1, x2, yTarget] of data){
				const yPred = p.forward(x1, x2)[0][0];
				const err = yPred.real - yTarget;
				sum += err * err;
			}
			return sum / data.length;
		}

		for(let epoch = 0; epoch < this.epochs; epoch++){
			for(const [x1, x2, yTarget] of training){
				p1.forward(x1, x2);
				p1.backward(yTarget);
				p1.update(this.learningRate);
			}
		}

		const trainedError = mse(p1, test);
		const trainedWeights = p1.weights();
		logger.info(`Trained model test MSE: ${trainedError.toFixed(4)}`);

		const p2 = new Perceptron();
		p2.weights(...trainedWeights);

		({ training, test } = generate2x1Data());
		const error = mse(p2, test);

		logger.info(`Transferred model test MSE: ${error.toFixed(4)}`);
		assert((Math.abs(error - trainedError) / trainedError) < 0.15, `Expected transferred model error to be close to trained model error (${error} vs ${trainedError})`);
	});
});

