const {assert} = require('chai');
const Logger = require('log-ng');
const path = require('node:path');
const {
	generate2x1Data,
	generate3x2Data,
} = require('./datagen.js');
const {
	DualArray,
	DualMatrix,
	DualNumber
} = require('./Dual.js');
const {Matrix, Array2D} = require('./Matrix.js');
const { Perceptron, Layer } = require('./Perceptron.js');

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

	it('should throw on dimension mismatch for add (no broadcasting)', function(){
		const A = Matrix(2, 2, [1, 2, 3, 4]);
		const B = Matrix(2, 1, [10, 20]);
		assert.throws(() => A.add(B), /dimension mismatch/);
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

	it('should provide a formatted string representation via toString', function(){
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
		this.timeout(10e3);

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

describe('Array2D', function(){
	it('should support 2D indexing and mutation', function(){
		const A = Array2D(2, 3);
		assert.equal(A[0][0], 0);
		assert.equal(A[1][2], 0);

		A[0][1] = 5;
		A[1][0] = -3;
		assert.equal(A[0][1], 5);
		assert.equal(A[1][0], -3);
	});

	it('should support broadcasting in addition', function(){
		const A = Array2D(2, 3, [1, 2, 3, 4, 5, 6]);

		// Column vector broadcasting (2x1)
		const col = Array2D(2, 1, [10, 20]);
		const B = A.add(col);
		assert.deepEqual(B.data, new Float64Array([11, 12, 13, 24, 25, 26]));

		// Row vector broadcasting (1x3)
		const row = Array2D(1, 3, [100, 200, 300]);
		const C = A.add(row);
		assert.deepEqual(C.data, new Float64Array([101, 202, 303, 104, 205, 306]));
	});

	it('should support broadcasting in subtraction', function(){
		const A = Array2D(2, 2, [10, 20, 30, 40]);
		const col = Array2D(2, 1, [1, 2]);
		const B = A.sub(col);
		assert.deepEqual(B.data, new Float64Array([9, 19, 28, 38]));
	});

	it('should support broadcasting in min and max', function(){
		const A = Array2D(2, 2, [10, 5, 2, 20]);
		const threshold = Array2D(2, 1, [8, 15]);

		const high = A.max(threshold);
		assert.deepEqual(high.data, new Float64Array([10, 8, 15, 20]));

		const low = A.min(threshold);
		assert.deepEqual(low.data, new Float64Array([8, 5, 2, 15]));
	});

	it('should multiply two arrays', function(){
		const A = Array2D(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
		const B = Array2D(3, 2, new Float64Array([7, 8, 9, 10, 11, 12]));
		const C = A.multiply(B);

		assert.deepEqual(C.data, new Float64Array([58, 64, 139, 154]));
	});

	it('should scale an array by a scalar', function(){
		const A = Array2D(2, 2, new Float64Array([1, 2, 3, 4]));
		const B = A.scalar(2);

		assert.deepEqual(B.data, new Float64Array([2, 4, 6, 8]));
	});

	it('should transpose an array', function(){
		const A = Array2D(2, 3, new Float64Array([1, 2, 3, 4, 5, 6]));
		const B = A.transpose();

		assert.deepEqual(B.data, new Float64Array([1, 4, 2, 5, 3, 6]));
		assert.deepEqual(B.dimensions, [3, 2]);
	});

	it('should support sum reduction', function(){
		const A = Array2D(2, 3, [1, 2, 3, 4, 5, 6]);

		// Column sum (axis 0) -> 1x3 result
		const colSum = A.sum(0);
		assert.deepEqual(colSum.data, new Float64Array([5, 7, 9]));

		// Row sum (axis 1) -> 2x1 result
		const rowSum = A.sum(1);
		assert.deepEqual(rowSum.data, new Float64Array([6, 15]));

		// Global sum -> 1x1 result
		const totalSum = A.sum();
		assert.deepEqual(totalSum.data, new Float64Array([21]));
	});

	it('should provide a formatted string representation via toString', function(){
		const A = Array2D(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
		const strA = A.toString();
		assert.ok(strA.includes('┌ 1.00  2.00  3.00 ┐'));
		assert.ok(strA.includes('│ 4.00  5.00  6.00 │'));
		assert.ok(strA.includes('└ 7.00  8.00  9.00 ┘'));
	});

	it('should throw on dimension mismatch for add (no valid broadcast)', function(){
		const A = Array2D(2, 2);
		const B = Array2D(3, 2);

		assert.throws(() => A.add(B), /dimension mismatch/);
	});

	it('should throw on dimension mismatch for multiply', function(){
		const A = Array2D(2, 2);
		const B = Array2D(3, 3);

		assert.throws(() => A.multiply(B), /dimension mismatch/);
	});

	it('should handle 1x1 array arithmetic', function(){
		const A = Array2D(1, 1, [5]);
		const B = Array2D(1, 1, [10]);
		assert.equal(A.add(B)[0][0], 15);
		assert.equal(A.multiply(B)[0][0], 50);
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

	it('should propagate dual property (Forward-Mode AD) through DualMatrix', function(){
		const A = DualMatrix(2, 2, [1, 2, 3, 4], [1, 0, 0, 1]);
		const B = DualMatrix(2, 2, [5, 6, 7, 8], [0, 1, 1, 0]);
		const C = A.add(B);
		assert.deepEqual(C.dual.data, new Float64Array([1, 1, 1, 1]));
		const D = A.multiply(B);
		assert.deepEqual(D.dual.data, new Float64Array([7, 7, 11, 11]));
	});
});

describe('DualArray', function(){
	it('should support broadcasting in addition (forward and backward)', function(){
		const A = DualArray(2, 3, [1, 2, 3, 4, 5, 6]);
		const col = DualArray(2, 1, [10, 20]);

		const sum = A.add(col);
		assert.deepEqual(sum.real.data, new Float64Array([11, 12, 13, 24, 25, 26]));

		sum.backprop();

		// Gradient for A should be all ones
		assert.deepEqual(A.grad.data, new Float64Array([1, 1, 1, 1, 1, 1]));

		// Gradient for col (2x1) should be the sum of gradients across each row of the output
		assert.deepEqual(col.grad.data, new Float64Array([3, 3]));
	});

	it('should support broadcasting in subtraction (forward and backward)', function(){
		const A = DualArray(2, 2, [10, 20, 30, 40]);
		const col = DualArray(2, 1, [1, 2]);

		const diff = A.sub(col);
		assert.deepEqual(diff.real.data, new Float64Array([9, 19, 28, 38]));

		diff.backprop();

		// Gradient for A should be all ones
		assert.deepEqual(A.grad.data, new Float64Array([1, 1, 1, 1]));

		// Gradient for col (2x1) should be -sum of gradients across each row
		assert.deepEqual(col.grad.data, new Float64Array([-2, -2]));
	});

	it('should support broadcasting in min and max (forward and backward)', function(){
		const A = DualArray(2, 2, [10, 5, 2, 20]);
		const threshold = DualArray(2, 1, [8, 15]);

		// Max test
		const high = A.max(threshold);
		assert.deepEqual(high.real.data, new Float64Array([10, 8, 15, 20]));

		high.backprop();
		// Indices where A > threshold: [0, 0] (10 > 8), [1, 1] (20 > 15)
		// Indices where threshold > A: [0, 1] (8 > 5), [1, 0] (15 > 2)
		assert.deepEqual(A.grad.data, new Float64Array([1, 0, 0, 1]));
		assert.deepEqual(threshold.grad.data, new Float64Array([1, 1]));

		A.zeroGrads();
		threshold.zeroGrads();

		// Min test
		const low = A.min(threshold);
		assert.deepEqual(low.real.data, new Float64Array([8, 5, 2, 15]));

		low.backprop();
		// Indices where A < threshold: [0, 1] (5 < 8), [1, 0] (2 < 15)
		// Indices where threshold < A: [0, 0] (8 < 10), [1, 1] (15 < 20)
		assert.deepEqual(A.grad.data, new Float64Array([0, 1, 1, 0]));
		assert.deepEqual(threshold.grad.data, new Float64Array([1, 1]));
	});

	it('should handle ties in min and max by splitting gradients', function(){
		const A = DualArray(1, 2, [10, 10]);
		const B = DualArray(1, 2, [10, 5]);

		const res = A.max(B);
		res.backprop();

		// At index 0, A[0]==B[0]==10, so gradient should be 0.5 each
		assert.equal(A.grad.data[0], 0.5);
		assert.equal(B.grad.data[0], 0.5);

		// At index 1, A[1]==10 > B[1]==5, so A gets full gradient
		assert.equal(A.grad.data[1], 1.0);
		assert.equal(B.grad.data[1], 0.0);
	});

	it('should support sum reduction (forward and backward)', function(){
		const A = DualArray(2, 3, [1, 2, 3, 4, 5, 6]);

		// Row sum (axis 1) -> 2x1 result
		const rowSum = A.sum(1);
		assert.deepEqual(rowSum.real.data, new Float64Array([6, 15]));

		rowSum.backprop();
		assert.deepEqual(A.grad.data, new Float64Array([1, 1, 1, 1, 1, 1]));
	});

	it('should provide a string representation via toString', function(){
		const A = DualArray(1, 1, [5]);
		const str = A.toString();
		assert.ok(str.includes('DualArray(1x1)'));
		assert.ok(str.includes('Real:'));
		assert.ok(str.includes('5'));
	});
});

describe('Layer', function(){
	it('should initialize with correct weight and bias shapes', function(){
		const l = new Layer(3, 2);
		assert.equal(l.weights().length, 6);
		assert.equal(l.bias().length, 2);
	});

	it('should forward propagate correctly (W*X + B)', function(){
		const l = new Layer(2, 1);
		l.activation = Perceptron.IDENTITY;
		l.weights(0.5, 0.5);
		l.bias(0.1);

		const input = DualMatrix(2, 1, [1, 1]);
		const output = l.forward(input);

		assert.approximately(output[0][0].real, 1.1, 1e-10);
	});

	it('should update weights and biases based on gradients', function(){
		const l = new Layer(1, 1);
		l.activation = Perceptron.IDENTITY;
		l.weights(1.0);
		l.bias(0.0);

		const input = DualMatrix(1, 1, [1.0]);
		const output = l.forward(input);

		const diff = output[0][0].sub(2.0);
		const loss = diff.mul(diff).mul(0.5);
		loss.backprop();

		l.update(0.1);

		assert.approximately(l.weights()[0], 1.1, 1e-10);
		assert.approximately(l.bias()[0], 0.1, 1e-10);
	});
});

describe('Perceptron', function(){
	it('should initialize weights with no hidden layer', function(){
		const p = new Perceptron([2, 1]);
		assert.equal(p.weights().length, 3);
	});

	it('should initialize with correct total weight and bias count', function(){
		// [2 inputs, 3 hidden, 1 output]
		const p = new Perceptron([2, 3, 1]);
		// Layer 1: 2 inputs -> 3 outputs. Weights: 2*3=6, Biases: 3. Total: 9
		// Layer 2: 3 inputs -> 1 output.  Weights: 3*1=3, Biases: 1. Total: 4
		// Global Total: 13
		assert.equal(p.weights().length, 13);
	});

	it('should allow setting activation function', function(){
		const p = new Perceptron([2, 1]);
		p.activation = x => x.mul(2);
		const y = p.forward(1, 1)[0][0];
		logger.debug(`Custom activation output: ${y}`);
		assert.ok(y instanceof DualNumber);
	});

	it('should update all layers when activation property is set', function(){
		const p = new Perceptron([2, 2, 1]);

		p.activation = Perceptron.RELU;

		// We can't easily check private layer activation property,
		// but we can check the getter which pulls from layers[0]
		assert.strictEqual(p.activation, Perceptron.RELU);

		p.activation = Perceptron.IDENTITY;
		assert.strictEqual(p.activation, Perceptron.IDENTITY);
	});

	it('should allow setting loss function', function(){
		const p = new Perceptron([2, 1]);
		p.loss = (yPred, yTarget) => {
			const diff = yPred.sub(DualNumber(yTarget, 0));
			return diff.mul(diff).mul(DualNumber(0.5, 0));
		};
		const y = p.forward(1, 1)[0][0];
		const loss = p.loss(y, 3);
		logger.debug(`Custom loss output: ${loss}`);
		assert.ok(loss instanceof DualNumber);
	});

	it('should allow switching between built-in loss functions', function(){
		const p = new Perceptron([2, 1]);
		p.activation = Perceptron.IDENTITY;
		p.forward(1, 1);

		p.loss = Perceptron.MAE;
		p.backward([2]);

		p.update(0.1);

		p.loss = Perceptron.HUBER;
		p.forward(1, 1);
		p.backward([2]);
		p.update(0.1);
	});

	it('should allow manually setting weights', function(){
		const p = new Perceptron([2, 1]);
		p.activation = Perceptron.RELU;

		p.weights(0.1, 0.2, 0.3);
		const y = p.forward(0, 0)[0][0];
		assert.strictEqual(y.real, 0.3);
	});

	it('should correctly calculate number of weights', function(){
		const numInputs = 3;
		const numOutputs = 2;

		const p = new Perceptron([numInputs, numOutputs]);
		assert.equal(p.weights().length, (numInputs + 1) * numOutputs);
	});

	it('should handle arbitrary number of inputs', function(){
		const numInputs = 5;
		const p = new Perceptron([numInputs, 1]);
		p.activation = Perceptron.RELU;

		const inputs = [1, 2, 3, 4, 5];
		const weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
		p.weights(...weights);

		const expectedOutput = inputs.reduce((sum, x, i) => sum + x * weights[i], weights[weights.length - 1]);

		const y = p.forward(...inputs)[0][0];
		logger.debug(`Output with ${numInputs} inputs: ${y}`);
		assert.approximately(y.real, expectedOutput, 1e-10);
	});

	it('should handle arbitrary number of outputs', function(){
		const numInputs = 2;
		const numOutputs = 3;
		const p = new Perceptron([numInputs, numOutputs]);
		p.activation = Perceptron.IDENTITY;

		const inputs = [1, 2];
		const weights = [
			0.1, 0.2,
			0.4, 0.5,
			0.7, 0.8,
			0.3, 0.6, 0.9
		];
		p.weights(...weights);

		const y = p.forward(...inputs);
		assert.equal(y.dimensions[0], numOutputs);

		assert.approximately(y[0][0].real, 0.1 * 1 + 0.2 * 2 + 0.3, 1e-10);
		assert.approximately(y[1][0].real, 0.4 * 1 + 0.5 * 2 + 0.6, 1e-10);
		assert.approximately(y[2][0].real, 0.7 * 1 + 0.8 * 2 + 0.9, 1e-10);
	});

	it('should forward propagate correctly with identity activation', function(){
		const p = new Perceptron([2, 1]);
		p.weights(0.5, 0.5, 0.1);
		p.activation = x => x;
		const y = p.forward(1, 1)[0][0];
		// sum = 1*0.5 + 1*0.5 + 1*0.1 = 1.1
		logger.debug(`Forward output: ${y}`);
		assert.strictEqual(y.real, 1.1);
	});

	it('should forward propagate correctly through multiple layers', function(){
		const p = new Perceptron([2, 2, 1]);
		p.activation = Perceptron.IDENTITY;

		// Layer 1 weights (2x2) and biases (2x1)
		// W1 = [[0.5, 0.5], [0.1, 0.1]], B1 = [0.1, 0.1]
		// Layer 2 weights (1x2) and biases (1x1)
		// W2 = [[0.5, 0.5]], B2 = [0.1]
		// Total weights = 4 + 2 + 2 + 1 = 9
		p.weights(
			0.5, 0.5, 0.1, 0.1, // W1
			0.1, 0.1,           // B1
			0.5, 0.5,           // W2
			0.1                 // B2
		);

		const y = p.forward(1, 1)[0][0];
		// Layer 1 hidden state:
		// h1 = 1*0.5 + 1*0.5 + 0.1 = 1.1
		// h2 = 1*0.1 + 1*0.1 + 0.1 = 0.3
		// Layer 2 output:
		// y = 1.1*0.5 + 0.3*0.5 + 0.1 = 0.55 + 0.15 + 0.1 = 0.8
		assert.approximately(y.real, 0.8, 1e-10);
	});

	it('should compute backward gradients correctly for identity/MSE', function(){
		const p = new Perceptron([2, 1]);
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

	it('should compute backward gradients through multiple layers', function(){
		const p = new Perceptron([1, 1, 1]);
		p.activation = Perceptron.IDENTITY;

		// W1 = [1.0], B1 = [0.0]
		// W2 = [1.0], B2 = [0.0]
		// Total = 1 + 1 + 1 + 1 = 4
		p.weights(1.0, 0.0, 1.0, 0.0);

		const initialY = p.forward(1.0)[0][0];
		assert.strictEqual(initialY.real, 1.0);

		const target = 2.0;
		p.backward([target]);

		const oldWeights = p.weights();

		p.update(0.1);

		const newWeights = p.weights();
		const newY = p.forward(1.0)[0][0];

		logger.debug(`MLP backward move: ${initialY.real} -> ${newY.real}`);

		for(let i = 0; i < oldWeights.length; ++i){
			assert.notEqual(oldWeights[i], newWeights[i], `Weight at index ${i} did not move`);
		}

		assert(Math.abs(newY.real - target) < Math.abs(initialY.real - target));
	});
});

describe('Activation and Loss Functions', function(){
	describe('Activations', function(){
		const testCases = [
			{ name: 'IDENTITY', act: Perceptron.IDENTITY, x: 2.0, expected: 2.0, grad: 1.0 },
			{ name: 'STEP (pos)', act: Perceptron.STEP, x: 0.5, expected: 1.0, grad: 0.0 },
			{ name: 'STEP (neg)', act: Perceptron.STEP, x: -0.5, expected: 0.0, grad: 0.0 },
			{ name: 'STEP (zero)', act: Perceptron.STEP, x: 0.0, expected: 1.0, grad: 0.0 },
			{ name: 'RELU (pos)', act: Perceptron.RELU, x: 1.0, expected: 1.0, grad: 1.0 },
			{ name: 'RELU (neg)', act: Perceptron.RELU, x: -1.0, expected: 0.0, grad: 0.0 },
			{ name: 'RELU (zero)', act: Perceptron.RELU, x: 0.0, expected: 0.0, grad: 0.0 },
			{ name: 'SIGMOID (zero)', act: Perceptron.SIGMOID, x: 0.0, expected: 0.5, grad: 0.25 },
			{
				name: 'SIGMOID (pos)',
				act: Perceptron.SIGMOID,
				x: 1.0,
				expected: 1 / (1 + Math.exp(-1)),
				grad: (1 / (1 + Math.exp(-1))) * (1 - (1 / (1 + Math.exp(-1))))
			},
			{ name: 'TANH (zero)', act: Perceptron.TANH, x: 0.0, expected: 0.0, grad: 1.0 },
			{
				name: 'TANH (pos)',
				act: Perceptron.TANH,
				x: 1.0,
				expected: Math.tanh(1),
				grad: 1 - Math.pow(Math.tanh(1), 2)
			}
		];

		testCases.forEach(({ name, act, x, expected, grad }) => {
			it(`should verify ${name} forward and backward`, function(){
				const dm = DualMatrix(1, 1, [x]);
				const out = dm.map(act);
				assert.approximately(out.real.data[0], expected, 1e-10);
				out.backprop([1]);
				assert.approximately(dm.grad.data[0], grad, 1e-10);
			});
		});

		it('should verify SOFTMAX Jacobian', function(){
			const x = DualMatrix(3, 1, [1, 2, 3]);
			const y = Perceptron.SOFTMAX(x);
			const sum = y.real.data.reduce((a, b) => a + b, 0);
			assert.approximately(sum, 1.0, 1e-10);

			const seed = [1, 0, 0];
			y.backprop(seed);

			const y0 = y.real.data[0];
			const y1 = y.real.data[1];
			const y2 = y.real.data[2];

			assert.approximately(x.grad.data[0], y0 * (1 - y0), 1e-10);
			assert.approximately(x.grad.data[1], -y0 * y1, 1e-10);
			assert.approximately(x.grad.data[2], -y0 * y2, 1e-10);
		});
	});

	describe('Loss Functions', function(){
		const yPredArr = [0.5, 0.8];
		const yTargetArr = [1.0, 0.0];

		it('should verify MSE forward and backward', function(){
			const yPred = DualMatrix(2, 1, yPredArr);
			const loss = Perceptron.MSE(yPred, yTargetArr);
			const n = yPred.dimensions[0];

			const expected = yPredArr.reduce((sum, p, i) => {
				return sum + 0.5 * Math.pow(p - yTargetArr[i], 2);
			}, 0) / n;

			assert.approximately(loss.real, expected, 1e-10);

			loss.backprop();
			// dL/dy_i = (y_i - t_i) / n
			assert.approximately(yPred.grad.data[0], (0.5 - 1.0) / n, 1e-10);
			assert.approximately(yPred.grad.data[1], (0.8 - 0.0) / n, 1e-10);
		});

		it('should verify MAE forward and backward', function(){
			const yPred = DualMatrix(2, 1, yPredArr);
			const loss = Perceptron.MAE(yPred, yTargetArr);
			const n = yPred.dimensions[0];

			const expected = yPredArr.reduce((sum, p, i) => {
				return sum + Math.abs(p - yTargetArr[i]);
			}, 0) / n;

			assert.approximately(loss.real, expected, 1e-10);

			loss.backprop();
			// dL/dy_i = sign(y_i - t_i) / n
			assert.approximately(yPred.grad.data[0], -1 / n, 1e-10);
			assert.approximately(yPred.grad.data[1], 1 / n, 1e-10);
		});

		it('should verify HUBER forward and backward', function(){
			const yPred = DualMatrix(1, 1, [0.5]);
			const yTarget = [1.0];
			const loss = Perceptron.HUBER(yPred, yTarget, 1.0);
			assert.approximately(loss.real, 0.125, 1e-10);
			loss.backprop();
			assert.approximately(yPred.grad.data[0], -0.5, 1e-10);
		});

		it('should verify CROSS_ENTROPY forward and backward', function(){
			const yPred = DualMatrix(1, 1, [0.5]);
			const yTarget = [1.0];
			const loss = Perceptron.CROSS_ENTROPY(yPred, yTarget);
			assert.approximately(loss.real, Math.log(2), 1e-10);
			loss.backprop();
			assert.approximately(yPred.grad.data[0], -2.0, 1e-10);
		});

		it('should maintain numerical stability in CROSS_ENTROPY with extreme values', function(){
			// Test exactly 0 and 1
			const yPredZero = DualMatrix(1, 1, [0]);
			const yPredOne = DualMatrix(1, 1, [1]);
			const yTarget = [1];

			const lossZero = Perceptron.CROSS_ENTROPY(yPredZero, yTarget);
			assert.ok(Number.isFinite(lossZero.real), 'Loss should be finite at yPred=0');
			lossZero.backprop();
			assert.ok(Number.isFinite(yPredZero.grad.data[0]), 'Gradient should be finite at yPred=0');

			const lossOne = Perceptron.CROSS_ENTROPY(yPredOne, yTarget);
			assert.ok(Number.isFinite(lossOne.real), 'Loss should be finite at yPred=1');
			lossOne.backprop();
			assert.ok(Number.isFinite(yPredOne.grad.data[0]), 'Gradient should be finite at yPred=1');

			// Test near epsilon boundary
			const yPredNearZero = DualMatrix(1, 1, [1e-8]);
			const lossNearZero = Perceptron.CROSS_ENTROPY(yPredNearZero, yTarget);
			assert.ok(Number.isFinite(lossNearZero.real), 'Loss should be finite near 0');
		});
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
		this.timeout(10e3);
		const { training, test } = generate2x1Data(this.numSamples);

		const p = new Perceptron([2, 1]);
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
		this.timeout(10e3);
		const { training, test } = generate3x2Data(this.numSamples);
		const p = new Perceptron([3, 2]);
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
		this.timeout(10e3);
		let { training, test } = generate2x1Data(this.numSamples);
		const p1 = new Perceptron([2, 1]);

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

		const p2 = new Perceptron([2, 1]);
		p2.weights(...trainedWeights);

		({ training, test } = generate2x1Data());
		const error = mse(p2, test);

		logger.info(`Transferred model test MSE: ${error.toFixed(4)}`);
		assert((Math.abs(error - trainedError) / trainedError) < 0.15, `Expected transferred model error to be close to trained model error (${error} vs ${trainedError})`);
	});

	it('should solve XOR with a hidden layer (MLP) but fail without one (SLP)', function(){
		this.timeout(10e3);

		const trainingData = [
			{ input: [0, 0], target: [0] },
			{ input: [0, 1], target: [1] },
			{ input: [1, 0], target: [1] },
			{ input: [1, 1], target: [0] }
		];

		// 1. SLP (No hidden layer) - Mathematically cannot solve XOR
		const slp = new Perceptron([2, 1]);
		slp.activation = Perceptron.SIGMOID;

		for(let i = 0; i < 2000; i++){
			const data = trainingData[i % 4];
			slp.forward(...data.input);
			slp.backward(data.target);
			slp.update(0.5);
		}

		// SLP will likely predict ~0.5 for everything as it tries to find a middle ground
		const slpResult = slp.forward(0, 1)[0][0].real;
		logger.debug(`SLP (0,1) prediction: ${slpResult.toFixed(3)}`);
		// We expect it to be far from 1.0 (usually around 0.5)
		assert.ok(slpResult < 0.7, 'SLP should not be able to confidently solve XOR');

		// 2. MLP (Hidden layer) - Can solve XOR
		const mlp = new Perceptron([2, 4, 1]);
		mlp.activation = Perceptron.SIGMOID;

		for(let i = 0; i < 5000; i++){
			const data = trainingData[i % 4];
			mlp.forward(...data.input);
			mlp.backward(data.target);
			mlp.update(0.5);
		}

		const results = trainingData.map(d => mlp.forward(...d.input)[0][0].real);
		logger.debug(`MLP XOR Results: [0,0]->${results[0].toFixed(3)}, [0,1]->${results[1].toFixed(3)}, [1,0]->${results[2].toFixed(3)}, [1,1]->${results[3].toFixed(3)}`);

		assert.approximately(results[0], 0, 0.15, 'MLP should solve XOR [0,0]');
		assert.approximately(results[1], 1, 0.15, 'MLP should solve XOR [0,1]');
		assert.approximately(results[2], 1, 0.15, 'MLP should solve XOR [1,0]');
		assert.approximately(results[3], 0, 0.15, 'MLP should solve XOR [1,1]');
	});
});

