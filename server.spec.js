const {assert} = require('chai');
const Logger = require('log-ng');
const path = require('node:path');
const Matrix = require('./Matrix.js');
// const Perceptron = require('./Perceptron.js');

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
        this.timeout(10000); // allow longer test time

        const size = 500; // 500x500 matrices
        const A = Matrix.fromArray(new Array(size * size).fill(1), size, size);
        const B = Matrix.fromArray(new Array(size * size).fill(1), size, size);

        const start = performance.now();
        const C = A.multiply(B);
        const end = performance.now();

        const elapsedMs = end - start;
        logger.info(`Multiplying two ${size}x${size} matrices took ${elapsedMs.toFixed(2)} ms`);

        // Basic sanity check
        assert.equal(C.dimensions[0], size);
        assert.equal(C.dimensions[1], size);
        assert.equal(C[0][0], size); // 1*1 summed across 'size' elements
    });
});

describe('Perceptron', function(){
	it('should have tests', function(){
		logger.info('server logs!');
		assert.equal(1 + 1, 2);
	});
});
