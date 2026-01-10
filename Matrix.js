const Logger = require('log-ng');
const path = require('node:path');

const logger = new Logger(path.basename(__filename));

/**
 * Matrix constructor
 *
 * A simple m x n matrix implementation.  All arithmetic operations are non-volatile and return
 * new Matrix instances.
 */
function Matrix(m, n, dataArray){
	if(!new.target){
		return new Matrix(...arguments);
	}
	if(!Number.isInteger(m) || m <= 0 || !Number.isInteger(n) || n <= 0){
		throw new Error('Matrix dimensions must be positive integers');
	}
	if(dataArray && dataArray.length !== m * n){
		throw new Error('Data array length does not match matrix dimensions');
	}
	logger.info(`Creating a ${m}x${n} matrix`);

	const data = new Float64Array(dataArray ?? m * n);

	function index(row, col){
		return row * n + col;
	}

	Object.defineProperties(this, {
		add: {
			value: function(matrixB){
				if(matrixB.dimensions[0] !== m || matrixB.dimensions[1] !== n){
					throw new Error('Matrix addition dimension mismatch');
				}

				const newArray = data.slice();
				for(let i = 0; i < newArray.length; ++i){
					newArray[i] += matrixB.data[i];
				}

				return Matrix.fromArray(newArray, m, n);
			}
		},
		data: {
			value: data
		},
		determinant: {
			value: function(){
				if(m !== n){
					throw new Error('Determinant is only defined for square matrices');
				}
				if(m === 1){
					return data[0];
				}
				if(m === 2){
					return data[0] * data[3] - data[1] * data[2];
				}
				// For larger matrices, use Laplace expansion (not efficient for large matrices)
				let det = 0;
				for(let col = 0; col < n; ++col){
					const subMatrixData = [];
					for(let i = 1; i < m; ++i){
						for(let j = 0; j < n; ++j){
							if(j !== col){
								subMatrixData.push(data[index(i, j)]);
							}
						}
					}
					const subMatrix = Matrix.fromArray(subMatrixData, m - 1, n - 1);
					det += ((col % 2 === 0 ? 1 : -1) * data[index(0, col)] * subMatrix.determinant());
				}
				return det;
			}
		},
		dimensions: {
			value: [m, n]
		},
		multiply: {
			value: function(matrixB){
				if(n !== matrixB.dimensions[0]){
					throw new Error('Matrix multiplication dimension mismatch');
				}

				const newCols = matrixB.dimensions[1];
				const newArray = new Float64Array(m * newCols);
				for(let i = 0; i < m; ++i){
					for(let j = 0; j < newCols; ++j){
						let sum = 0;
						for(let k = 0; k < n; ++k){
							sum += data[i * n + k] * matrixB.data[k * newCols + j];
						}
						newArray[i * newCols + j] = sum;
					}
				}

				return Matrix.fromArray(newArray, m, matrixB.dimensions[1]);
			}
		},
		scalar: {
			value: function(scalar){
				const scaled = data.slice();
				for(let i = 0; i < data.length; ++i){
					scaled[i] *= scalar;
				}
				return Matrix.fromArray(scaled, m, n);
			}
		},
		transpose: {
			value: function(){
				const transposedArray = new Float64Array(n * m);
				for(let i = 0; i < m; ++i){
					for(let j = 0; j < n; ++j){
						transposedArray[j * m + i] = data[i * n + j];
					}
				}

				return Matrix.fromArray(transposedArray, n, m);
			}
		}
	});

	// Create a proxy to allow 2D indexing, ie. matrix[i][j]
	const indexable = new Proxy(this, {
		get(target, prop, receiver){
			const row = Number(prop);
			if(!Number.isNaN(row) && row >= 0 && row < m){
				return new Proxy({}, {
					get(_, colProp){
						const col = Number(colProp);
						if(!Number.isNaN(col) && col >= 0 && col < n){
							return data[row * n + col];
						}
						return undefined;
					},
					set(_, colProp, value){
						const col = Number(colProp);
						if(!Number.isNaN(col) && col >= 0 && col < n){
							data[row * n + col] = value;
							return true;
						}
						return false;
					}
				});
			}
			return Reflect.get(target, prop, receiver);
		},
		set(target, prop, value, receiver){
			const row = Number(prop);
			if(!Number.isNaN(row) && row >= 0 && row < m){
				throw new Error('Use matrix[row][col] to set individual elements.');
			}
			return Reflect.set(target, prop, value, receiver);
		}
	});

	return indexable;
}

Object.defineProperties(Matrix, {
	fromArray: {
		value: function(array, m, n){
			if(array.length % n !== 0){
				throw new Error('Array length is not compatible with specified number of columns');
			}
			return new Matrix(m, n, array);
		}
	},
	identity: {
		value: function(size){
			const identity = new Float64Array(size * size);
			for(let i = 0; i < size; ++i){
				identity[i * size + i] = 1;
			}
			return Matrix.fromArray(identity, size, size);
		}
	}
});

module.exports = Matrix;
