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
	logger.debug(`Creating a ${m}x${n} matrix`);

	const data = new Float64Array(dataArray ?? m * n);

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

				return Matrix(m, n, newArray);
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

				// LU decomposition with partial pivoting
				// Create working copy of the matrix data
				const lu = data.slice();
				let swaps = 0; // track row swaps for sign of determinant

				// Perform LU decomposition
				for(let k = 0; k < n - 1; ++k){
					// Find pivot (largest absolute value in column k, from row k downward)
					let maxRow = k;
					let maxVal = Math.abs(lu[k * n + k]);
					for(let i = k + 1; i < n; ++i){
						const val = Math.abs(lu[i * n + k]);
						if(val > maxVal){
							maxVal = val;
							maxRow = i;
						}
					}

					// Swap rows if needed
					if(maxRow !== k){
						for(let j = 0; j < n; ++j){
							const temp = lu[k * n + j];
							lu[k * n + j] = lu[maxRow * n + j];
							lu[maxRow * n + j] = temp;
						}
						swaps++;
					}

					// Check for singular matrix
					if(Math.abs(lu[k * n + k]) < 1e-10){
						return 0;
					}

					// Eliminate column k below diagonal
					for(let i = k + 1; i < n; ++i){
						const factor = lu[i * n + k] / lu[k * n + k];
						lu[i * n + k] = factor; // store multiplier (not needed for det, but keeps it LU)
						for(let j = k + 1; j < n; ++j){
							lu[i * n + j] -= factor * lu[k * n + j];
						}
					}
				}

				// Determinant is product of diagonal elements times sign from swaps
				let det = (swaps % 2 === 0) ? 1 : -1;
				for(let i = 0; i < n; ++i){
					det *= lu[i * n + i];
				}

				return det;
			}
		},
		dimensions: {
			value: [m, n]
		},
		inverse: {
			value: function(){
				if(m !== n){
					throw new Error('Inverse is only defined for square matrices');
				}

				// Special case for 1x1
				if(m === 1){
					if(Math.abs(data[0]) < 1e-10){
						throw new Error('Matrix is singular and cannot be inverted');
					}
					return Matrix(1, 1, [1 / data[0]]);
				}

				// Special case for 2x2 (direct formula is faster)
				if(m === 2){
					const det = data[0] * data[3] - data[1] * data[2];
					if(Math.abs(det) < 1e-10){
						throw new Error('Matrix is singular and cannot be inverted');
					}
					return Matrix(2, 2, [
						data[3] / det, -data[1] / det,
						-data[2] / det, data[0] / det
					]);
				}

				// For larger matrices, use LU decomposition with partial pivoting
				const lu = data.slice();
				const perm = new Array(n); // permutation array to track row swaps
				for(let i = 0; i < n; ++i){
					perm[i] = i;
				}

				// LU decomposition with partial pivoting
				for(let k = 0; k < n - 1; ++k){
					// Find pivot
					let maxRow = k;
					let maxVal = Math.abs(lu[k * n + k]);
					for(let i = k + 1; i < n; ++i){
						const val = Math.abs(lu[i * n + k]);
						if(val > maxVal){
							maxVal = val;
							maxRow = i;
						}
					}

					// Swap rows in LU and permutation array
					if(maxRow !== k){
						for(let j = 0; j < n; ++j){
							const temp = lu[k * n + j];
							lu[k * n + j] = lu[maxRow * n + j];
							lu[maxRow * n + j] = temp;
						}
						const tempPerm = perm[k];
						perm[k] = perm[maxRow];
						perm[maxRow] = tempPerm;
					}

					// Check for singular matrix
					if(Math.abs(lu[k * n + k]) < 1e-10){
						throw new Error('Matrix is singular and cannot be inverted');
					}

					// Eliminate below diagonal
					for(let i = k + 1; i < n; ++i){
						const factor = lu[i * n + k] / lu[k * n + k];
						lu[i * n + k] = factor;
						for(let j = k + 1; j < n; ++j){
							lu[i * n + j] -= factor * lu[k * n + j];
						}
					}
				}

				// Check final diagonal element
				if(Math.abs(lu[(n - 1) * n + (n - 1)]) < 1e-10){
					throw new Error('Matrix is singular and cannot be inverted');
				}

				// Solve AX = I by solving for each column of the inverse
				const invData = new Float64Array(n * n);

				for(let col = 0; col < n; ++col){
					// Create the col-th column of identity matrix with permutation
					const b = new Float64Array(n);
					b[perm[col]] = 1;

					// Forward substitution (solve Ly = Pb where L has 1's on diagonal)
					const y = new Float64Array(n);
					for(let i = 0; i < n; ++i){
						let sum = b[i];
						for(let j = 0; j < i; ++j){
							sum -= lu[i * n + j] * y[j];
						}
						y[i] = sum;
					}

					// Backward substitution (solve Ux = y)
					const x = new Float64Array(n);
					for(let i = n - 1; i >= 0; --i){
						let sum = y[i];
						for(let j = i + 1; j < n; ++j){
							sum -= lu[i * n + j] * x[j];
						}
						x[i] = sum / lu[i * n + i];
					}

					// Store this column in the inverse matrix
					for(let row = 0; row < n; ++row){
						invData[row * n + col] = x[row];
					}
				}

				return Matrix(n, n, invData);
			}
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

				return Matrix(m, newCols, newArray);
			}
		},
		scalar: {
			value: function(scalar){
				const scaled = data.slice();
				for(let i = 0; i < data.length; ++i){
					scaled[i] *= scalar;
				}
				return Matrix(m, n, scaled);
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

				return Matrix(n, m, transposedArray);
			}
		}
	});

	// allow 2D indexing, ie. matrix[i][j]
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
	identity: {
		value: function(size){
			const identity = new Float64Array(size * size);
			for(let i = 0; i < size; ++i){
				identity[i * size + i] = 1;
			}
			return Matrix(size, size, identity);
		}
	}
});

module.exports = Matrix;
