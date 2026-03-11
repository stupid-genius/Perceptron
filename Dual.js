const Logger = require('log-ng');
const path = require('node:path');
const Matrix = require('./Matrix.js');

const logger = new Logger(path.basename(__filename));

/**
 * Internal utility to traverse the computation graph in topological order
 * @param {DualNumber|DualMatrix} root - The starting node
 * @param {Set} visited - Set to keep track of visited nodes
 * @param {function} callback - Function to call on each node
 */
function traverse(root, visited, callback){
	const stack = [[root, 0]];

	while(stack.length > 0){
		const entry = stack[stack.length - 1];
		const node = entry[0];
		const pIdx = entry[1];

		if(visited.has(node)){
			stack.pop();
			continue;
		}

		if(pIdx < node.parents.length){
			entry[1]++;
			stack.push([node.parents[pIdx], 0]);
		}else{
			visited.add(node);
			callback(node);
			stack.pop();
		}
	}
}

/**
 * DualNumber constructor for automatic differentiation
 * @param {number} [real=0] - The real part
 * @param {number} [dual=0] - The dual part (seed for derivative)
 *
 * @example
 * // Forward Mode Differentiation:
 * // f(x) = x^2 + 2x + 1, f'(x) = 2x + 2
 * // For x = 3: f(3) = 16, f'(3) = 8
 * const x = DualNumber(3, 1); // seed dual with 1 for f'(x)
 * const f = x.mul(x).add(x.mul(2)).add(1);
 * console.log(f.real); // 16
 * console.log(f.dual); // 8
 *
 * @example
 * // Backward Mode (Reverse) Differentiation:
 * // Use for multiple inputs or complex graphs
 * const a = DualNumber(3);
 * const b = DualNumber(2);
 * const out = a.mul(a).add(a.mul(b)); // f(a, b) = a^2 + ab
 * out.backprop();
 * console.log(a.grad); // df/da = 2a + b = 2(3) + 2 = 8
 * console.log(b.grad); // df/db = a = 3
 */
function DualNumber(real = 0, dual = 0){
	if(!new.target){
		return new DualNumber(...arguments);
	}

	Object.defineProperties(this, {
		real: {
			value: real,
			writable: true
		},
		dual: {
			value: dual,
			writable: true
		},
		grad: {
			value: 0,
			writable: true
		},
		add: {
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const sum = new DualNumber(
					this.real + dualB.real,
					this.dual + dualB.dual
				);
				sum.backward = () => {
					this.grad += sum.grad;
					dualB.grad += sum.grad;
				};

				sum.parents.push(this, dualB);
				return sum;
			}
		},
		sub: {
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const diff =  DualNumber(
					this.real - dualB.real,
					this.dual - dualB.dual
				);
				diff.backward = () => {
					this.grad += diff.grad;
					dualB.grad -= diff.grad;
				};

				diff.parents.push(this, dualB);
				return diff;
			}
		},
		mul: {
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const prod = DualNumber(
					this.real * dualB.real,
					this.real * dualB.dual + this.dual * dualB.real
				);
				prod.backward = () => {
					this.grad += dualB.real * prod.grad;
					dualB.grad += this.real * prod.grad;
				};

				prod.parents.push(this, dualB);
				return prod;
			}
		},
		div: {
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const quotient = DualNumber(
					this.real / dualB.real,
					(this.dual * dualB.real - this.real * dualB.dual) / (dualB.real * dualB.real)
				);
				quotient.backward = () => {
					this.grad += (1 / dualB.real) * quotient.grad;
					dualB.grad += (-this.real / (dualB.real * dualB.real)) * quotient.grad;
				};

				quotient.parents.push(this, dualB);
				return quotient;
			}
		},
		pow: {
			value: function(exponent){
				const powReal = Math.pow(this.real, exponent);
				const powDual = DualNumber(
					powReal,
					exponent * Math.pow(this.real, exponent - 1) * this.dual
				);
				powDual.backward = () => {
					this.grad += exponent * Math.pow(this.real, exponent - 1) * powDual.grad;
				};

				powDual.parents.push(this);
				return powDual;
			}
		},
		exp: {
			value: function(){
				const expValue = Math.exp(this.real);
				const expDual = DualNumber(
					expValue,
					expValue * this.dual
				);
				expDual.backward = () => {
					this.grad += expValue * expDual.grad;
				};

				expDual.parents.push(this);
				return expDual;
			}
		},
		log: {
			value: function(){
				const logDual = DualNumber(
					Math.log(this.real),
					this.dual / this.real
				);
				logDual.backward = () => {
					this.grad += (1 / this.real) * logDual.grad;
				};

				logDual.parents.push(this);
				return logDual;
			}
		},
		abs: {
			value: function(){
				const absDual = DualNumber(Math.abs(this.real), (this.real >= 0 ? 1 : -1) * this.dual);
				absDual.backward = () => {
					this.grad += (this.real >= 0 ? 1 : -1) * absDual.grad;
				};

				absDual.parents.push(this);
				return absDual;
			}
		},
		// sign: {
		// 	value: function(){
		// 		const out = DualNumber(this.real === 0 ? 0 : (this.real > 0 ? 1 : -1), 0);
		// 		out.backward = () => {
		// 			this.grad += 0;
		// 		};
		// 		out.parents.push(this);
		// 		return out;
		// 	}
		// },
		clip: {
			value: function(low, high){
				return this.max(low).min(high);
			}
		},
		max: {
			// TODO use spread operator to handle multiple args
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const max = DualNumber(
					this.real > dualB.real ? this.real : dualB.real,
					this.real > dualB.real ? this.dual : dualB.dual
				);
				max.backward = () => {
					if(this.real > dualB.real){
						this.grad += max.grad;
					}else if(dualB.real > this.real){
						dualB.grad += max.grad;
					}else{
						this.grad += max.grad * 0.5;
						dualB.grad += max.grad * 0.5;
					}
				};

				max.parents.push(this, dualB);
				return max;
			}
		},
		min: {
			// TODO use spread operator to handle multiple args
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const min = DualNumber(
					this.real < dualB.real ? this.real : dualB.real,
					this.real < dualB.real ? this.dual : dualB.dual
				);
				min.backward = () => {
					if(this.real < dualB.real){
						this.grad += min.grad;
					}else if(dualB.real < this.real){
						dualB.grad += min.grad;
					}else{
						this.grad += min.grad * 0.5;
						dualB.grad += min.grad * 0.5;
					}
				};

				min.parents.push(this, dualB);
				return min;
			}
		},
		parents: {
			value: []
		},
		zeroGrads: {
			value: function(){
				traverse(this, new Set(), (node) => {
					if(node.grad instanceof Matrix){
						node.grad.data.fill(0);
					}else{
						node.grad = 0;
					}
				});
			}
		},
		backprop: {
			value: function(seed = 1){
				const topo = [];
				traverse(this, new Set(), (node) => {
					topo.push(node);
				});

				this.grad += seed;

				for(let i = topo.length - 1; i >= 0; --i){
					const node = topo[i];
					node.backward?.();
				}
			}
		}
	});
}

/**
 * DualMatrix constructor for matrix operations with automatic differentiation
 * @param {number} m - Number of rows
 * @param {number} n - Number of columns
 * @param {Float64Array|Array<number>} [dataArray] - Flattened data (row-major)
 */
function DualMatrix(m, n, dataArray){
	if(!new.target){
		return new DualMatrix(...arguments);
	}

	const real = Matrix(m, n, dataArray);
	const grad = Matrix(m, n); // initialized to zeros

	Object.defineProperties(this, {
		real: {
			value: real
		},
		grad: {
			value: grad
		},
		parents: {
			value: []
		},
		dimensions: {
			get: () => [m, n]
		},
		add: {
			value: function(other){
				if(!(other instanceof DualMatrix)){
					throw new Error('DualMatrix addition only supports DualMatrix for now');
				}

				const sum = DualMatrix(m, n, this.real.add(other.real).data);
				sum.backward = () => {
					// dLoss/dA = dLoss/dOut, dLoss/dB = dLoss/dOut
					for(let i = 0; i < this.grad.data.length; ++i){
						this.grad.data[i] += sum.grad.data[i];
						other.grad.data[i] += sum.grad.data[i];
					}
				};

				sum.parents.push(this, other);
				return sum;
			}
		},
		multiply: {
			value: function(other){
				if(!(other instanceof DualMatrix)){
					throw new Error('DualMatrix multiplication only supports DualMatrix for now');
				}

				const prodReal = this.real.multiply(other.real);
				const prod = DualMatrix(m, other.real.dimensions[1], prodReal.data);

				prod.backward = () => {
					// Matrix Calculus:
					// If Y = A * B, then:
					// gradA += gradY * B^T
					// gradB += A^T * gradY
					const gradA = prod.grad.multiply(other.real.transpose());
					const gradB = this.real.transpose().multiply(prod.grad);

					for(let i = 0; i < this.grad.data.length; ++i){
						this.grad.data[i] += gradA.data[i];
					}
					for(let i = 0; i < other.grad.data.length; ++i){
						other.grad.data[i] += gradB.data[i];
					}
				};

				prod.parents.push(this, other);
				return prod;
			}
		},
		transpose: {
			value: function(){
				const out = DualMatrix(n, m, this.real.transpose().data);
				out.backward = () => {
					const gradT = out.grad.transpose();
					for(let i = 0; i < this.grad.data.length; ++i){
						this.grad.data[i] += gradT.data[i];
					}
				};
				out.parents.push(this);
				return out;
			}
		},
		map: {
			value: function(fn){
				const outData = new Float64Array(m * n);
				const innerResults = [];

				for(let i = 0; i < m * n; ++i){
					const dn = DualNumber(this.real.data[i]);
					const res = fn(dn);
					outData[i] = res.real;
					innerResults.push({ dn, res });
				}

				const out = DualMatrix(m, n, outData);
				out.backward = () => {
					for(let i = 0; i < m * n; ++i){
						const { dn, res } = innerResults[i];
						res.backprop(out.grad.data[i]);
						this.grad.data[i] += dn.grad;
					}
				};

				out.parents.push(this);
				return out;
			}
		},
		toString: {
			value: function(){
				return `DualMatrix(${m}x${n}):\nReal:\n${this.real.toString()}\nGrad:\n${this.grad.toString()}`;
			}
		},
		zeroGrads: {
			value: function(){
				traverse(this, new Set(), (node) => {
					// Handle both DualNumber (scalar .grad) and DualMatrix (matrix .grad)
					if(node.grad instanceof Matrix){
						node.grad.data.fill(0);
					}else{
						node.grad = 0;
					}
				});
			}
		},
		backprop: {
			value: function(seed){
				const topo = [];
				traverse(this, new Set(), (node) => {
					topo.push(node);
				});

				if(!seed){
					// Default to a matrix of ones if no seed is provided
					seed = Matrix(m, n, new Float64Array(m * n).fill(1));
				}

				if(seed instanceof Matrix){
					for(let i = 0; i < grad.data.length; ++i){
						grad.data[i] += seed.data[i];
					}
				}else{
					for(let i = 0; i < grad.data.length; ++i){
						grad.data[i] += seed;
					}
				}

				for(let i = topo.length - 1; i >= 0; --i){
					const node = topo[i];
					node.backward?.();
				}
			}
		}
	});

	const indexable = new Proxy(this, {
		get(target, prop, receiver){
			if(typeof prop === 'string'){
				const row = Number(prop);
				if(!Number.isNaN(row) && row >= 0 && row < m){
					return new Proxy({}, {
						get(_, colProp){
							if(typeof colProp === 'string'){
								const col = Number(colProp);
								if(!Number.isNaN(col) && col >= 0 && col < n){
									const index = row * n + col;
									const out = DualNumber(target.real.data[index]);
									out.backward = () => {
										target.grad.data[index] += out.grad;
									};
									out.parents.push(target);
									return out;
								}
							}
							return undefined;
						}
					});
				}
			}
			return Reflect.get(target, prop, receiver);
		}
	});

	return indexable;
}

module.exports = {
	DualNumber,
	DualMatrix
};

