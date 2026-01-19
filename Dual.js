const Logger = require('log-ng');
const path = require('node:path');

const logger = new Logger(path.basename(__filename));

function DualNumber(real = 0, dual = 0){
	if(!new.target){
		return new DualNumber(...arguments);
	}

	function traverse(node, visited, callback) {
		if(visited.has(node)) return;
		visited.add(node);
		for(const parent of node.parents){
			traverse(parent, visited, callback);
		}
		callback(node);
	};

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

				const out = DualNumber(
					this.real > dualB.real ? this.real : dualB.real,
					this.real > dualB.real ? this.dual : dualB.dual
				);
				out.backward = () => {
					if(this.real > dualB.real){
						this.grad += out.grad;
					}else if(dualB.real > this.real){
						dualB.grad += out.grad;
					}else{
						this.grad += out.grad * 0.5;
						dualB.grad += out.grad * 0.5;
					}
				};

				out.parents.push(this, dualB);
				return out;
			}
		},
		min: {
			// TODO use spread operator to handle multiple args
			value: function(dualB){
				if(!(dualB instanceof DualNumber)){
					dualB = DualNumber(dualB, 0);
				}

				const out = DualNumber(
					this.real < dualB.real ? this.real : dualB.real,
					this.real < dualB.real ? this.dual : dualB.dual
				);
				out.backward = () => {
					if(this.real < dualB.real){
						this.grad += out.grad;
					}else if(dualB.real < this.real){
						dualB.grad += out.grad;
					}else{
						this.grad += out.grad * 0.5;
						dualB.grad += out.grad * 0.5;
					}
				};

				out.parents.push(this, dualB);
				return out;
			}
		},
		parents: {
			value: []
		},
		zeroGrads: {
			value: function(){
				traverse(this, new Set(), (node) => {
					node.grad = 0;
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

// function DualMatrix(){
// 	if(!new.target){
// 		return new DualMatrix(...arguments);
// 	}
// }

module.exports = DualNumber;

