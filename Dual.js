const Logger = require('log-ng');
const path = require('node:path');

const logger = new Logger(path.basename(__filename));

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
		parents: {
			value: []
		},
		backprop: {
			value: function(){
				this.grad = 1;
				const stack = [this];
				const visited = new Set();
				while(stack.length > 0){
					const node = stack.pop();
					if(node.backward && !visited.has(node)){
						node.backward();
						visited.add(node);
						for(const parent of node.parents){
							stack.push(parent);
						}
					}
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

