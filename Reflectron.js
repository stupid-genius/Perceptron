const { Layer, Perceptron } = require('./Perceptron.js');

/**
 * Reflectron constructor (Self-Attention Block)
 *
 * @param {number} dim - The dimensionality of the input and output (d_model)
 */
function Reflectron(dim){
    if(!new.target){
        return new Reflectron(...arguments);
    }

	const gain = 0.1;
    const squelch = 1 / Math.sqrt(dim);

    const qLayer = new Layer(dim, dim);
    const kLayer = new Layer(dim, dim);
    const vLayer = new Layer(dim, dim);
    const outLayer = new Layer(dim, dim);

    const layers = [qLayer, kLayer, vLayer, outLayer];

    layers.forEach(l => {
        l.activation = Perceptron.IDENTITY;
        const w = l.weights();
        for(let i = 0; i < w.length; i++){
            w[i] *= gain;
        }
    });

    Object.defineProperties(this, {
        dim: {
			value: dim
		},

        /**
         * Forward pass: The "Reflexive" transformation
         * @param {DualMatrix} x - The input column vector (dim x 1)
         * @returns {DualMatrix} The contextualized output (dim x 1)
         */
        forward: {
            value: function(x){
                const Q = qLayer.forward(x);
                const K = kLayer.forward(x);
                const V = vLayer.forward(x);

                const score = Q.transpose().multiply(K).map(v => v.mul(squelch));

                const attended = V.multiply(score);

                const delta = outLayer.forward(attended);
                return x.add(delta);
            }
        },

        /**
         * Update all internal projection weights
         * @param {number} learningRate
         */
        update: {
            value: function(learningRate){
                layers.forEach(l => l.update(learningRate));
                return this;
            }
        },

        /**
         * Get or set all weights and biases as a single flat array
         * @param {Float64Array|Array<number>|...number} [data] - New weights/biases to set
         * @returns {Float64Array|this} All weights/biases if no arguments, or 'this' for chaining
         */
        weights: {
            value: function(){
                if(arguments.length === 0){
                    let totalSize = 0;
                    for(const l of layers){
                        totalSize += l.weights().length + l.bias().length;
                    }
                    const all = new Float64Array(totalSize);
                    let offset = 0;
                    for(const l of layers){
                        all.set(l.weights(), offset);
                        offset += l.weights().length;
                        all.set(l.bias(), offset);
                        offset += l.bias().length;
                    }
                    return all;
                }

                let allData;
                if(arguments.length === 1 && (arguments[0] instanceof Float64Array || Array.isArray(arguments[0]))){
                    allData = arguments[0];
                }else{
                    allData = Array.from(arguments);
                }

                let offset = 0;
                for(const l of layers){
                    const wSize = l.numInputs * l.numOutputs;
                    const bSize = l.numOutputs;
                    l.weights(allData.slice(offset, offset + wSize));
                    offset += wSize;
                    l.bias(allData.slice(offset, offset + bSize));
                    offset += bSize;
                }
                return this;
            }
        }
    });
}

module.exports = { Reflectron };

