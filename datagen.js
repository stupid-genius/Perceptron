const fs = require('node:fs');
const path = require('node:path');

const nSamples = 100;
const trainRatio = 0.75;
const normalize = true;
const noiseRange = 0.05;
const trainFile = 'training.csv';
const testFile = 'test.csv';

/**
 * Generates synthetic data for (hours slept, hours studied) -> grade
 * and writes training and test sets to CSV files
 *
 * @param {Object} options
 * @param {number} options.nSamples Total number of samples (default 40)
 * @param {number} options.trainRatio Fraction of samples for training (default 0.75)
 * @param {boolean} options.normalize Whether to normalize inputs to [0,1] (default true)
 * @param {number} options.noiseRange Optional noise amplitude ± (default 0.05)
 * @param {string} options.trainFile Path to write training CSV (default 'training.csv')
 * @param {string} options.testFile Path to write test CSV (default 'test.csv')
 */
function generateData(){
    const scaleX = normalize ? 12 : 1;
    const scaleY = 1;

    function trueFunction(x1, x2){
        let y = 0.1*x1 + 0.07*x2 + 0.2;
        if(noiseRange){
            y += (Math.random() * 2 * noiseRange) - noiseRange;
        }
        return y;
    }

    const nTrain = Math.floor(nSamples * trainRatio);
    const nTest = nSamples - nTrain;

    const training = [];
    const test = [];

    for(let i = 0; i < nTrain; i++){
        const x1 = Math.random()*12;
        const x2 = Math.random()*12;
        const y = trueFunction(x1, x2);
        training.push([
            normalize ? x1/scaleX : x1,
            normalize ? x2/scaleX : x2,
            normalize ? y/scaleY : y
        ]);
    }

    for(let i = 0; i < nTest; i++){
        const x1 = Math.random()*12;
        const x2 = Math.random()*12;
        const y = trueFunction(x1, x2);
        test.push([
            normalize ? x1/scaleX : x1,
            normalize ? x2/scaleX : x2,
            normalize ? y/scaleY : y
        ]);
    }

    return { training, test };
}

function writeCSV(filename, data){
	const content = data.map(row => row.join(',')).join('\n');
	fs.writeFileSync(path.resolve(filename), content, 'utf8');
	console.log(`Wrote ${data.length} rows to ${filename}`);
}

if(require.main === module){
	const { training, test } = generateData();

    writeCSV(trainFile, training);
    writeCSV(testFile, test);
}else{
	module.exports = generateData;
}

