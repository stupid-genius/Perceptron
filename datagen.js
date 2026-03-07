const fs = require('node:fs');
const process = require('node:process');
const path = require('node:path');

/**
 * Generates synthetic data for (hours slept, hours studied) -> (grade)
 *
 * @param {number} nSamples - Total number of samples (default 100)
 * @param {number} trainRatio - Fraction of samples for training (default 0.75)
 * @param {boolean} normalize - Whether to normalize inputs and outputs to [0,1] (default true)
 * @param {number} noiseRange - Optional noise amplitude ± (default 0.05)
 * @returns {Object} An object containing training and test arrays
 */
function generate2x1Data(nSamples = 100, trainRatio = 0.75, normalize = true, noiseRange = 0.05){
	const scaleX = normalize ? 12 : 1;

	function trueGrade(x1, x2){
		let y = 0.1 * x1 + 0.07 * x2 + 0.2;
		y += (Math.random() * 2 * noiseRange) - noiseRange;
		if(normalize){
			// Range: [0.2, 2.24]
			y = (y - 0.2) / 2.04;
		}
		return y;
	}

	const nTrain = Math.floor(nSamples * trainRatio);
	const nTest = nSamples - nTrain;

	const training = [];
	const test = [];

	for(let i = 0; i < nTrain; i++){
		const x1 = Math.random() * 12;
		const x2 = Math.random() * 12;
		const y1 = trueGrade(x1, x2);
		training.push([
			x1 / scaleX,
			x2 / scaleX,
			y1
		]);
	}

	for(let i = 0; i < nTest; i++){
		const x1 = Math.random() * 12;
		const x2 = Math.random() * 12;
		const y1 = trueGrade(x1, x2);
		test.push([
			x1 / scaleX,
			x2 / scaleX,
			y1
		]);
	}

	return { training, test };
}

/**
 * Generates synthetic data for (hours slept, hours studied, hours partied) -> (grade, fatigue)
 *
 * @param {number} nSamples - Total number of samples (default 100)
 * @param {number} trainRatio - Fraction of samples for training (default 0.75)
 * @param {boolean} normalize - Whether to normalize inputs and outputs to [0,1] (default true)
 * @param {number} noiseRange - Optional noise amplitude ± (default 0.05)
 * @returns {Object} An object containing training and test arrays
 */
function generate3x2Data(nSamples = 100, trainRatio = 0.75, normalize = true, noiseRange = 0.05){
	const scaleX = normalize ? 12 : 1;

	function trueGrade(x1, x2, x3){
		let y = 0.1 * x1 + 0.07 * x2 - 0.05 * x3 + 0.6;
		y += (Math.random() * 2 * noiseRange) - noiseRange;
		if(normalize){
			// Range: [0.0, 2.64]
			y = (y - 0.0) / 2.64;
		}
		return y;
	}

	function trueFatigue(x1, x2, x3){
		let y = 1.0 - 0.06 * x1 + 0.02 * x2 + 0.08 * x3;
		y += (Math.random() * 2 * noiseRange) - noiseRange;
		if(normalize){
			// Range: [0.28, 2.20]
			y = (y - 0.28) / 1.92;
		}
		return y;
	}

	const nTrain = Math.floor(nSamples * trainRatio);
	const nTest = nSamples - nTrain;

	const training = [];
	const test = [];

	for(let i = 0; i < nTrain; i++){
		const x1 = Math.random() * 12;
		const x2 = Math.random() * 12;
		const x3 = Math.random() * 12;
		const y1 = trueGrade(x1, x2, x3);
		const y2 = trueFatigue(x1, x2, x3);
		training.push([
			x1 / scaleX,
			x2 / scaleX,
			x3 / scaleX,
			[y1, y2]
		]);
	}

	for(let i = 0; i < nTest; i++){
		const x1 = Math.random() * 12;
		const x2 = Math.random() * 12;
		const x3 = Math.random() * 12;
		const y1 = trueGrade(x1, x2, x3);
		const y2 = trueFatigue(x1, x2, x3);
		test.push([
			x1 / scaleX,
			x2 / scaleX,
			x3 / scaleX,
			[y1, y2]
		]);
	}

	return { training, test };
}

function generateData(type){
	switch(type){
		case '2x1':
			return generate2x1Data;
		case '3x2':
			return generate3x2Data;
		default:
			throw new Error(`Unknown data type: ${type}`);
	}
}

function writeCSV(filename, data){
	const content = data.map(row => row.join(',')).join('\n');
	fs.writeFileSync(path.resolve(filename), content, 'utf8');
	console.log(`Wrote ${data.length} rows to ${filename}`);
}

if(require.main === module){
	const type = process.argv[2] || '2x1';
	const options = {
		nSamples: parseInt(process.argv[3]) || 100,
		trainRatio: parseFloat(process.argv[4]) || 0.75,
		normalize: process.argv[5] !== 'false',
		noiseRange: parseFloat(process.argv[6]) || 0.05
	};

	const { training, test } = generateData(type)(...Object.values(options));

	const trainFile = 'training.csv';
	const testFile = 'test.csv';
    writeCSV(trainFile, training);
    writeCSV(testFile, test);
}else{
	module.exports = {
		generate2x1Data,
		generate3x2Data
	}
}

