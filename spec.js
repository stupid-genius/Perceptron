const { assert } = require('chai');
const { default: Logger } = require('log-ng');

const logger = new Logger('spec.js');

describe('Client', function(){
	before(function(){
		Logger.setLogLevel('debug');
	});
	it('should have tests', function(){
		logger.info('browser logs!');
		assert.equal(1 + 1, 2);
	});
});
