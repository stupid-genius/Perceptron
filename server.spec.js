const { assert } = require('chai');
const Logger = require('log-ng');
const path = require('path');

const logger = new Logger(path.basename(__filename));

describe('Server', function(){
	it('should have tests', function(){
		logger.info('server logs!');
		assert.equal(1 + 1, 2);
	});
});
