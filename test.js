const Logger = require('log-ng');
const path = require('path');

Logger({logLevel: 'info', logFile: 'mochaTest.log'});
const logger = new Logger(path.basename(__filename));

const app = require('./testServer.js');

exports.mochaHooks = {
	beforeAll(done){
		this.testServer = app.listen(3000, 'localhost', done);
		logger.info('Test server started');
	},
	afterAll(done){
		this.testServer.close(done);
		logger.info('Test server closed');
	}
};
