// Karma configuration
// Generated on Fri Jan 05 2024 15:51:09 GMT-0500 (Eastern Standard Time)
const fs = require('fs');

module.exports = function(config) {
	config.set({
		// base path that will be used to resolve all patterns (eg. files, exclude)
		basePath: '',
		// frameworks to use
		// available frameworks: https://www.npmjs.com/search?q=keywords:karma-adapter
		frameworks: ['mocha'],
		// list of files / patterns to load in the browser
		files: [
			'spec.js'
		],
		// list of files / patterns to exclude
		exclude: [
		],
		// preprocess matching files before serving them to the browser
		// available preprocessors: https://www.npmjs.com/search?q=keywords:karma-preprocessor
		preprocessors: {
			'spec.js': ['esbuild']
		},
		esbuild: {
			bundle: true,
			// external: [],
			// format: 'cjs'
			// platform: 'node',
			target: 'es2022'
		},
		// test results reporter to use
		// possible values: 'dots', 'progress'
		// available reporters: https://www.npmjs.com/search?q=keywords:karma-reporter
		reporters: ['mocha'],
		// protocol: 'https:',
		// web server port
		port: 9876,
		// httpsServerOptions: {
		// 	key: fs.readFileSync('key.pem'),
		// 	cert: fs.readFileSync('cert.pem'),
		// },
		// enable / disable colors in the output (reporters and logs)
		colors: true,
		// level of logging
		// possible values: config.LOG_DISABLE || config.LOG_ERROR || config.LOG_WARN || config.LOG_INFO || config.LOG_DEBUG
		logLevel: config.LOG_INFO,
		// loggers: [
		// 	{type: 'console'}
		// ],
		client: {
			// allowConsoleLogs: false,
			captureConsole: true
		},
		// browserConsoleLogOptions: {
		// 	level: 'log',
		// 	format: '%b %T: %m',
		// 	terminal: true,
		// },
		// enable / disable watching file and executing tests whenever any file changes
		autoWatch: false,
		customLaunchers: {
			ChromeHeadlessSecure: {
				base: 'ChromeHeadless',
				flags: [
					'--disable-web-security',
					'--ignore-certificate-errors',
					'--enable-features=SharedArrayBuffer'
				]
			},
			FirefoxHeadlessSecure: {
				base: 'Firefox',
				prefs: {
					// 'javascript.options.shared_memory': true,
					// 'dom.postMessage.sharedArrayBuffer.withCOOP_COEP': true,
					// 'security.fileuri.strict_origin_policy': false,
					// 'security.fileuri.origin_policy': 'null',
					// 'dom.webgpu.enabled': false,
					// 'security.sandbox.content.level': 0,
					// 'network.http.use-cache': false
				},
				flags: [
					'-headless',
					// '--disable-gpu',
					// '--allow-insecure-localhost'
				]
			},
			FirefoxHeadless: {
				base: 'Firefox',
				flags: ['-headless']
			}
		},
		// start these browsers
		// available browser launchers: https://www.npmjs.com/search?q=keywords:karma-launcher
		browsers: ['ChromeHeadless', 'FirefoxHeadless'],
		// browsers: ['ChromeHeadlessSecure', 'FirefoxHeadlessSecure'],
		// Continuous Integration mode
		// if true, Karma captures browsers, runs the tests and exits
		singleRun: true,
		// Concurrency level
		// how many browser instances should be started simultaneously
		concurrency: Infinity,
		// browserDisconnectTimeout: 10000,
		// browserNoActivityTimeout: 300000
	});
};
