	parser.addArgument('--gpu',
		{ action: 'storeTrue', help: 'Use tfjs-node-gpu for training (required CUDA GPU).' }
	);
	parser.addArgument('--epochs',
		{ type: 'int', defaultValue: constants.defaults.epochs, help: 'Number of training epochs.' },
	);
	parser.addArgument('--batchSize',
		{ type: 'int', defaultValue: constants.defaults.batchSize, help: 'Batch size to be used during training.' }
	);
	parser.addArgument('--latentSpaceSize',
		{ type: 'int', defaultValue: constants.defaults.latentSpaceSize, help: 'Size of the latent space (z-space).' }
	);
	parser.addArgument('--learningRate',
		{ type: 'float', defaultValue: constants.defaults.learningRate, help: 'Learning rate.' }
	);
	parser.addArgument('--adamBeta',
		{ type: 'float', defaultValue: constants.defaults.adamBeta, help: 'Beta1 parameter of the ADAM optimizer.' }
	);
	parser.addArgument('--generatorSavePath',
		{ type: 'string', defaultValue: constants.defaults.generatorSavePath, help: 'Path to which the generator model will be saved after every epoch.' }
	);
	parser.addArgument('--logDir',
		{ type: 'string', defaultValue: constants.defaults.logDir, help: 'Optional log directory to which the loss values will be written.' }
	);
	parser.addArgument('--classes',
		{ type: 'int', defaultValue: constants.defaults.classes, help: 'The amount of possible image classes.' }
	);
	parser.addArgument('--squareDim',
		{ type: 'int', defaultValue: constants.defaults.squareDim, help: 'Image dimension to be squared automatically.' }
	);
