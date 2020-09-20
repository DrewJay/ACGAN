let tf = require('@tensorflow/tfjs');

const fs = require('fs');
const path = require('path');

const argparse = require('argparse');
const data = require('./data/data');

// System constants.
const constants = {
	SOFT_ONE: 0.95,
	defaults: {
		gpu: false,
		epochs: 100,
		batchSize: 100,
		latentSpaceSize: 100,
		learningRate: 0.0002,
		adamBeta: 0.5,
		generatorSavePath: './log',
		logDir: './dist/generator',
		classes: 10,
		squareDim: 28,
	},
};

let NUM_CLASSES, IMAGE_SIZE;

/**
 * First part of composite sequetial neural network. Generator takes random noise
 * as an input and using convolutional layers, it reshapes the noise into 28x28 MNIST
 * format image. Generator learns to create better results using the embedding layer.
 * 
 * @param latentSpaceSize - The size of "latent space" vectors - or the noise.
 * @returns Tensorflow generator model.
 */
function buildGenerator(latentSpaceSize) {
    const cnn = tf.sequential();

    // Output of this layer is [null, 3456].
    cnn.add(
        tf.layers.dense({
            units: 3 * 3 * 384,
            inputShape: [latentSpaceSize],
            activation: 'relu',
        })
    );

    // Total target shape (dimensions multiplied) have to correspnd to units amount
    // of previous layer.
    cnn.add(tf.layers.reshape({targetShape: [3, 3, 384]}));

    // When kernel of size 5 moves over [3, 3, ] matrix with 1 stride and valid padding,
    // it creates [7, 7, ] matrix. Valid padding means whole [3, 3, ] matrix has to fully
    // fit into moving kernel.
    cnn.add(tf.layers.conv2dTranspose({
        filters: 192,
        kernelSize: 5,
        strides: 1,
        padding: 'valid',
        activation: 'relu',
        kernelInitializer: 'glorotNormal'
    }));
    cnn.add(tf.layers.batchNormalization());

    // Same padding with stride equal to 2 is basically dimension multiplication.
    // So 2 * 7 = 14. Same padding means center of the kernel moving over matrix.
    // https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
    cnn.add(tf.layers.conv2dTranspose({
        filters: 96,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'relu',
        kernelInitializer: 'glorotNormal',
    }));
    cnn.add(tf.layers.batchNormalization());

    // Same as before, stride 2 means doubling the dimensions.
    cnn.add(tf.layers.conv2dTranspose({
        filters: 1,
        kernelSize: 5,
        strides: 2,
        padding: 'same',
        activation: 'tanh',
        kernelInitializer: 'glorotNormal',
    }));

    // First generator input is latent space.
    const latent = tf.input({ shape: [latentSpaceSize] });

    // Second input is desired number drawing we want to generate.
    const imageClass = tf.input({ shape: [1] });

    // Embedding layer verctorizes scalar value. Alternative to embedding layer
    // would be dense layer that takes [1] input and converts it into [latentSpaceSize]
    // input. Embedding is smart shortcut that fits this situation perfectly.
    const classEmbedding = tf.layers.embedding({
		inputDim: NUM_CLASSES,
		outputDim: latentSpaceSize,
		embeddingsInitializer: 'glorotNormal',
    }).apply(imageClass);

    // Now we need to create correlation between original generated latent space
    // and the one that is being trained from embedding layer. Hadamard product
    // "mixes" the noise with embedding vector and uses it as a seed of randomness.
    const h = tf.layers.multiply().apply([latent, classEmbedding]);

    // Feed the correlated latent space into our upscaling CNN.
    const fakeImage = cnn.apply(h);

    // Return the model itself. Inputs are latent space and seeked image class,
    // output is the image.
    return tf.model({ inputs: [latent, imageClass], outputs: fakeImage });
}

/**
 * Last part of composite sequential neural network. It is taught to reliably
 * distinguish between generated images and training MNIST images. It has to
 * be trained during the generator training, because it uses generator internally.
 * Discriminator needs to be taught continually, as generator teaches to generate
 * better fakes, since the better looking fakes are still not real and discriminator
 * needs to be able to detect that.
 * 
 * @returns Tensorflow discriminator model.
 */
function buildDiscriminator() {
    const cnn = tf.sequential();

    // Standard MNIST [28, 28, 1] image  will be subject of multiple
    // convolution layers.
    cnn.add(tf.layers.conv2d({
        filters: 32,
        kernelSize: 3,
        padding: 'same',
        strides: 2,
        inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
    }));
    cnn.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    cnn.add(tf.layers.dropout({ rate: 0.3 }));

    // Another convolution layer with increased amount of filters.
    cnn.add(tf.layers.conv2d({
        filters: 64,
        kernelSize: 3,
        padding: 'same',
        strides: 1,
    }));
    cnn.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    cnn.add(tf.layers.dropout({ rate: 0.3 }));

    // Double the amount of filters to catch more image features.
    cnn.add(tf.layers.conv2d({
        filters: 128,
        kernelSize: 3,
        padding: 'same',
        strides: 2,
    }));
    cnn.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    cnn.add(tf.layers.dropout({ rate: 0.3 }));

    // Again, double the amount of filters.
    cnn.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: 3,
        padding: 'same',
        strides: 1,
    }));
    cnn.add(tf.layers.leakyReLU({ alpha: 0.2 }));
    cnn.add(tf.layers.dropout({ rate: 0.3 }));

    // Finally - flatten the data.
    cnn.add(tf.layers.flatten());

    // Input to discriminator is [28, 28, 1] MNIST image.
    const image = tf.input({ shape: [IMAGE_SIZE, IMAGE_SIZE, 1] });

    // And we feed it into the CNN.
    const features = cnn.apply(image);

    // Sigmoid is used for binary classifications. Here we are determining whether
    // image belongs to 1 (real) class or 0 (fake) class.
    const realnessScore = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
    }).apply(features);

    // Softmax is used for multiclass classifications. Here we are determining
    // probabilities of source value belonging to every possible class.
    const aux = tf.layers.dense({
        units: NUM_CLASSES,
        activation: 'softmax',
    }).apply(features);

    // Return the model itself. Input is the image and output both
    // realness score (realness class) and auxilliary probabilites.
    return tf.model({ inputs: image, outputs: [realnessScore, aux] });
}

/**
 * Stick generator and discriminator together. It this chained model,
 * only generator is being taught.
 *
 * @param latentSpaceSize - Latent vectors size.
 * @param generator - The generator model.
 * @param discriminator - The discriminator model.
 * @param optimizer - Discriminator optimizer.
 * 
 * @returns Combined tensorflow model.
 */
function buildCombinedModel(
	latentSpaceSize,
	generator,
	discriminator,
	optimizer,
) {
    // Generated latent space.
    const latent = tf.input({shape: [latentSpaceSize]});
    // Desired image class.
    const imageClass = tf.input({shape: [1]});
    // Symbolic output from generator.
    let fakeImage = generator.apply([latent, imageClass]);

    // We only want to be able to train generation for the combined model.
    discriminator.trainable = false;
    // Get symbolic outputs from discriminator.
    [fakeScore, auxScore] = discriminator.apply(fakeImage);

    // Build combined model using symbolic tensors.
    const combined = tf.model({
        inputs: [latent, imageClass], outputs: [fakeScore, auxScore],
    });

    // Compile the final model. Binary cross entropy is used as a loss
    // function for binary classification (fakeness), sparse categorical cross entropy
    // for multiclass classification (numeric class allegiance).
    combined.compile({
        optimizer,
        loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy'],
    });
    return combined;
}

/**
 * Discriminator needs to be trained simultaneously along the generator.
 *
 * @param xTrain - MNIST training images. 
 * @param yTrain - MNIST labels. 
 * @param batchStart - Current batch start index.
 * @param batchSize - Batch size.
 * @param latentSpaceSize - Latent vectors size.
 * @param generator - Generator model reference.
 * @param discriminator - Discriminator model reference.
 *
 * @returns Discriminator losses.
 */
async function trainDiscriminatorOneStep(
	xTrain,
	yTrain,
	batchStart,
	batchSize,
	latentSpaceSize,
	generator,
	discriminator,
) {
    const [x, y, auxY] = tf.tidy(() => {
        // MNIST images.
        const imageBatch = xTrain.slice(batchStart, batchSize);
        // Corresponding MNIST labels.
        const labelBatch = yTrain.slice(batchStart, batchSize).asType('float32');
        
        // Here we generate the "latent space". It is the fake counterside of imageBatch.
        let zVectors = tf.randomUniform([batchSize, latentSpaceSize], -1, 1);
        // Generate numeric class for each latent space created before.
        let sampledLabels = tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32').asType('float32');

        // Generate image using generator.
        const generatedImages = generator.predict([zVectors, sampledLabels], { batchSize });

        // Concatenate MNIST images and generated images.
        const x = tf.concat([imageBatch, generatedImages], 0);
        // Concatenate labels for real and fake images. That's why first labels are 1s
        // followed by zeros (1s for real images, 0s for fake images).
        const y = tf.tidy(() =>
            tf.concat([
                    tf.ones([batchSize, 1]).mul(constants.SOFT_ONE),
                    tf.zeros([batchSize, 1]),
                ]
            )
        );

        // Create auxilliary classes tensor. First are MNIST labels followed
        // by generated sampled labels.
        const auxY = tf.concat([labelBatch, sampledLabels], 0);
        // Return generated images and labels (fakeness + aux).
        return [x, y, auxY];
    });

    // Train discriminator on single batch.
    const losses = await discriminator.trainOnBatch(x, [y, auxY]);
    tf.dispose([x, y, auxY]);

    // Return fakeness + aux score. 
    return losses;
}

/**
 * Training combined model means technically training generator based
 * on losses generated by current state of discriminator.
 *
 * @param batchSize - Batch size.
 * @param latentSpaceSize - Latent vectors size. 
 * @param combined - Combined models reference.
 * 
 * @returns Generator model losses. 
 */
async function trainCombinedModelOneStep(batchSize, latentSpaceSize, combined) {
    const [noise, sampledLabels, fakeTrick] = tf.tidy(() => {
        // Again, generate latent vectors.
        const zVectors = tf.randomUniform([batchSize, latentSpaceSize], -1, 1);
        // Generate label for every latent space.
        const sampledLabels = tf.randomUniform([batchSize, 1], 0, NUM_CLASSES, 'int32').asType('float32');

        // This is the most important part in GAN training. During the combined model
        // training only the generator is trained. If we fed fake latent vectors inside
        // with fake realness scores, we would teach generator to generate fake images.
        // Since we want to teach it to generate real images, we need to set target
        // fakeness score to 1. In early training stages convolutions in discriminator
        // will detect that it's very far from being real, this it will generate large
        // cross-entropy loss quantities and will make generator learn fast to generate
        // more real images. Embedding layer in generator is the one being taught.
        const fakeTrick = tf.tidy(() => tf.ones([batchSize, 1]).mul(constants.SOFT_ONE));

        // Return latent vectors, labels and fakeness score.
        return [zVectors, sampledLabels, fakeTrick];
    });

    // Train combined model one step.
    const losses = await combined.trainOnBatch([noise, sampledLabels], [fakeTrick, sampledLabels]);
    tf.dispose([noise, sampledLabels, fakeTrick]);

    // Return generator losses.
    return losses;
}

/**
 * Simply parse command line arguments and return them.
 *
 * @returns Parsed command line arguments.
 */
function parseArguments() {
	const parser = new argparse.ArgumentParser();

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

	// Return the final parsed arguments.
	return parser.parseArgs();
}

/**
 * Generate information to be written into the file each epoch.
 *
 * @param totalEpochs - Total scheduled epochs. 
 * @param currentEpoch - Current epoch index.
 * @param completed - Completed epochs.
 * 
 * @returns Metadata object.
 */
function makeMetadata(totalEpochs, currentEpoch, completed) {
	return {
		totalEpochs,
		currentEpoch,
		completed,
		lastUpdated: new Date().getTime(),
	};
}

/**
 * Run the whole bad boi ova here.
 */
async function run() {
	const args = parseArguments();

	// Set image size and number of classes first.
	NUM_CLASSES = args.classes;
	IMAGE_SIZE = args.squareDim;

	// Log current application setup on launch.
	console.log('\n-----Runtime configuration-----')
	for (let key of Object.keys(args)) {
		console.log(`${key}: ${args[key]}`);
	}
	console.log('-------------------------------\n');

	if (NUM_CLASSES === args.classes && IMAGE_SIZE === args.squareDim) {
		console.log(`Arguments propagated into system variables correctly ✓`);
	}

	// GPU acceleration is supported only by graphics cards with CUDA cores.
	if (args.gpu) {
		console.log('Using GPU.');
		tf = require('@tensorflow/tfjs-node-gpu');
	} else {
		console.log('Using CPU.');
		tf = require('@tensorflow/tfjs-node');
	}

	// Make directory where generator output will be saved.
	if (!fs.existsSync(path.dirname(args.generatorSavePath))) {
		fs.mkdirSync(path.dirname(args.generatorSavePath));
	}

	// Create generator saving path and metadata generation path (in generator save path).
	const saveURL = `file://${args.generatorSavePath}`;
	const metadataPath = path.join(args.generatorSavePath, 'acgan-metadata.json');

	// Build the discriminator.
	const discriminator = buildDiscriminator();
	discriminator.compile({
		optimizer: tf.train.adam(args.learningRate, args.adamBeta),
		loss: ['binaryCrossentropy', 'sparseCategoricalCrossentropy'],
	});

	// Build the generator. Does not need to be compiled since it's part
	// of the combined model.
	const generator = buildGenerator(args.latentSpaceSize);

	// Build the already precompiled combined model.
	const optimizer = tf.train.adam(args.learningRate, args.adamBeta);
	const combined = buildCombinedModel(
		args.latentSpaceSize, generator, discriminator, optimizer,
	);

	// Load and prepare MNIST training data.
	await data.loadData();
	let { images: xTrain, labels: yTrain } = data.getTrainData();
	yTrain = tf.expandDims(yTrain.argMax(-1), -1);

	// Save the generator model once before starting the training.
	await generator.save(saveURL);

	// Setup tensorboard logging directory.
	let numTensors;
	let logWriter;
	if (args.logDir) {
		console.log(`Logging to tensorboard at logdir: ${args.logDir}.`);
		logWriter = tf.node.summaryFileWriter(args.logDir);
	}

	// Primary epoch iterations.
	let step = 0;
	for (let epoch = 0; epoch < args.epochs; epoch += 1) {
		// Write some spicy metadata on each epoch start.
		fs.writeFileSync(
			metadataPath,
			JSON.stringify(makeMetadata(args.epochs, epoch, false)),
		);

		// Capture the current time.
		const tBatchBegin = tf.util.now();

		// Ceiled number of batches.
		const numBatches = Math.ceil(xTrain.shape[0] / args.batchSize);

		// Batch iterations.
		for (let batch = 0; batch < numBatches; ++batch) {
			// Batch size can not exceed total data size. Let it shrink at the end.
			const actualBatchSize = (batch + 1) * args.batchSize >= xTrain.shape[0] ?
				(xTrain.shape[0] - batch * args.batchSize) :
				args.batchSize;

			// Train the discriminator one step. Remember that discriminiator generates
			// batchSize amount of fake images and uses batchSize amount of real images,
			// that's why discriminator optimizer processes double-sized batch at given
			// time.
			const dLoss = await trainDiscriminatorOneStep(
				xTrain, yTrain, batch * args.batchSize, actualBatchSize,
				args.latentSpaceSize, generator, discriminator
			);

			// For the reasons mentioned above, combined model will use double-sized batch too.
			const gLoss = await trainCombinedModelOneStep(2 * actualBatchSize, args.latentSpaceSize, combined);

			// Show information about epochs and current loss levels.
			console.log(
				`epoch ${epoch + 1} / ${args.epochs}, batch ${batch + 1} / ${numBatches}: ` +
				`dLoss = ${dLoss[0].toFixed(6)}, gLoss = ${gLoss[0].toFixed(6)}.`);

			// Generate tensorboard logs.
			if (logWriter != null) {
				logWriter.scalar('dLoss', dLoss[0], step);
				logWriter.scalar('gLoss', gLoss[0], step);
				step += 1;
			}

			// If tensors are not being cleared properly, throw an exception.
			if (numTensors == null) {
				numTensors = tf.memory().numTensors;
			} else {
				// This validation checks if tensors are being cleared each cycle.
				tf.util.assert(
					tf.memory().numTensors === numTensors,
					`Leaked ${tf.memory().numTensors - numTensors} tensors`,
				);
			}
		}

		// Save the newly trained generator after each epoch.
		await generator.save(saveURL);
		// Log information about current epoch elapsed time and generator's storage url.
		console.log(
			`Epoch ${epoch + 1} elapsed time: ` +
			`${((tf.util.now() - tBatchBegin) / 1e3).toFixed(1)}s.`
		);
		console.log(`Saved generator model to: ${saveURL}.\n`);
	}

	// Write metadata to disk to indicate the end of the training.
	fs.writeFileSync(
		metadataPath,
		JSON.stringify(makeMetadata(args.epochs, args.epochs, true)),
	);
}

// Module is launched from command line directly.
if (require.main === module) {
    run();
}

module.exports = {
  buildCombinedModel,
  buildDiscriminator,
  buildGenerator,
  trainCombinedModelOneStep,
  trainDiscriminatorOneStep
};
