const tf = require('@tensorflow/tfjs');

// Require necessary libraries.
const assert = require('assert');
const fs = require('fs');
const https = require('https');
const zlib = require('zlib');

// MNIST data constants.
const BASE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/';
const TRAIN_IMAGES_FILE = 'train-images-idx3-ubyte.gz';
const TRAIN_LABELS_FILE = 'train-labels-idx1-ubyte.gz';
const TEST_IMAGES_FILE = 't10k-images-idx3-ubyte.gz';
const TEST_LABELS_FILE = 't10k-labels-idx1-ubyte.gz';
const IMAGE_HEADER_MAGIC_NUM = 2051;
const IMAGE_HEADER_BYTES = 16;
const IMAGE_HEIGHT = 28;
const IMAGE_WIDTH = 28;
const IMAGE_FLAT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
const LABEL_HEADER_MAGIC_NUM = 2049;
const LABEL_HEADER_BYTES = 8;
const LABEL_RECORD_BYTE = 1;
const LABEL_FLAT_SIZE = 10;

/**
 * Download MNIST data from url resource.
 *
 * @param fileName - Name of the g-zipped data file.
 * @returns File content stream.
 */
async function fetchAndSaveTrainingData(fileName) {
	return new Promise(async (resolve) => {
		const url = `${BASE_URL}${fileName}`;
			
		// If stream files already exist, just return them.
		if (fs.existsSync(fileName)) {
			const result = await fs.readFileSync(fileName, () => {});
			return resolve(result);
		}

		// Create new stream files.
		const file = fs.createWriteStream(fileName);

		// GET the file from resource url.
		https.get(url, async (response) => {
			const unzip = zlib.createGunzip();
			response.pipe(unzip).pipe(file);

			unzip.on('end', async () => {
				const result = await fs.readFileSync(fileName, () => {});
				return resolve(result);
			});
		});
	});
}

/**
 * Load data from image header.
 *
 * @param buffer - Image data buffer.
 * @param headerLength - Total header length.
 * @returns Array of header values.
 */
function loadHeaderValues(buffer, headerLength) {
	const headerValues = [];
	const headerNumInt32s = headerLength / 4;
	
	// We read 4 byte sequences.
	for (let i = 0; i < headerNumInt32s; i++) {
		// Header data is stored in-order (aka big-endian).
		headerValues[i] = buffer.readUInt32BE(i * 4);
  	}
  return headerValues;
}

/**
 * Load images from g-zipped dataset.
 *
 * @param fileName - File name of g-zipped dataset.
 * @returns Array of image data.
 */
async function loadImages(fileName) {
	const buffer = await fetchAndSaveTrainingData(fileName);

	const headerBytes = IMAGE_HEADER_BYTES;
	const recordBytes = IMAGE_HEIGHT * IMAGE_WIDTH;

	const images = [];
	// Skip header bytes.
	let index = headerBytes;

	while (index < buffer.byteLength) {
    	const array = new Float32Array(recordBytes);

		for (let i = 0; i < recordBytes; i++) {
			// Normalize the pixel values into the 0-1 interval, from the original [-1, 1] interval.
      		array[i] = (buffer.readUInt8(index++) - 127.5) / 127.5;
    	}
		images.push(array);
  	}

	// This is two-dimensional array consisting of image pixel sets.
	return images;
}

/**
 * Load labels from g-zipped dataset.
 *
 * @param fileName - File name of g-zipped dataset.
 * @returns Array of labels data.
 */
async function loadLabels(fileName) {
	const buffer = await fetchAndSaveTrainingData(fileName);

	const headerBytes = LABEL_HEADER_BYTES;
	const recordBytes = LABEL_RECORD_BYTE;
	const labels = [];

	// Skip header bytes.
	let index = headerBytes;

	// Iterate until the end of the byte stream.
	while (index < buffer.byteLength) {
		// This means that each label is data type of size ${recordBytes} bytes.
		const array = new Int32Array(recordBytes);
		for (let i = 0; i < recordBytes; i++) {
			array[i] = buffer.readUInt8(index++);
		}
		labels.push(array);
	}

	// Return collected labels.
	return labels;
}

/**
 * Stage the training/testing images and labels for further use.
 */
class DatasetStager {
	/**
	 * Set the default variables.
	 */
	constructor() {
		this.dataset = [];
		this.trainSize = 0;
		this.testSize = 0;
	}

	/**
	 * Load the datasets.
	 */
	async loadData() {
		this.dataset = await Promise.all([
			loadImages(TRAIN_IMAGES_FILE), // Train image data (index 0).
			loadLabels(TRAIN_LABELS_FILE),
			loadImages(TEST_IMAGES_FILE), // Train labels data (index 2).
			loadLabels(TEST_LABELS_FILE),
		]);
		// Fetch the sizes from indices.
		this.trainSize = this.dataset[0].length;
		this.testSize = this.dataset[2].length;
	}

	/**
	 * Get training dataset.
	 * 
	 * @returns Array of training images and labels
	 */
	getTrainData() {
		return this.getData(true);
	}

	/**
	 * Get testing dataset.
	 * 
	 * @returns Array of testing images and labels
	 */
	getTestData() {
		return this.getData(false);
	}

	/**
	 * Primary data retrieval method.
	 *
	 * @param isTrainingData - Training or testing data flag.
	 */
	getData(isTrainingData) {
		const imagesIndex = (isTrainingData) ? 0 : 2;
		const labelsIndex = (isTrainingData) ? 1 : 3;

		// The amount of test or train images.
		const imagesAmount = this.dataset[imagesIndex].length;
		// 1 at the end means we have IMAGE_HEIGHT x IMAGE_WIDTH pixels represented by 1 intensity.
		const imageShape = [imagesAmount, IMAGE_HEIGHT, IMAGE_WIDTH, 1];

		// Only create one big array to hold batch of images.
		const images = new Float32Array(tf.util.sizeFromShape(imageShape));
		// 1 at the end means we have imagesAmount labels which are 0-9 integers (1 number).
		const labels = new Int32Array(tf.util.sizeFromShape([imagesAmount, 1]));

		let imageOffset = 0;
		let labelOffset = 0;

		for (let i = 0; i < imagesAmount; ++i) {
			// Add current image pixels into images superflat array.
			images.set(this.dataset[imagesIndex][i], imageOffset);
			// Add labels into label superflat array.
			labels.set(this.dataset[labelsIndex][i], labelOffset);
			// Move to the right by total amount of image pixels.
			imageOffset += IMAGE_FLAT_SIZE;
			// Move to the right by 1, since labels are numbers in range 0-9 that take 1 byte.
			labelOffset += 1;
		}

		// Convert images pixels superflat array into 4d tensor and
		// convert labels to one hot encoded arrays.
		return {
			images: tf.tensor4d(images, imageShape),
			labels: tf.oneHot(tf.tensor1d(labels, 'int32'), LABEL_FLAT_SIZE).toFloat(),
		};
	}
}

module.exports = new DatasetStager();
