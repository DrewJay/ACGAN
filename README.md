<img src="https://i.postimg.cc/fTWtcmj2/tools.png" width="300">

# Vectorr
Is implementation of "general purpose auxiliary generative adversarial neural network" (ACGANN). Current state of the neural network excels at learning characteristics of MNIST format images and generating random variations of them. It utilizes discrete data vectorization in order to convert classes into pixels.
## Credits
File core.js that contains primary ACGANN implementations is heavily based on work of <a href="https://github.com/tensorflow/tfjs-examples/tree/master/mnist-acgan">Google</a>. My implementation includes refactored and prettyfied code and highly descriptive documentation.
## Good reads
Folder good reads contains relevant scientific papers dedicated to the topic of GANNs and ACGANNs.
## Installation
It is *very* complicated to install this thing properly. It is strongly recommended to have global dependency `windows-build-tools` installed. Also is it is relatively complicated to sync node.js version with tfjs and tfjs-node versions. This project uses verified working combination of:

Node.js: `v10.16.3`

@tensorflow/tfjs: `v1.2.11`

@tensorflow/tfjs-node: `v1.2.9`

@tensorflow/tfjs-node-gpu: `v1.2.9`

### GPU Training
To run this application on GPU, one needs to do all the steps above plus the ones to be mentioned in this section. You will very likely run into number of errors during the installation, so I recommend to <a href="https://github.com/tensorflow/tfjs/issues/2003">pay attention to this github issue convering most of them</a>.

<a href="https://www.tensorflow.org/install/gpu#windows_setup">This link</a> is also very helpful.

You will first need to enable CUDA computing on your graphics card. You can achieve it by downloading cuDNN >= `7.4.1` from <a href="https://developer.nvidia.com/rdp/cudnn-download">here</a>.

Then you will need to install CUDA toolkit v `10.0.0` <a href="https://developer.nvidia.com/cuda-downloads">from here</a>.

Finally, you will need to set environment variables into user path:

C:\tools\cuda\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\libx64
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include

This should get you covered and training should now be possible on gpu by running `node core.js --gpu`.

## Run the training
Using `node core.js` or `node core.js --gpu` if your graphics card supports CUDA computing.
You will need to download CUDA development kit from here: https://developer.nvidia.com/cuda-downloads.
