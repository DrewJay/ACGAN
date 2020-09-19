<img src="https://i.postimg.cc/fTWtcmj2/tools.png" width="300">

# General Purpose Auxiliary Generative Adversarial Neural Network
Is implementation of generative neural network. Current state of the neural network excels at learning characteristics of MNIST format images and generating random variations of them. It utilizes discrete data vectorization in order to convert classes into pixels.
## Good reads
Folder good reads contains relevant scientific papers dedicated to the topic of GANs and ACGANs.
## Installation
It is strongly recommended to have global dependency `windows-build-tools` installed. Also is it is relatively complicated to sync node.js version with tfjs and tfjs-node versions. This project uses verified working combination of:

Node.js: `v10.16.3`

@tensorflow/tfjs: `v1.2.11`

@tensorflow/tfjs-node: `v1.2.11`

## Run the training
Using `node core.js` or `node core.js --gpu` if your graphics card supports CUDA computing.
You will need to download CUDA development kit from here: https://developer.nvidia.com/cuda-downloads.
