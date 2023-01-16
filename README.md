# Prosper

In this package the neural network architectures developed by Hans-Georg Zimmermann are implemented in PyTorch.
The full documentation can be found here https://iis-scs-a.pages.fraunhofer.de/prosper/prosper/. There are also tutorials that show how to work with the package.


## Working with the repository

1. Cloning the git repository into a directory with the terminal:  
	`git clone -b hackaweek https://gitlab.cc-asp.fraunhofer.de/bkn1/prosper.git`  

2. Creating the Conda python environment from the file in the prosper directory:  
	`conda env create -f environment.yml`  

3. Let's start :-)


## Package installation

1. Open https://gitlab.cc-asp.fraunhofer.de/iis-scs-a/prosper/prosper/-/packages or in the Git repository 'Packages and Registries' -> 'Package Registry', download the package and copy it to your working directory. Then  terminal of your current Python interpreter (the interpreter you want to install the package to).
2. With a terminal of your current Python interpreter (the interpreter you want to install the package to) execute the following terminal command (insert the corresponding version number):
	`pip install prosper_nn-X.X.X-py3-none-any.whl`
3. The Prosper package will now be installed to your environment

## Package versioning

- Package version is in the format: x.y.z
- Increase version when something is merged/committed in the master branch and you changed the prosper package itself:
	- increase z, if bugfix or smaller changes
	- increase y, if you add new features
	- increase x, if your code is not compatible with the old one
- If your changes do not affect the package (e.g. changes in tutorial) delete the current package from the package registry (https://gitlab.cc-asp.fraunhofer.de/iis-scs-a/prosper/prosper/-/packages)

## Directory structure

- examples (this directory contains example use cases)
	- ecnn_example.py  
	- hcnn_example.py  

- docs/source/tutorials (tutorial notebooks)
	- Regression Flow.ipynb
	- HCNN_Flow.ipynb

- prosper (from this the package is created)
	- models (all model architectures are located here)
		- ecnn
		- hcnn
		- ensemble

	- utils (collection of support functions for evaluation and visualisation)
		- neuron_correlation_hidden_layer.py
		- sensitivity_analysis



## Visualization of Neural Networks with Tensorboard

1. Define neural network

2. Define SummaryWriter and as an argument give a folder where to store the files:  
	`writer = SummaryWriter('runs/tensorboard_test')`
	
3. Add Graph to Writer. The variable 'network' is the torch model and the variable 'X' is a set of matching data:  
	`writer.add_graph(network, X)`
	
4. Add scalar information to Tensorbaord by logging it to writer. For example the training loss during training:  
`for epoch in epochs:` \
    `    writer.add_scalar('Loss/train', loss, epoch)`

5. Add a histograms of a matrix or a vector:  
	`writer.add_histogram('hidden_layer0', net.hidden_layer0.weight)`

6. Close writer object:  
	`writer.close()`
	
7. Start Tensorbaord in terminal:  
	6.1 Move to right folder  
	6.2 Activate environment:  
		`conda activate prosper`  
	6.3 Start Tensorboard:  
		`tensorboard --logdir runs/`  
		
8. Show results in browser with the link in the terminal
