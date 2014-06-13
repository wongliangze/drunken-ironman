======================
Neural Network Library
======================

A modular neural network library with the following flexibilities:

* Number of layers

* Connection structures

* Transfer functions (linear, logistic, tanh)

* Layer cost functions (e.g. MSE, sparsity penalty)

* Param cost functions (e.g. decay weight)


And the following functionalities:

* Fine-tuning (i.e. training)

* Saving and loading

* Numerical gradient check

* Breakdown of cost, for tracking progress of training


The following have not been implemented yet:

* Greedy layer-wise pre-training

* Hyperparameter search

* Normalization

* Display

All optimization will be carried out by scipy.optimize.fmin_l_bfgs_b.


Contributors
============

This is based on work by:

* SW, DC: Original SAE

* OCC: Inheritance from sklearn; object oriented framework

* WY: stacked SAE; new connection structures; greedy layer-wise training

* Neurolab: code structure and organization



