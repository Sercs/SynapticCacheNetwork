"""
    Used to test certain configurations.
"""

#%% Imports
import sys
sys.path.append("..")
from SynapticCacheNetwork import SynapticCacheNetwork, Activation, load_mnist
import numpy as np
import matplotlib.pyplot as plt

#%% Data loading
x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = load_mnist(True)

#%% Network profile

np.random.seed(1)
network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.001, 0.001, 0, 2, 0.0007849108367626886)
        
#%% Training
n_epochs = 10
n_samples_in_batch = 4

epochs, accuracies, energies = network.train_until(95, x_train, y_train, n_samples_in_batch)