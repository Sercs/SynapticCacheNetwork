"""
    SynapticCacheExample
"""
#%% Imports
import sys
sys.path.append("..")
from SynapticCacheNetwork import SynapticCacheNetwork, Activation, load_mnist
import matplotlib.pyplot as plt

#%% Data loading
x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = load_mnist(True)

#%% Network profile
layers = [img_size, 50, n_labels]
learning_rate = 0.01
activation_function = Activation("sigmoid")
eLTP_upkeep = 0.001
decay_rate = 0
scheme = 2
threshold = 0.0078

network = SynapticCacheNetwork(layers, activation_function, learning_rate, eLTP_upkeep, decay_rate, scheme, threshold)

#%% Training
n_epochs = 10
n_samples_in_batch = 4

accuracies, energies = network.train(x_train, y_train, n_epochs, n_samples_in_batch)

#%% Plotting
plt.plot(accuracies, energies)
