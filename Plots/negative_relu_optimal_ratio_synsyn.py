#%% Imports
import sys
sys.path.append("..")
from SynapticCacheNetwork import SynapticCacheNetwork, Activation, load_mnist
import numpy as np
import matplotlib.pyplot as plt

#%% Data loading
x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = load_mnist(True)

n_samples_in_batch = 4
layers = [img_size, 50, n_labels]
#learning_rate = varies
learning_rate = 0.01
threshold = 0.00745244031479191
#activation_function = varies
eLTP_upkeep = 0.001
decay_rate = 0
scheme = 2

energies_all = []
accuracies_all = []

ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_energy = []
ratio_accuracy = []

#%% Network profile
for ratio in ratios:
    np.random.seed(1)
    network = SynapticCacheNetwork(layers, Activation("mixed relu", ratio), learning_rate, eLTP_upkeep, decay_rate, scheme, threshold)      
    #%% Training
    accuracies, energies = network.train(x_train, y_train, 100, n_samples_in_batch)
    ratio_energy.append(energies[-1])
    ratio_accuracy.append(accuracies[-1])
    plt.plot(accuracies, energies)
    plt.gca().set_xlim(left=90)
    plt.show()
    
print(ratios[np.argmin(ratio_energy)])
print(ratios[np.argmax(ratio_accuracy)])