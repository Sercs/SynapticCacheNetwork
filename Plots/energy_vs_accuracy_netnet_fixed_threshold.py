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
learning_rates = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
activation_function = Activation("sigmoid")
eLTP_upkeep = 0.001
decay_rate = 0
scheme = 8

energies_all = []
accuracies_all = []

#%% Network profile
for learning_rate in learning_rates:
    np.random.seed(1)
    network = SynapticCacheNetwork(layers, activation_function, learning_rate, eLTP_upkeep, decay_rate, scheme, 0.006393514990375226)      
    #%% Training
    epochs, accuracies, energies = network.train_until(95, x_train, y_train, n_samples_in_batch)
    energies_all.append(energies)
    accuracies_all.append(accuracies)
    
with open('fig_energy_vs_accuracy_netnet_fixed_threshold.csv', 'w') as file:
    for i in range(len(accuracies_all)):
        for j in range(len(accuracies_all[i])):
            file.write(str(accuracies_all[i][j]))
            file.write(",")
        file.write("\n")
        for j in range(len(energies_all[i])):
            file.write(str(energies_all[i][j]))
            file.write(", ")
        file.write("\n") 

for i in range(len(accuracies_all[i])):
    plt.plot(accuracies_all[i], energies_all[i])
plt.show()    
    