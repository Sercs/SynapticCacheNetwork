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
optimal_thresholds = [1.026837204846203647e-03, 2.107876294614544413e-03, 2.689174257258788395e-03, 3.496073935000621781e-03, 4.200233281680548075e-03, 4.196715233832555916e-03, 4.887815568509147959e-03, 6.089919249951673136e-03, 6.399257239593231748e-03, 7.236742491954264653e-03]
activation_function = Activation("sigmoid")
eLTP_upkeep = 0.001
decay_rate = 0
scheme = 8

energies_all = []
accuracies_all = []

#%% Network profile
for i in range(len(learning_rates)):
    learning_rate = learning_rates[i]
    optimal_threshold = optimal_thresholds[i]
    np.random.seed(1)
    network = SynapticCacheNetwork(layers, activation_function, learning_rate, eLTP_upkeep, decay_rate, scheme, optimal_threshold)      
    #%% Training
    epochs, accuracies, energies = network.train_until(95, x_train, y_train, n_samples_in_batch)
    energies_all.append(energies)
    accuracies_all.append(accuracies)
    
with open('fig_energy_vs_accuracy_netnet_optimal_thresholds.csv', 'w') as file:
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
    