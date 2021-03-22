#%% Imports
import sys
sys.path.append("..")
from SynapticCacheNetwork import SynapticCacheNetwork, Activation, load_mnist
import numpy as np
import matplotlib.pyplot as plt

#%% Data loading
x_train, y_train, x_test, y_test, n_samples, n_labels, img_size = load_mnist(True)

#%% Network profile
layers = [img_size, 50, n_labels]
learning_rate = 0.05
activation_function = Activation("sigmoid")
eLTP_upkeep = 0.001
decay_rate = 0
scheme = 2
#threshold = varies

#%% Search 
"""
    This is a personally developed optimiser algorithm, which I call a 'zoom
    optimiser'. It notes when a turning point happens, creates a new range of 
    values between before and after the turning point and iteratively 'zooms'
    in on the turning point. It works particularly well for parabolic functions.
    While I'm sure someone has invented/discovered something like this before
    I don't know what it is or would be called and would appreciate if anyone 
    reading this that does know, could point me in its direction.
"""

# Starting optimal threshold guess range
# n_spaces = 10
# search_space = np.linspace(0.0007, 0.008, n_spaces)
# energies = []
# Search depth
max_depth = 7

best_thresholds = []
# for learning_rate in [0.005, 0.006]:
best_threshold = 0.0
n_spaces = 10
search_space = np.logspace(-1, 1, n_spaces)
energies = []
print("Learning Rate: ", learning_rate)
for depth in range(0, max_depth):
    i = 0
    increasing = False
    energies = []
    print("Depth: ", depth)
    for threshold in search_space:
        print("Threshold: ", search_space[i])
        np.random.seed(1)
        network = SynapticCacheNetwork(layers, activation_function, learning_rate, eLTP_upkeep, decay_rate, scheme, threshold)
        epochs, accuracy, energy = network.train_until(93, x_train, y_train, 4)
        energies.append(energy[-1])
        if i >= 2:
            if energies[-1] < energies[-2]:
                increasing = False
            else:
                increasing = True 
        if increasing:
            best_threshold = search_space[i-1]
            print("Best Threshold: ", search_space[i-1])
            search_space = np.linspace(search_space[i-2], search_space[i], n_spaces)
            print("Search Space: ", search_space)
            break
        i = i + 1
best_thresholds.append(best_threshold)
    
# with open('best_thresholds_netnet.csv', 'ab') as file:
#       np.savetxt(file, best_thresholds, encoding='utf8', delimiter=',')

print(best_thresholds)

"""
    Optimal Thresholds for Network Profiles:
        network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.005, 0.001, 0, 2, 0.004008230452674897)
        network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.01, 0.001, 0, 2, 0.007849108367626886) @ 90% accuracy
        NOTE: The threshold decreases with higher accuracy
        network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.01, 0.001, 0, 2, 0.007081527394386207) @ 95% accuracy | depth: 7
        network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.01, 0.001, 0, 8, 0.006393514990375226) @ 95% accuracy | depth: 7

        network = SynapticCacheNetwork([784, 50, 10], Activation("sigmoid"), 0.001, 0.001, 0, 2, 0.0007818333549726122) @ 90% accuracy | depth: 7
"""
    

    
    
