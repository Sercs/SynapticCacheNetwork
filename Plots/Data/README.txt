These plots were generated with a slight error in the code.
The weight changes were not applied by the accumulated gradient,
and instead the code only applied the gradient of a single 
sample in the batch. Therefore this used mini-batches of a 
single sample across 1/4 of the dataset. (mini-batch of 4.)

This does not change the interpretation of the data.