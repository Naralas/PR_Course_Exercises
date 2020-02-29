"""
Pattern Recognition Course MCS 2020
K_NN implementation by hand
Herbelin Ludovic

Usage : You will need the following modules to run it : scipy, tqdm, numpy

I could not use all of the dataset since it would take too much time to compute.
At first, I did the distance computation by hand as well but it was way too slow and using the cdist function from scipy already
improved the performance by ~2.

The next step would be to instead of computing sample by sample, compute the cdist between the test set and the train set (as matrices),
using scipy cdist as well and then only retrieving the distance each time.
"""


from math import pow, sqrt
from collections import Counter
import numpy as np
import scipy.spatial as sp
from tqdm import tqdm
import time
import random


# don't train with the whole dataset because it takes too much time
N_SAMPLES = 1500
N_NEIGHBORS = 6


# distance functions were done by hand at first and then I used scipy's cdist which already implement them
"""distance_functions = {
    'euclidian': lambda s_1, s_2: sqrt((s_1 - s_2)**2),
    'manhattan': lambda s_1, s_2: abs(s_1 - s_2),
}"""

class Sample:
    def __init__(self, label, feature):
        self.label = label
        self.feature = feature

"""
"Condense" the dataset, only finding the instances that are at the frontiers
Classify each instance with 1-NN, if it was classified wrong, add it to the important samples set
"""
def condense(samples):
    MIN_CHANGES = 4

    raw_set = samples.copy()

    r_set = [raw_set.pop()]
    n_changes = MIN_CHANGES

    # replaced while there was at least one change with a threshold for testing
    while n_changes >= MIN_CHANGES:
        n_samples_before = len(raw_set)

        # loop through in reverse so we can remove elements 
        for i in tqdm(range(len(raw_set) -1, -1, -1)):
            xi = raw_set[i]
            # classify with 1-NN
            predicted_class = k_nn(samples_set=r_set, new_instance=xi, nb_neighbors=1, f_distance='euclidean')
            
            if predicted_class != xi.label:
                r_set.append(xi)
                del raw_set[i]

        n_changes = n_samples_before - len(raw_set)
        print(f"Removed {n_samples_before - len(raw_set)} instances for this iteration")
        print(f"Total of important instances found : {len(r_set)}")
    
    return r_set


"""
Reduce the dataset, try to find the outliers
Classify with 3-NN, if it was classified wrong, remove it
"""
def edit(samples):
    # list of indices which mark which samples need to be removed
    marked_samples_indices = []
    for i in tqdm(range(0, len(samples))):
        xi = samples[i]
        predicted_class = k_nn((sample for sample in samples if sample != xi), xi, 3, f_distance='euclidean')
        if predicted_class != xi.label:
            marked_samples_indices.append(i)

    return [samples[i] for i in range(0, len(samples)) if i not in marked_samples_indices]
        

"""
K_NN classify instance method
By default the distance function is the cosine
"""
def k_nn(samples_set, new_instance, nb_neighbors, f_distance='cosine'):
    sample_distances = {}

    # compute the time elapsed when computing the distances
    #dist_time_start = time.time()
    
    for sample in samples_set:
        # compute the distance between each element of the matrix A and B
        sample_distances[sample] = sp.distance.cdist(sample.feature, new_instance.feature, metric=f_distance)

    """dist_time_end = time.time()
    print(f"Distances computation time : {dist_time_end - dist_time_start}.")
    #0.25-0.3s for 500 samples => this is the bottleneck"""
    
    """minima_time_start = time.time()"""

    # order the distances samples pairs by the minimal distance
    ordered_distances = {key: value for key,value in sorted(sample_distances.items(), key=lambda entry:entry[1])}
    # get the n first with the smallest distance
    n_nearest = {key: ordered_distances[key] for key in list(ordered_distances.keys())[:nb_neighbors]}

    class_counter = Counter((sample.label for sample in n_nearest))

    # in case of conflict, always get the first one in the counter, maybe needs modified
    most_common_class = class_counter.most_common()[0][0]

    """minima_time_end = time.time()
    print(f"Most common class computation time : {minima_time_end - minima_time_start}")"""

    return most_common_class

def read_dataset(filepath):
    with open(filepath, 'r') as file:
        dataset = file.readlines()
        return dataset

def prepare_dataset(dataset):
    # remove space, CR, ... and split into columns
    dataset = (sample.strip().split(',') for sample in dataset)

    # format the set as (data, class) for each sample, reshape to [[data]] to use scipy cdist function
    dataset = [Sample(instance[0], np.array(instance[1:]).reshape(1, -1)) for instance in dataset]
    
    # shuffle so we don't train with specific data distribution
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    train_set = read_dataset('data/train.csv')[:N_SAMPLES]
    train_set = prepare_dataset(train_set)

    test_set = read_dataset('data/test.csv')[:N_SAMPLES]
    test_set = prepare_dataset(test_set)

    print("Condensing dataset...")
    train_set = condense(train_set)
    print("Editing dataset...")
    train_set = edit(train_set)
    

    print(f"Raw set length : {N_SAMPLES}, reduced set length : {len(train_set)}")


    print(f"Testing...")
    predicted_right = 0
    for sample in tqdm(test_set):
        prediction = k_nn(train_set, sample, nb_neighbors=N_NEIGHBORS, f_distance='cosine')
        if prediction == sample.label:
            predicted_right += 1
    
    print(f"Accuracy of the system with {len(train_set)} samples and {N_NEIGHBORS} neighbors : {predicted_right / len(test_set) * 100} %")
    # using 1500 samples, editing and condensing -> 190 samples, 6-NN : ~65%
    # condensing and editing seems to remove too many samples as when tested with 500 samples and 6-NN : ~80%

