from math import pow, sqrt
from collections import Counter
import numpy as np
import scipy.spatial as sp
from tqdm import tqdm



# don't train with the whole dataset because it takes too much time
N_SAMPLES = 500
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

def condense(samples):
    MIN_CHANGES = 5
    raw_set = samples.copy()

    r_set = [raw_set.pop()]
    n_changes = MIN_CHANGES

    # replace while there was at least one change with a threshold so I don't spend 10 minutes getting 1 sample changed each time
    while n_changes >= MIN_CHANGES:
        n_samples_before = len(raw_set)

        # loop through in reverse so we can remove elements 
        for i in tqdm(range(len(raw_set) -1, -1, -1)):
            xi = raw_set[i]
            # classify with 1-NN
            predicted_class = k_nn(samples_set=r_set, new_instance=xi, nb_neighbors=1)
            
            if predicted_class != xi.label:
                r_set.append(xi)
                del raw_set[i]

        n_changes = n_samples_before - len(raw_set)
        print(f"Nb of changes : {n_samples_before - len(raw_set)}")
    
    return r_set

def edit(samples):
    # list of indices which mark which samples need to be removed
    marked_samples_indices = []
    for i in tqdm(range(0, len(samples))):
        xi = samples[i]
        predicted_class = k_nn((sample for sample in samples if sample != xi), xi, 3)
        if predicted_class != xi.label:
            marked_samples_indices.append(i)

    return [samples[i] for i in range(0, len(samples)) if i not in marked_samples_indices]
    #return (sample for sample, marking in marked_samples if marking != True)
        

"""
K_NN classify instance method
By default the distance function is the cosine
"""
def k_nn(samples_set, new_instance, nb_neighbors, f_distance='cosine'):
    sample_distances = {}
    for sample in samples_set:
        sample_distances[sample] = sp.distance.cdist(sample.feature, new_instance.feature, metric=f_distance)
    
    # order the distances samples pairs by the minimal distance
    ordered_distances = {key: value for key,value in sorted(sample_distances.items(), key=lambda entry:entry[1])}
    # get the n first with the smallest 
    n_nearest = {key: ordered_distances[key] for key in list(ordered_distances.keys())[:nb_neighbors]}

    class_counter = Counter((sample.label for sample in n_nearest))

    # in case of conflict, always get the first one in the counter, maybe needs modified
    most_common_class = class_counter.most_common()[0][0]

    return most_common_class

def read_dataset(filepath):
    with open(filepath, 'r') as file:
        dataset = file.readlines()
        return dataset

def prepare_dataset(dataset):
    dataset = (sample.strip().split(',') for sample in dataset)
    # format the set as (data, class) for each sample
    dataset = [Sample(instance[0], np.array(instance[1:]).reshape(1, -1)) for instance in dataset]
    return dataset

if __name__ == "__main__":
    train_set = read_dataset('data/train.csv')[:N_SAMPLES]
    train_set = prepare_dataset(train_set)

    test_set = read_dataset('data/test.csv')[:N_SAMPLES]
    test_set = prepare_dataset(test_set)

    print("Condensing dataset...")
    #train_set = condense(train_set)
    print("Editing dataset...")
    train_set = edit(train_set)
    

    print(f"Raw set length : {N_SAMPLES}, reduced set length : {len(train_set)}")

    #condensed_samples = train_set

    predicted_right = 0

    print(f"Testing...")
    for sample in tqdm(test_set):
        prediction = k_nn(train_set, sample, nb_neighbors=N_NEIGHBORS, f_distance='cosine')
        if prediction == sample.label:
            predicted_right += 1
    
    print(f"Accuracy of the system with {N_SAMPLES} samples and {N_NEIGHBORS} neighbors : {predicted_right / len(test_set) * 100} %")
    

