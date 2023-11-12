# Author: 
# Date:
# Project: 
# Acknowledgements: 
#


from typing import Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('AllData.csv', delimiter=';',dtype = object)
data = df.to_numpy(dtype=float)
targets = data[:,0]
classes = np.unique(targets, return_counts=False)
features = data[:,1:]
print(features)


def split_train_test(features: np.ndarray, targets: np.ndarray,
    train_ratio:float=0.8) -> Union[tuple, tuple]:
    '''
    Shuffle the features and targets in unison and return
    two tuples of datasets, first being the training set,
    where the number of items in the training set is according
    to the given train_ratio
    '''
    p = np.random.permutation(features.shape[0])
    features = features[p]
    targets = targets[p]

    split_index = int(features.shape[0] * train_ratio)

    train_features, train_targets = features[0:split_index, :],\
    targets[0:split_index]
    test_features, test_targets = features[split_index:-1, :],\
        targets[split_index: -1]

    return (train_features, train_targets), (test_features, test_targets)

def prior(targets,classes):
    class_counts = np.zeros(len(classes))
    for i, class_type in enumerate(classes):
        class_counts[i] = np.sum(np.array(targets) == class_type)
    return class_counts/len(targets)
# print(prior(targets,classes))

def split_data(
    features,
    targets: np.ndarray,
    split_feature_index: int,
    theta: float
) -> Union[tuple, tuple]:
    '''
    Split a dataset and targets into two seperate datasets
    where data with split_feature < theta goes to 1 otherwise 2
    '''
    mask_1 = features[:, split_feature_index] < theta
    mask_2 = ~mask_1  # Inverse of mask_1
    # Apply the masks to split the data and targets
    features_1 = features[mask_1]
    targets_1 = targets[mask_1]
    
    features_2 = features[mask_2]
    targets_2 = targets[mask_2]

    return (features_1, targets_1), (features_2, targets_2)
(f_1, t_1), (f_2, t_2) = split_data(features, targets, 8, 0.09)
# print((f_2[:,8], t_2))
def gini_impurity(targets: np.ndarray, classes: list) -> float:
    '''
    Calculate:
        i(S_k) = 1/2 * (1 - sum_i P{C_i}**2)
    '''
    return (1-np.sum(prior(targets,classes)**2))/2
# print(gini_impurity(t_2,classes))


def weighted_impurity(
    t1: np.ndarray,
    t2: np.ndarray,
    classes: list
) -> float:
    '''
    Given targets of two branches, return the weighted
    sum of gini branch impurities
    '''
    g1 = gini_impurity(t1, classes)
    g2 = gini_impurity(t2,classes)
    n = t1.shape[0] + t2.shape[0]
    return t1.shape[0]*g1/n +t2.shape[0]*g2/n
# print(weighted_impurity(t_1,t_2,classes))


def total_gini_impurity(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    split_feature_index: int,
    theta: float
) -> float:
    (f_1, t_1), (f_2, t_2) = split_data(features, targets, split_feature_index, theta)
    return weighted_impurity(t_1,t_2,classes)


def brute_best_split(
    features: np.ndarray,
    targets: np.ndarray,
    classes: list,
    num_tries: int
) -> Union[float, int, float]:
    '''
    Find the best split for the given data. Test splitting
    on each feature dimension num_tries times.

    Return the lowest gini impurity, the feature dimension and
    the threshold
    '''
    best_gini, best_dim, best_theta = float("inf"), None, None
    # iterate feature dimensions
    for i in range(features.shape[1]):
        # create the thresholds
        min_val, max_val = np.min(features[:,i]), np.max(features[:,i])
        thetas = np.linspace(min_val, max_val, num_tries)
        # iterate thresholds

        for theta in thetas:
            gini = total_gini_impurity(features,targets,classes,i,theta)
            if gini < best_gini:
                best_gini = gini
                best_dim = i
                best_theta = theta
    return best_gini, best_dim, best_theta
# print(brute_best_split(features, targets, classes, 30))


class ComposerClassTrainer:
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        classes: list = [0, 1, 2], 
        train_ratio: float = 0.8
    ):
        '''
        train_ratio: The ratio of the dataset that will
        be dedicated to training.
        '''
        (self.train_features, self.train_targets),\
            (self.test_features, self.test_targets) =\
            split_train_test(features, targets, train_ratio)

        self.classes = classes
        self.tree = DecisionTreeClassifier(max_depth=3)

    def train(self):
        self.tree.fit(self.train_features, self.train_targets)

    def accuracy(self):
        # Make predictions on the test data
        predictions = self.tree.predict(self.test_features)
        # Calculate accuracy by comparing predicted labels to true labels
        acc = accuracy_score(self.test_targets, predictions)
        return acc
    def plot(self):
        plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
        plot_tree(self.tree, filled=True,class_names=["Bach","Beethoven","Mozart"],feature_names=list(df.columns[1:]))
        plt.show()
    def guess(self):
        # Make predictions on the test data features
        predictions = self.tree.predict(self.test_features)
        return predictions
    def confusion_matrix(self):
        # Make predictions on the test data features
        predictions = self.tree.predict(self.test_features)
        
        # Initialize the confusion matrix
        num_classes = len(self.classes)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        
        true_labels = [int(label) for label in self.test_targets]
        predicted_labels = [int(label) for label in predictions]

    # Populate the confusion matrix
        for true_label, predicted_label in zip(true_labels, predicted_labels):
            confusion_matrix[true_label][predicted_label] += 1
        
        return confusion_matrix
dt = ComposerClassTrainer(features, targets, classes=classes)
dt.train()
print(classes)
print(f'The accuracy is: {dt.accuracy()}')
dt.plot()
print(f'I guessed: {dt.guess()[:10]}')
print(f'The true targets are: {dt.test_targets}')
print(dt.confusion_matrix())

