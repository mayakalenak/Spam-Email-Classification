'''knn.py
K-Nearest Neighbors algorithm for classification
Maya Kalenak
CS 251: Data Analysis and Visualization
Fall 2024
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from classifier import Classifier

class KNN(Classifier):
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes):
        '''KNN constructor

        TODO:
        - Call superclass constructor
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

        super().__init__(num_classes)

    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where 
        training samples have corresponding class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). 
        Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). 
        Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that
        the classifier memorizes the training data.
        '''
        self.exemplars = data
        self.classes = y

    def predict(self, data, k):
        '''Use the trained KNN classifier to predict the class label 
        of each test sample in `data`.
        Determine class by voting: find the closest `k` 
        training exemplars (training samples) and
        the class is the majority vote of the classes of these 
        training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). 
        Data to predict the class of.
        Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points
        around each test sample used to make class predictions. 
        In other words, how many training samples vote to determine the
        predicted class of a nearby test sample.

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). 
        Predicted class of each test data sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, 
        count up how many belong to which class.
        - The predicted class of the test sample is the majority vote.
        '''
        predicted = []
        predict_class = np.array((data.shape[0],))
        for i, test_sample in enumerate(data):
            distances = np.sqrt(np.sum((self.exemplars-test_sample) **2, axis=1))
            sorted_idx_dist = np.argsort(distances)

            k_nearest_idx = sorted_idx_dist[: k]
            k_nearest_labels = self.classes[k_nearest_idx]

            vote_count_labels = np.bincount(k_nearest_labels.astype(int))
            predict = np.argmax(vote_count_labels)
            predict_class[i:,] = predict

            predicted.append(predict)

        return np.array(predicted)

    def plot_predictions(self, k, n_sample_pts):
        '''Paints the data space in colors corresponding to which class
        the classifier would hypothetically assign to data samples 
        appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points
        around each test sample used to make class predictions. 
        In other words, how many training samples vote to determine the
        predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along
            the x and y axes to plug into KNN at which we are determining
            the predicted class. Think of this as regularly spaced 2D 
            "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, 
        like in the clustering project, to use either the Okabe & Ito or
        one of the Petroff color palettes: 
        https://github.com/proplot-dev/proplot/issues/424
        - Wrap your colors list as a `ListedColormap` object 
        (already imported above) so that matplotlib can parse it.
        - Make an ndarray of length `n_sample_pts` of regularly spaced points 
        between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get 
        the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a 
        single ndarray and reshape it so that you can plug it in as your
        `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts).
            You want to make your input to self.predict of 
            shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format 
        for plotting in 2D. shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. 
        Use the `cmap` optional parameter to specify your
        discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        colors = [
            "#1845fb", "#ff5e02", "#c91f16", "#c849a9", "#adad7d", "#86c8dd",
            "#578dff", "#656364"]
        cmap = ListedColormap(colors)

        fake_data = np.linspace(self.exemplars.min(),self.exemplars.max(),
                                n_sample_pts)

        samp_vec = fake_data.reshape((fake_data.shape[0], 1))

        x, y =  np.meshgrid(samp_vec, samp_vec)

        xx = x.reshape((n_sample_pts*n_sample_pts, 1))
        yy = y.reshape((n_sample_pts*n_sample_pts, 1))

        xy = np.hstack((xx, yy))

        predicted = self.predict(xy, k)
        predicted = np.array(predicted).reshape((n_sample_pts, n_sample_pts))

        plt.pcolormesh(x, y, predicted, cmap=cmap, shading='auto')
        plt.colorbar()
        plt.title("KNN Landscape")
        plt.show()
