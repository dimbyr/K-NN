import numpy as np

class KNN(object):
    def __init__(self, k= 5):
        self.k = k
        self.data = None
        self.labels = None
        
    def distances(self, x, datapoints):
        '''
        Euclidean distance
        '''
        n = len(datapoints)
        d = [0]*n
        for i in range(n):
            d[i] = np.linalg.norm(x-datapoints[i])
        return np.array(d)
    
    def kneighbors(self, x, datapoints):
        """
        Get the indices of the k nearest neigbors of x
        """
        sorted_neighbors = np.argsort(self.distances(x, datapoints))
        r= [sorted_neighbors[i] for i in range(self.k)]
        return list(r)

    def fit(self, features, target):
        '''
        features: nDarray
        target : labels of the data, ndarray
        '''
        self.data = features
        self.labels = target
        print(f'Info: \n k-NN with  k = {self.k} and {len(self.data)} datapoints and {len(set(self.labels))} labels')
        
    def predict(self, new_x):
        unique_labels = list(set(self.labels))
        pred = [self.labels[0]]*len(new_x)
        for i  in range(len(new_x)):
            dx = self.kneighbors(new_x[i], self.data)
            labels = list(self.labels[dx])
            nearest_labels_count = [labels.count(c) for c in unique_labels]
            pred[i] = unique_labels[np.argmax(nearest_labels_count)]
        return np.array(pred)
