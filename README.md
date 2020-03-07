# K-NN
A simple implementation of KNN in numpy, python 3. 
The file knn.py contains the class defining the K-nearest neighbors model. 
The notebook main.ipynb test the model by using the iris dataset. 

Recall that the $k$-nn is a non-parametric classification model. The idea is, given a data point $x$, find the $k$ closest points $x$ and then label $x$ to the class containing the majority of these $k$ datapoints. 

It can be done using any kind of distance, but in this implementation, we use the Euclidean (or $L_2$) norm. The Euclidean distance of two points $x = (x_1,...,x_n)$ and $y = (y_1, ..., y_n)$ in $\mathbb{R}^n$ is defined by:
$$ d(x,y) = \sqrt{\sum_{i=1}^{n} (x_i-y_i)^2 .$$

