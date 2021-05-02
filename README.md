# Affinity Propagation

Affinity propagation is a clustering algorithm introduced by [Frey and Dueck (2007)](https://science.sciencemag.org/content/315/5814/972) in which observations are grouped together by passing messages to each other. Those observations who can serve as exemplars for others will eventually become the cluster centers.

The `affprop` package provides an implementation of that algorithm! Find out more about the algorithm by reading Frey and Dueck's paper, our term paper (in this repository) or by browsing the examples in the Jupyter Notebooks in this repository.

## Installation
Installation of the package is straightforward:
```
pip install affprop
```
That's it!

## Basic Usage
```
# Example
import affprop as ap
import matplotlb.pyplot as plt
from sklearn import datasets

# Create the data and plotting it for inspection
data, labels = datasets.make_blobs(n_samples=250, n_features=2, centers=4, center_box=(-20,20), random_state=42)
plt.scatter(data[:,0],data[:,1], c=labels)
plt.show()

# Perform affinity propagation
exemplars, labels, centers = ap.affinity_prop(data, preference="min")

# Plot the result
ap.cplot(data,labels)
