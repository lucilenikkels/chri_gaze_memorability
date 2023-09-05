# AEON Time Series Classifiers

## Dataformat

```python
# 2D array example (univariate)
# Two samples, one channel, seven series length
X = [[1, 2, 3, 4, 5, 6, 7],  
     [4, 4, 4, 5, 6, 7, 3]]

# 3D array example (multivariate)
# Two samples, two channels, four series length
X2 = [[[1, 2, 3, 4], [3, 8, 3, 8]],  
      [[5, 2, 1, 5], [3, 8, 3, 8]]]  

# class labels for each sample
y = [0, 1] 
```

## Example classifier with DTW-KNN

```python
import numpy as np
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

X = np.array(X)

y = np.array(y)

clf = KNeighborsTimeSeriesClassifier(distance="dtw")

clf.fit(X, y)  # fit the classifier on train data
```

KNeighborsTimeSeriesClassifier()

```python
X_test = np.array([[2, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4]])

y_pred = clf.predict(X_test)  # make class predictions on new data
```

[0 1]
