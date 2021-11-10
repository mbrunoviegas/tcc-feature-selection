from sklearn.feature_selection import VarianceThreshold
from CFSmethod.CFS import cfs
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def varianceTrashHold(X):
    sel = VarianceThreshold(threshold=(.7 * (1 - .7)))
    return sel.fit_transform(X)

def correlationBasedFeature(X, y):
  return cfs(X, y)

def univariateFeatureSelection(X, y):
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    return X_new