from sklearn.feature_selection import VarianceThreshold

class FeatureSelection():
  def varianceTrashHold(self, X):
     sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
     return sel.fit_transform(X)
     