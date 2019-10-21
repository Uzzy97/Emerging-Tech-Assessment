from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np


digits = datasets.load_digits()
features = digits.data
labels = digits.target

clf = SVC(gamma = 0.001)
clf.fit(features, labels)




img = misc.imread("tests/7.jpg")
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
print(features[-1])
print(img)