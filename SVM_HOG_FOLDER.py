import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import hog

img = cv2.imread("2.JPG")

X_train_feature = []
for i in range(10):
    for j in range(200):
           path = "E:/TraindataWithSVM/traindata/%d/n-%d.jpg" % (i,j)
           #print path
           im = cv2.imread(path)
           im = cv2.resize(im,(40, 40), interpolation = cv2.INTER_CUBIC)
           feature = hog(np.array(im),orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),block_norm="L1")
           X_train_feature.append(feature)
X_train_feature = np.array(X_train_feature).astype(np.float32)
print X_train_feature.shape
X_test_feature = []

img = cv2.resize(img,(40, 40), interpolation = cv2.INTER_CUBIC)
feature1 = [hog(np.array(img),orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),block_norm="L1")]
X_test_feature = np.array(feature1).astype(np.float32)
k = (np.repeat(np.arange(10),200)[:,np.newaxis])
svm = cv2.ml.SVM_create()
svm.train(X_train_feature,cv2.ml.ROW_SAMPLE,k)
svm.save('E:/TraindataWithSVM/svm_hog_traindata.xml')
test = svm.predict(X_test_feature)
print test[1][0][0]



