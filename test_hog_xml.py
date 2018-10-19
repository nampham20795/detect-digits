import cv2
import numpy as np
from skimage.feature import hog
img = cv2.imread("test2.jpg")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((5,5),np.float32)/25
dilation = cv2.dilate(thresh,kernel,iterations = 8)
img2, contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(contours)):
    cnt = contours[i]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cut_img = img[y:y+h,x:x+w]
    cut_img = cv2.resize(cut_img,(40, 40), interpolation = cv2.INTER_CUBIC)
    X_test_feature = []
    feature = [hog(np.array(cut_img),orientations=8,pixels_per_cell=(16,16),cells_per_block=(1,1),block_norm="L1")]
    X_test_feature = np.array(feature).astype(np.float32)
    svm = cv2.ml.SVM_load('E:/TraindataWithSVM/svm_hog_traindata.xml')
    test = svm.predict(X_test_feature)
    if (test[1][0][0] == 0):
        cv2.putText(img, "0", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 1):
        cv2.putText(img, "1", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 2):
        cv2.putText(img, "2", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 3):
        cv2.putText(img, "3", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 4):
        cv2.putText(img, "4", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 5):
        cv2.putText(img, "5", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 6):
        cv2.putText(img, "6", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 7):
        cv2.putText(img, "7", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 8):
        cv2.putText(img, "8", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if (test[1][0][0] == 9):
        cv2.putText(img, "9", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
        

cv2.imshow("frame",img)
cv2.waitKey(0)
