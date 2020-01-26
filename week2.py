import cv2
import numpy as np

img = cv2.imread('noisy_lenna.jpg',0)
img_medianblur = cv2.medianBlur(img, 5)

def medianBlur(img, kernel, padding_way):
    W,H = img.shape
    m,n = kernel.shape
    img_median = np.zeros((W+m-1,H+n-1))
    if padding_way == 'ZERO':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'constant')
    if padding_way == 'REPLICA':
        img_pad = np.pad(img,((m-1,m-1),(n-1,n-1)),'edge')
    for x in range(W+m-1):
        for y in range(H+n-1):
            window = img_pad[x:x+m,y:y+n]*kernel
            img_median[x,y] = int(np.median(window))
    return img_median

kernel = np.ones((5,5))
A = medianBlur(img,kernel,'ZERO')
A1 = A.astype(np.uint8)
B = medianBlur(img,kernel,'REPLICA')
B1 = B.astype(np.uint8)

cv2.imshow('img',img)
cv2.imshow('img_medianblur', img_medianblur)
cv2.imshow('img_zero',A1)
cv2.imshow('img_replica',B1)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()