import cv2
import matplotlib.pyplot as plt
import numpy as np
img_ori = cv2.imread('lenna.jpg',1)
img_gray = cv2.imread('lenna.jpg',0)
img_ori.shape
cv2.imshow('lenna',img_ori)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()

plt.imshow(img_ori)
plt.show()

plt.imshow(cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB))
plt.show()

plt.subplot(121)
plt.imshow(img_ori)
plt.subplot(122)
plt.imshow(cv2.cvtColor(img_ori,cv2.COLOR_BGR2RGB))
plt.show()

def my_show(img,size=(2,2)):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

my_show(img_ori)

#image crop
img_crop = img_ori[100:300, 100:200]
my_show(img_crop)

#channel split
plt.figure(figsize=(2,2))
plt.imshow(img_gray, cmap='gray')
plt.show()

B, G, R = cv2.split(img_ori)
B
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)

plt.subplot(131)
plt.imshow(B,cmap='gray')
plt.subplot(132)
plt.imshow(G,cmap='gray')
plt.subplot(133)
plt.imshow(R,cmap='gray')
plt.show()

# image cooler
def img_cooler(img,b_increase,r_decrease):
    B,G,R = cv2.split(img)
    b_lim = 255 - b_increase
    B[B > b_lim] = 255
    B[B <= b_lim] = (b_increase + B[B <= b_lim]).astype(img.dtype)
    r_lim = r_decrease
    R[R < r_lim] = 0
    R[R >= r_lim] = (R[R >= r_lim] - r_decrease).astype(img.dtype)
    return cv2.merge((B,G,R))

img_cool = img_cooler(img_ori,30,10)
my_show(img_cool)

# Gamma change
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)

img_dark = cv2.imread('dark.jpg')
my_show(img_dark,size=(6,6))

img_brighter = adjust_gamma(img_dark,5)
my_show(img_brighter,size=(6,6))

# histogram equalization
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
plt.show()

plt.subplot(121)
plt.hist(img_dark.flatten(), 256, [0, 256], color = 'b')
plt.subplot(122)
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
plt.show()

# YUV adjust luminance
img_yuv = cv2.cvtColor(img_brighter, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output  = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

my_show(img_output,size=(6,6))

plt.subplot(131)
plt.hist(img_dark.flatten(), 256, [0, 256], color = 'b')
plt.subplot(132)
plt.hist(img_brighter.flatten(), 256, [0, 256], color = 'r')
plt.subplot(133)
plt.hist(img_output.flatten(), 256, [0, 256], color = 'g')
plt.show()

# transform
# rotate
M = cv2.getRotationMatrix2D((img_ori.shape[1] / 2, img_ori.shape[0] / 2), 30, 1)
img_rotate = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))
my_show(img_rotate)

# affine transform
rows, cols, ch = img_ori.shape
pts1 = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1]])
pts2 = np.float32([[cols * 0.2, rows * 0.1], [cols * 0.7, rows * 0.2], [cols * 0.1, rows * 0.9]])

M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img_ori, M, (cols, rows))
my_show(dst)

# perspective transform
pts1 = np.float32([[0,0],[0,500],[500,0],[500,500]])
pts2 = np.float32([[5,19],[19,460],[460,9],[410,420]])

M = cv2.getPerspectiveTransform(pts1,pts2)
img_warp = cv2.warpPerspective(img_ori,M,(500,500))
my_show(img_warp)

# erode and dilate
img_writing = cv2.imread('libai.png',0)
plt.figure(figsize=(10,8))
plt.imshow(img_writing,cmap='gray')
plt.show()

# erode
erode_writing = cv2.erode(img_writing,None,iterations=1)
plt.figure(figsize=(10,8))
plt.imshow(erode_writing,cmap='gray')
plt.show()

# dilate
dilate_writing = cv2.dilate(img_writing,None,iterations=1)
plt.figure(figsize=(10,8))
plt.imshow(dilate_writing,cmap='gray')
plt.show()






