import numpy as np
import cv2
import math
img1=cv2.imread("1_cat.bmp")
# 1_cat.bmp     1_dog.bmp
# 2_einstein.bmp    2_marilyn.bmp
# 3_che-guevara.jpeg    3_jim-morisson.jpeg
# 4_joconde-mc-curry.jpeg   4_mona-lisa.jpeg
# 5_fish.bmp    5_submarine.bmp
img2=cv2.imread("1_dog.bmp")
def norimage(img):
    ima=img.max()
    imi=img.min()
    return (img-imi)/(ima-imi)
def extension(img,kernel):
    img=np.array(img)
    k=np.array(kernel)
    img_sp=img.shape
    img_h=img_sp[0]
    img_w=img_sp[1]
    k_sp=k.shape
    k_h=k_sp[0]
    k_w=k_sp[1]
    h=k_h//2
    w=k_w//2
    for i in range(h):
        img=np.row_stack([img,np.zeros(img_w)])
        img=np.row_stack([np.zeros(img_w),img])
    for i in range(w):
        img=np.column_stack([img,np.zeros(img_h+h*2)])
        img=np.column_stack([np.zeros(img_h+h*2),img])
    return img
def cross_correlation_2d(img, kernel):
    img=extension(img, kernel)
    img_sp=img.shape
    k_sp=kernel.shape

    img_h=img_sp[0]
    img_w=img_sp[1]
    k_h=k_sp[0]
    k_w=k_sp[1]
    half_k_h=k_h//2
    half_k_w=k_w//2
    img_new=[]
    for i in range(half_k_h,img_h-half_k_h):
        line=[]
        for j in range(half_k_w,img_w-half_k_w):
            a=img[i-half_k_h:i+half_k_h+1,j-half_k_w:j+half_k_w+1]
            line.append(np.sum(np.multiply(a,kernel)))
        img_new.append(line)
    return img_new
def convolve_2d(img, kernel):
    img=np.array(img)
    if img.ndim==3:
        r=img[:,:,0]
        g=img[:,:,1]
        b=img[:,:,2]


        r_new=cross_correlation_2d(r,kernel)
        g_new=cross_correlation_2d(g,kernel)
        b_new=cross_correlation_2d(b,kernel)

        img[:,:,0]=r_new
        img[:,:,1]=g_new
        img[:,:,2]=b_new
        return img
    else:
        img_ex=extension(img,kernel)
        img=cross_correlation_2d(img_ex,kernel)
        return img
def gaussian_blur_kernel_2d(height, width,sigma=1.5):
    xs=1.0/(2*math.pi*math.pow(sigma,2))
    midh=height//2
    midw=width//2
    kernel=[]
    for i in range(height):
        line=[]
        for j in range(width):
             line.append(xs*math.exp(-1*(math.pow(i-midh,2)+math.pow(j-midw,2))/(2*math.pow(sigma,2))))
        kernel.append(line)
    kernel=np.array(kernel)
    kernel=normalize(kernel)
    return kernel
def normalize(kernel):
    s = kernel.sum()
    sp=kernel.shape
    height=sp[0]
    width=sp[1]
    for i in range(height):
        for j in range(width):
            kernel[i, j] = kernel[i, j] / s
    return kernel
def low_pass(img,height,width,sigma):
    kernel=np.zeros([height,width])
    kernel=gaussian_blur_kernel_2d(height,width,sigma)
    norimage(img)
    img = convolve_2d(img, kernel)
    img=norimage(img)
    return img
def high_pass(img,height,width,sigma):
    kernel=np.zeros([height,width])
    kernel=gaussian_blur_kernel_2d(height,width,sigma)
    img=norimage(img)
    imgn=img
    img=low_pass(img,height,width,sigma)
    img=norimage(img)
    img = imgn-img
    img=norimage(img)
    return img
def hybrid_image(img1,img2,height,width,sigma):
    img1=low_pass(img1,height,width,sigma)
    img2 = high_pass(img2, height, width, sigma)
    img1=img1+img2
    img1=norimage(img1)
    img1*=255
    return img1
img1=hybrid_image(img2,img1,25,25,13)
cv2.imwrite('output1.png', img1)