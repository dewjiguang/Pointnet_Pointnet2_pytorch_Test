from PIL import Image
import os
import cv2 as cv

def custom_blur_demo(src,outPut):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
    # src = cv.imread(src)
    dst = cv.filter2D(src, -1, kernel=kernel)
    cv2.imwrite(outPut, dst)







path = "D:/train/data/"
def image_processing(url,sizea,sizeb,outPutName):
    #  待处理图片路径
    # img_path = Image.open('./images/1.png')
    img_path = Image.open(url)
    #  resize图片大小，入口参数为一个tuple，新的图片的大小
    img_size = img_path.resize((sizea, sizeb))
    #  处理图片后存储路径，以及存储格式
    img_size.save('data/'+outPutName+'.jpg', 'JPEG')

def image_Splicing(img_1, img_2, id, flag='x',):
    img1 = Image.open(img_1)
    img2 = Image.open(img_2)
    size1, size2 = img1.size, img2.size
    if flag == 'x':
        joint = Image.new("RGB", (size1[0] + size2[0], size1[1]))
        loc1, loc2 = (0, 0), (size1[0], 0)
    else:
        joint = Image.new("RGB", (size1[0], size2[1]+size1[1]))
        loc1, loc2 = (0, 0), (0, size1[1])
    joint.paste(img1, loc1)
    joint.paste(img2, loc2)
    joint.save("D:/train/data2/" + str(id) +'.png')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
def GaussFilter(imgSrc):
    # img = cv2.imread(imgSrc, cv2.IMREAD_GRAYSCALE)
    img=cv2.imread(imgSrc)
    img=cv2.resize(src=img,dsize=(450,450))
    img=cv2.GaussianBlur(src=img,ksize=(11,11),sigmaX=0,sigmaY=1.0)
    return img

    # cv2.imshow('img',img)
    # cv2.imshow('img_src',img_src)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def resolveRGB(str):
    from matplotlib import pyplot as plt

    # Reading image from folder where it is stored

    img = str

    # denoising of image saving it into dst image

    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    cv2.imwrite('data/resizeA.jpg', dst)
    # Plotting of source and destination image
    return dst

def resolveZaoDian(a,outPutName):
    img=a
    # img = cv2.imread('data/a.jpg', cv2.IMREAD_GRAYSCALE)
    # plt.imshow(img, 'gray')
    # # plt.rc("font", family='Microsoft YaHei')
    # # img = plt.imread('data/a.jpg')
    #
    # plt.subplot(221), plt.title('原始图像')
    # plt.imshow(img, 'gray')
    # noise_img = skimage.util.random_noise(img, mode='salt') * 255
    # plt.subplot(222), plt.title('椒盐噪声图像')
    # plt.imshow(noise_img, 'gray')
    # 均值滤波

    mean_img = img
    noise_img=img
    for i in range(1, noise_img.shape[0] - 1):  # 第一列和最后一列用不到
        for j in range(1, noise_img.shape[1] - 1):  # 第一行和最后一行用不到
            tmp = 0  # 用来求和
            for k in range(-1, 2):
                for l in range(-1, 2):
                    tmp += noise_img[i + k][j + l]
            mean_img[i][j] = tmp / 9
    plt.subplot(223), plt.title('均值滤波后图像')
    plt.imshow(mean_img, 'gray')
    cv2.imwrite('data/' + outPutName + '.jpg', mean_img)
    # 中值滤波
    median_img = img

    for i in range(1, noise_img.shape[0] - 1):  # 第一列和最后一列用不到
        for j in range(1, noise_img.shape[1] - 1):  # 第一行和最后一行用不到
            tmpp = []  # 用来记录9个值
            for k in range(-1, 2):
                for l in range(-1, 2):
                    tmpp.append(noise_img[i + k][j + l])
            list.sort(np.asarray(tmpp).flatten().tolist())
            median_img[i][j] = tmpp[4]  # 取得中值
    cv2.imwrite('data/' + outPutName + '.jpg', median_img)
    # median_img.save('data/' + outPutName + '.jpg', 'JPEG')
    # plt.subplot(224), plt.title('中值滤波后图像')
    # plt.imshow(median_img, 'gray')
    # plt.show()

if __name__ == '__main__':



    # resolveZaoDian(GaussFilter(),)
    #resolveZaoDian()

    train_list = os.listdir(path)
    for i in range(len(train_list)):
        pic_list = os.listdir(path + f"{i}")
        image_processing(path + str(i) + '/a.jpg',200,400,'resizeA')
        image_processing(path + str(i) + '/b.jpg',200,400,'resizeB')
        result=resolveRGB(GaussFilter('data/resizeA.jpg'))
        # resolveZaoDian(GaussFilter('data/resizeA.jpg'),'resizeA')
        # cv2.imwrite('data/resizeB.jpg', GaussFilter('data/resizeB.jpg'))
        custom_blur_demo(result,'data/resizeA_ruiHua.jpg')
        image_processing('data/resizeA_ruiHua.jpg', 200, 400, 'resizeA_ruiHua')



        image_Splicing('data/resizeA_ruiHua.jpg','data/resizeB.jpg', i)
        print(i)
