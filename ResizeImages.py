# from concurrent.futures import process
from PIL import Image
# import cv2
# import numpy as np

def resizeImage(imageName):
    img = Image.open(imageName)

    # process(imageName, img, 100)

    basewidth = 100
    # wpercent = (basewidth/float(img.size[0]))
    # hsize = int((float(img.size[1])*float(wpercent)))
    hsize = 89
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def process(imageName, image, threshold) :
    gray = cv2.imread(imageName)
    # gray = np.array(image)
    # gray = gray > threshold 
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = gray > threshold

    image_new = Image.fromarray(gray)
    image_new.save(imageName)

for i in range(0, 101):
    print(i)
    # Mention the directory in which you wanna resize the images followed by the image name
    resizeImage("Dataset/ThumbTest/thumb_" + str(i) + '.png')
    resizeImage("Dataset/RightTest/thumb_" + str(i) + '.png')
    resizeImage("Dataset/LeftTest/thumb_" + str(i) + '.png')


