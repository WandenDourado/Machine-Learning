import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

def getFace(img):
    face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
    face_cascade_alt = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt.xml')
    face_cascade_alt2 = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
    face_cascade_alt_tree = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt_tree.xml')
    face_cascade_profile = cv2.CascadeClassifier('data/haarcascades/haarcascade_profileface.xml')

    #img = cv2.imread(path)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    face = haar_search(face_cascade, gray, img)
    if(len(face) != 0):
        return face

    face = haar_search(face_cascade_alt, gray, img)
    if(len(face) != 0):
        return face

    face = haar_search(face_cascade_alt2, gray, img)
    if(len(face) != 0):
        return face

    face = haar_search(face_cascade_alt_tree, gray, img)
    if(len(face) != 0):
        return face

    face = haar_search(face_cascade_profile, gray, img)
    if(len(face) != 0):
        return face

    return img


    #cv2.imshow('img',roi_color)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #return False

def haar_search(face_cascade, gray, img):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.8, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.8, 8)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.2, 2)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 2, 4)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 2, 8)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.4, 6)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.8, 4)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 1.8, 10)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 10, 10)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    faces = face_cascade.detectMultiScale(gray, 5, 10)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        return roi_color

    return []

def hogFeature(image):
    #image = data.astronaut()

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    if(False):
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    if(False):
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    plt.close()
    return hog_image_rescaled

def printImage(image):
    cv2.imshow('img',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
