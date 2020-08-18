import numpy as np

class Image(object):
    def __init__(self, img): 
        self.img = img
        if(len(img.shape) > 2):
            self.width, self.height, self.layers = img.shape 
        else:
            self.width, self.height = img.shape 
            self.layers = 1

    def setPixel(self, new_pixel, width, height):
        self.img[width, height] = new_pixel

    def setPixelByLayer(self, new_pixel, width, height, layer):
        self.img[width, height, layer] = new_pixel

class Package(object):
    images = np.empty(0)

    def __init__(self): 
        self.images = np.empty(0)

    def addImage(self, image):
        img = Image(image)
        self.images= np.append(self.images, img)

    def setPixelImage(self,index, new_pixel, width, height):
        self.images[index].setPixel(new_pixel, width, height)

    def setPixelImageByLayer(self,index, new_pixel, width, height, layer):
        self.images[index].setPixelByLayer(new_pixel, width, height, layer)

