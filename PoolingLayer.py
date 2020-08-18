import cv2
import numpy as np
import Image as Image

class PoolingLayer(object):

    output_packages = np.empty(0)
    input_packages = np.empty(0)

    def __init__(self, width_filter=2, height_filter=2, strider=1): 
        self.width_filter = width_filter
        self.height_filter = height_filter
        self.strider = strider
        self.mode = 'max'

    def backpropagation(self, pooling_backpropagation):  
        image_amount = len(self.output_packages[0].images)

        backpropagation_pkg = Image.Package()
            

        i = 1
        j = 1

        index = -1

        #posições dos pixels da nova imagem
        pooling_pixel_w = 0
        pooling_pixel_h = 0
        for image in self.input_packages[0].images:
            index = index + 1
            backpropagation_pkg.addImage(np.zeros((image.width,image.height,image.layers)))
            j = 1
            while (((image.height-j) - self.height_filter) > 0):
                i = 1
                pooling_pixel_h = 0
                while (((image.width-i) - self.width_filter) > 0):     
                    pixel_postion = 0
                    pooling_pixel_w = 0
                    for m in range(self.height_filter):
                        for n in range(self.width_filter):
                            neighborhood_width = (i-1) + n
                            neighborhood_height = (j-1) + m
                            for layer in range(image.layers):
                                    if(self.output_packages[0].images[index].img[pooling_pixel_w,pooling_pixel_h, layer] == image.img[neighborhood_width,neighborhood_height, layer]):
                                        new_pixel = pooling_backpropagation.images[index].img[pooling_pixel_w,pooling_pixel_h, layer]
                                        backpropagation_pkg.setPixelImageByLayer(index, new_pixel, neighborhood_width,neighborhood_height, layer)
                    pooling_pixel_w = pooling_pixel_w + 1
                    i = i + self.strider
                j = j + self.strider
                pooling_pixel_h = pooling_pixel_h + 1
        self.input_packages = np.delete(self.input_packages, (0), axis=0)
        self.output_packages = np.delete(self.output_packages, (0), axis=0)
        return backpropagation_pkg


    
    def process(self, package):
        self.input_packages = np.append(self.input_packages, package)
        pooling_pkg = Image.Package()    
        image_amount = len(package.images)
        filter_size = self.width_filter*self.height_filter
        index = -1

        for image in package.images:
            index = index + 1
            img = image.img

            width_image = image.width
            height_image = image.height

            new_w = round((width_image - self.width_filter)/self.strider + 1)
            new_h = round((height_image - self.height_filter)/self.strider + 1)

            pooling_pkg.addImage(np.zeros((new_w,new_h,image.layers), np.uint8))

            
            neighborhood_pixels = np.zeros((image.layers, filter_size))

            i = 1
            j = 1

            #posições dos pixels da nova imagem
            pooling_pixel_w = 0
            pooling_pixel_h = 0
            j = 1
            pooling_pixel_h = 0
            while (((height_image-j) - self.height_filter) > 0):
                i = 1
                pooling_pixel_w = 0
                while (((width_image-i) - self.width_filter) > 0):     
                    pixel_postion = 0                    
                    for m in range(self.height_filter):
                        for n in range(self.width_filter):
                            neighborhood_width = (i-1) + n
                            neighborhood_height = (j-1) + m
                            if(image.layers > 1):
                                neighborhood_pixels[0][pixel_postion] = img[neighborhood_width,neighborhood_height, 0]#R
                                neighborhood_pixels[1][pixel_postion] = img[neighborhood_width,neighborhood_height, 1]#G
                                neighborhood_pixels[2][pixel_postion] = img[neighborhood_width,neighborhood_height, 2]#B
                            else:
                                neighborhood_pixels[0][pixel_postion] = img[neighborhood_width,neighborhood_height]
                            pixel_postion = pixel_postion + 1
                    if(self.mode == 'max'):
                        #Max pooling
                        if(image.layers > 1):
                            pixel_red = np.max(neighborhood_pixels[:][0])
                            pixel_green = np.max(neighborhood_pixels[:][1])
                            pixel_blue = np.max(neighborhood_pixels[:][2])
                            new_pixel = (int(pixel_red), int(pixel_green),int(pixel_blue))
                        else:
                            new_pixel = np.max(neighborhood_pixels[:][0])
                    else:
                        #Average pooling
                        if(image.layers > 1):
                            pixel_red = np.sum(neighborhood_pixels[:][0])/filter_size
                            pixel_green = np.sum(neighborhood_pixels[:][1])/filter_size
                            pixel_blue = np.sum(neighborhood_pixels[:][2])/filter_size
                            new_pixel = (int(pixel_red), int(pixel_green),int(pixel_blue))
                        else:
                            new_pixel = np.sum(neighborhood_pixels[:])/filter_size
                    pooling_pkg.setPixelImage(index, new_pixel, pooling_pixel_w,pooling_pixel_h)
                    pooling_pixel_w = pooling_pixel_w + 1
                    i = i + self.strider
                j = j + self.strider
                pooling_pixel_h = pooling_pixel_h + 1


        cv2.imwrite('img/Pooling_layer.png', pooling_pkg.images[0].img)
        self.output_packages = np.append(self.output_packages, pooling_pkg)
        return pooling_pkg

