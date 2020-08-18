import cv2
import numpy as np
import Image as Image

class ConvolutionLayer(object):

    output_packages = np.empty(0)
    input_packages = np.empty(0)

    def __init__(self, width_filter=2, height_filter=2 ,strider=1, filter_amount=5): 
        self.width_filter = width_filter
        self.height_filter = height_filter
        self.strider = strider
        self.filter_amount = filter_amount
        self.filters = np.empty(self.filter_amount, dtype=object)
        self.bias = np.empty(self.filter_amount, dtype=object)
        for i in range(self.filter_amount):
            self.filters[i] = np.random.uniform(low=0, high=1, size=(self.width_filter,self.height_filter))
            self.bias[i] = np.random.uniform(low=0, high=1, size=(self.width_filter,self.height_filter))

    
    def relu(self, X):
        return np.maximum(0,X)

    def reluDerivative(self, x):
        if(x<=0):
            return 0
        if(x>0):
            return 1
        return x
            
    def backpropagation(self, layers_derivates):
        alpha = 0.05
        backpropagation_pkg = Image.Package()
        index = -1
        filters_new = np.empty(self.filter_amount, dtype=object)
        for image in self.input_packages[0].images:
            index = index + 1
            img = image.img
            width_image = image.width
            height_image = image.height      

            #passos dados pelo filtro
            i = 1
            j = 1

            #posições dos pixels da nova imagem
            convolution_pixel_w = 0
            convolution_pixel_h = 0

            backpropagation_pkg.addImage(np.zeros((image.width,image.height,image.layers)))
            filter_derivative = self.convolutionFunction(image, layers_derivates.images[index].img)
            for filter_index in range(self.filter_amount):
                filters_new[filter_index] = (self.filters[filter_index] + alpha*filter_derivative)

            

            while (((height_image-j) - self.height_filter) > 0):
                i = 1
                convolution_pixel_h = 0
                while (((width_image-i) - self.width_filter) > 0):   
                    pixel_postion = 0
                    convolution_pixel_w = 0
                    for m in range(self.height_filter):
                        for n in range(self.width_filter):
                            neighborhood_width = (i-1) + n
                            neighborhood_height = (j-1) + m
                            for layer in range(image.layers):
                                if(image.layers == 1):
                                    img_before_conv = img[neighborhood_width,neighborhood_height]
                                    img_before_relu = img[neighborhood_width,neighborhood_height]*self.filters[index][n,m] + self.bias[index][n,m]
                                else:
                                    img_before_conv = img[neighborhood_width,neighborhood_height, layer]
                                    img_before_relu = img[neighborhood_width,neighborhood_height, layer]*self.filters[index][n,m] + self.bias[index][n,m]
                                #derivatie image
                                current_pixel = self.relu(img_before_relu)
                                erro_x = layers_derivates.images[index].img[n,m][0]*self.filters[index][n,m]*self.reluDerivative(img_before_relu)
                                backpropagation_pkg.setPixelImageByLayer(index, erro_x, neighborhood_width,neighborhood_height, layer)
                                    
                    convolution_pixel_w = convolution_pixel_w + 1
                    i = i + self.strider
                j = j + self.strider
                convolution_pixel_h = convolution_pixel_h + 1

        for filter_index in range(self.filter_amount):
                filters_new[filter_index] = filters_new[filter_index]/len(self.input_packages[0].images)
        self.input_packages = np.delete(self.input_packages, (0), axis=0).tolist()
        self.output_packages = np.delete(self.output_packages, (0), axis=0).tolist()
        return backpropagation_pkg, filters_new

    def process(self, package):
        self.input_packages = np.append(self.input_packages, package)
        for image in package.images:
            img = image.img

            width_image = image.width
            height_image = image.height

            new_w = round((width_image - self.width_filter)/self.strider + 1)
            new_h = round((height_image - self.height_filter)/self.strider + 1)

            convolution_pkg = Image.Package()            
            for i in range(self.filter_amount):
                convolution_pkg.addImage(np.zeros((new_w,new_h,image.layers), np.uint8))

            filter_size = self.width_filter*self.height_filter
            #matrix  que irá conter o resultado da multiplicação da vizinhança
            neighborhood_pixels = np.zeros((image.layers, filter_size))

            #passos dados pelo filtro
            i = 1
            j = 1

            #posições dos pixels da nova imagem
            convolution_pixel_w = 0
            convolution_pixel_h = 0

            for index in range(self.filter_amount):
                j = 1
                convolution_pixel_h = 0
                while (((height_image-j) - self.height_filter) > 0): 
                    i = 1
                    convolution_pixel_w = 0
                    while (((width_image-i) - self.width_filter) > 0):    
                        pixel_postion = 0
                        for m in range(self.height_filter):
                            for n in range(self.width_filter):
                                neighborhood_width = (i-1) + n
                                neighborhood_height = (j-1) + m
                                for layer in range(image.layers):
                                    if(image.layers > 1):
                                        neighborhood_pixels[layer][pixel_postion] = img[neighborhood_width,neighborhood_height, layer]*self.filters[index][n,m] + self.bias[index][n,m]
                                    else:
                                        neighborhood_pixels[layer][pixel_postion] = img[neighborhood_width,neighborhood_height]*self.filters[index][n,m] + self.bias[index][n,m]
                                    neighborhood_pixels[layer][pixel_postion] = self.relu(neighborhood_pixels[layer][pixel_postion])  
                                pixel_postion = pixel_postion + 1
                        new_pixel = [] 
                        for layer in range(image.layers):
                            pix = np.sum(neighborhood_pixels[:][layer])/filter_size
                            pix = self.relu(pix)
                            new_pixel.append(int(pix))
                        convolution_pkg.setPixelImage(index, new_pixel, convolution_pixel_w,convolution_pixel_h)
                        convolution_pixel_w = convolution_pixel_w + 1
                        i = i + self.strider
                    j = j + self.strider
                    convolution_pixel_h = convolution_pixel_h + 1

        for i in range(len(convolution_pkg.images)):
            cv2.imwrite('img/convolution_layer'+str(i)+'.png',convolution_pkg.images[i].img)
        self.output_packages = np.append(self.output_packages, convolution_pkg)
        return convolution_pkg



    def convolutionFunction(self, image, filter):        
        img = image.img

        width_image = image.width
        height_image = image.height

        width_filter,height_filter, layers = filter.shape

        new_w = round((width_image - width_filter)/self.strider + 1)
        new_h = round((height_image - height_filter)/self.strider + 1)

        convolution = np.random.uniform(low=0, high=1, size=(new_w,new_h))

        filter_size = width_filter*height_filter
        #matrix  que irá conter o resultado da multiplicação da vizinhança
        neighborhood_pixels = np.zeros((image.layers, filter_size))

        #passos dados pelo filtro
        i = 1
        j = 1

        #posições dos pixels da nova imagem
        convolution_pixel_w = 0
        convolution_pixel_h = 0

        j = 1
        convolution_pixel_h = 0
        while (((height_image-j) - height_filter) > 0):
            i = 1
            convolution_pixel_w = 0
            while (((width_image-i) - width_filter) > 0):    
                pixel_postion = 0
                for m in range(height_filter):
                    for n in range(width_filter):
                        neighborhood_width = (i-1) + n
                        neighborhood_height = (j-1) + m
                        for layer in range(image.layers):
                            if(image.layers > 1):
                                neighborhood_pixels[layer][pixel_postion] = img[neighborhood_width,neighborhood_height, layer]*filter[n,m][layer]
                            else:
                                neighborhood_pixels[layer][pixel_postion] = img[neighborhood_width,neighborhood_height]*filter[n,m]
                        pixel_postion = pixel_postion + 1
                new_pixel = np.empty(0)
                for layer in range(image.layers):
                    pix = np.sum(neighborhood_pixels[:][layer])/filter_size
                    pix = self.relu(pix)
                    new_pixel = np.append(new_pixel, int(pix))
                convolution[convolution_pixel_w,convolution_pixel_h] = new_pixel[0]
                convolution_pixel_w = convolution_pixel_w + 1
                i = i + self.strider
            j = j + self.strider
            convolution_pixel_h = convolution_pixel_h + 1

        return convolution
