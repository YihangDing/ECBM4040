#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# This Python script contains the ImageGenrator class.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate


class ImageGenerator(object):

    def __init__(self, x, y):
        """
        Initialize an ImageGenerator instance.
        :param x: A Numpy array of input data. It has shape (num_of_samples, height, width, channels).
        :param y: A Numpy vector of labels. It has shape (num_of_samples, ).
        """

        # TODO: Your ImageGenerator instance has to store the following information:
        # x, y, num_of_samples, height, width, number of pixels translated, degree of rotation, is_horizontal_flip,
        # is_vertical_flip, is_add_noise. By default, set boolean values to False.
        #
        # Hint: Since you may directly perform transformations on x and y, and don't want your original data to be contaminated 
        # by those transformations, you should use numpy array build-in copy() method. 
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        #raise NotImplementedError
        #self.num_of_samples=np.shape(x)[0]
        self.num_of_samples,self.hight,self.width, self.channels=x.shape
        #self.width=np.shape(x)[2]
        #self.channels=np.shape(x)[3]
        self.num_of_pixels_translated=0
        self.degree_of_rotation=0.0
        self.is_horizontal_flip=False
        self.is_vertical_flip=False
        self.is_add_noise=False

        self.x=x.copy()
        self.y=y.copy()



        # One way to use augmented data is to store them after transformation (and then combine all of them to form new data set).
        # Following variables (along with create_aug_data() function) is one kind of implementation.
        # You can either figure out how to use them or find out your own ways to create the augmented dataset.
        # if you have your own idea of creating augmented dataset, just feel free to comment any codes you don't need
        
        self.translated = None
        self.rotated = None
        self.flipped = None
        self.added = None
        self.x_aug = self.x.copy()
        self.y_aug = self.y.copy()
        self.N_aug = self.num_of_samples

    def create_aug_data(self):
        # If you want to use function create_aug_data() to generate new dataset, you can perform the following operations in each
        # transformation function:
        #
        # 1.store the transformed data with their labels in a tuple called self.translated, self.rotated, self.flipped, etc. 
        # 2.increase self.N_aug by the number of transformed data,
        # 3.you should also return the transformed data in order to show them in task4 notebook
        # 
        
        '''
        Combine all the data to form a augmented dataset 
        '''
        if self.translated:
            self.x_aug = np.vstack((self.x_aug,self.translated[0]))
            self.y_aug = np.hstack((self.y_aug,self.translated[1]))
        if self.rotated:
            self.x_aug = np.vstack((self.x_aug,self.rotated[0]))
            self.y_aug = np.hstack((self.y_aug,self.rotated[1]))
        if self.flipped:
            self.x_aug = np.vstack((self.x_aug,self.flipped[0]))
            self.y_aug = np.hstack((self.y_aug,self.flipped[1]))
        if self.added:
            self.x_aug = np.vstack((self.x_aug,self.added[0]))
            self.y_aug = np.hstack((self.y_aug,self.added[1]))

    def next_batch_gen(self, batch_size, shuffle=True):
        """
        A python generator function that yields a batch of data infinitely.
        :param batch_size: The number of samples to return for each batch.
        :param shuffle: If True, shuffle the entire dataset after every sample has been returned once.
                        If False, the order or data samples stays the same.
        :return: A batch of data with size (batch_size, width, height, channels).
        """

        # TODO: Use 'yield' keyword, implement this generator. Pay attention to the following:
        # 1. The generator should return batches endlessly.
        # 2. Make sure the shuffle only happens after each sample has been visited once. Otherwise some samples might
        # not be output.

        # One possible pseudo code for your reference:
        #######################################################################
        #   calculate the total number of batches possible (if the rest is not sufficient to make up a batch, ignore)
        #   while True:
        #       if (batch_count < total number of batches possible):
        #           batch_count = batch_count + 1
        #           yield(next batch of x and y indicated by batch_count)
        #       else:
        #           shuffle(x)
        #           reset batch_count
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        num_of_samples=self.num_of_samples
        total_batch=num_of_samples//batch_size
        batch_count=0

        x=self.x
        y=self.y

        while 1:
            if(batch_count<total_batch):
                batch_count+=1
                yield (x[batch_count*batch_size:(batch_count+1)*batch_size,:,:,:],y[batch_count*batch_size:(batch_count+1)*batch_size])
            else:

                #shuffle()
                index=np.random.choice(self.num_of_samples,self.num_of_samples,replace=False)
                #np.random.shuffle(x)
                self.x=x[index]
                self.y=y[index]
                #reset batch_count
                batch_count=0



    def show(self, images=16):
        """
        Plot the top 16 images (index 0~15) for visualization.
        :param images: images to be shown
        """
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        plt.figure(figsize=(8,8))
        for i in range(images):
            plt.subplot(4,4,i+1)
            plt.imshow(self.x[i])

    def translate(self, shift_height, shift_width):
        """
        Translate self.x by the values given in shift.
        :param shift_height: the number of pixels to shift along height direction. Can be negative.
        :param shift_width: the number of pixels to shift along width direction. Can be negative.
        :return translated: translated dataset
        """

        # TODO: Implement the translate function. Remember to record the value of the number of pixels shifted.
        # Note: You may wonder what values to append to the edge after the shift. Here, use rolling instead. For
        # example, if you shift 3 pixels to the left, append the left-most 3 columns that are out of boundary to the
        # right edge of the picture.
        # Hint: Numpy.roll (https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.roll.html)
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        #record the value of the number of pixels shifted
        self.num_of_pixels_translated=self.hight*self.width-(self.hight-shift_height)*(self.width-shift_width)

        self.x = np.roll(self.x, shift_height, axis=1)
        self.x=np.roll(self.x, shift_width, axis=2)

        #prep to form argunemted dataset
        self.translated=[self.x,self.y]





    def rotate(self, angle=0.0):
        """
        Rotate self.x by the angles (in degree) given.
        :param angle: Rotation angle in degrees.
        :return rotated: rotated dataset
        - https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
        """
        # TODO: Implement the rotate function. Remember to record the value of
        # rotation degree.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        self.degree_of_rotation=angle

        #rotate
        self.x=rotate(self.x,angle=angle,axes=(1,2),reshape=False)

        # prep to form argunemted dataset
        self.rotated=[self.x,self.y]

    def flip(self, mode='h'):
        """
        Flip self.x according to the mode specified
        :param mode: 'h' or 'v' or 'hv'. 'h' means horizontal and 'v' means vertical.
        :return flipped: flipped dataset
        """
        # TODO: Implement the flip function. Remember to record the boolean values is_horizontal_flip and
        # is_vertical_flip.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################

        if mode=='h':
            self.is_horizontal_flip=True
            self.x = np.flip(self.x, 1)
        elif mode=='v':
            self.is_vertical_flip=True
            self.x=np.flip(self.x,0)
        else:
            self.is_horizontal_flip=True
            self.is_vertical_flip = True
            self.x = np.flip(self.x, 1)
            self.x = np.flip(self.x, 0)

        # prep to form argunemted dataset
        self.flipped=[self.x,self.y]




    def add_noise(self, portion, amplitude):
        """
        Add random integer noise to self.x.
        :param portion: The portion of self.x samples to inject noise. If x contains 10000 sample and portion = 0.1,
                        then 1000 samples will be noise-injected.
        :param amplitude: An integer scaling factor of the noise.
        :return added: dataset with noise added
        """
        # TODO: Implement the add_noise function. Remember to record the
        # boolean value is_add_noise. You can try uniform noise or Gaussian
        # noise or others ones that you think appropriate.
        #raise NotImplementedError
        #######################################################################
        #                                                                     #
        #                                                                     #
        #                         TODO: YOUR CODE HERE                        #
        #                                                                     #
        #                                                                     #
        #######################################################################
        #decide index
        num_add_noise=int(self.num_of_samples*portion)
        index=np.random.choice(self.num_of_samples,size=num_add_noise,replace=False)


        for i in index:
            self.x[i]+= int(amplitude*float(np.random.rand(1)))



        self.is_add_noise=True

        # prep to form argunemted dataset
        self.added=[self.x,self.y]

