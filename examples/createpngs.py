__author__ = 'sorensonderby'
import time
import gzip
import pickle
import os
import sys
import scipy.io as sio
import numpy as np
from helper_functions import *
from PIL import Image
from deepmodels.layers.draw_helpers import nn2att as nn2att









def write_images(c, epoch=None,folder=None):
    img_out_lst = []
    for i in range(400):
        img_out =c[i]
        img_out = (255*img_out).astype(np.uint8)
        img_out = Image.fromarray(img_out)
        if folder is not None and epoch is not None:
            img_out.save(os.path.join(folder,epoch + ".BMP"))
        img_out_lst.append(img_out)
    return img_out_lst




for epoch in [58]:
    print epoch

    folder = str(epoch)
    canvas_name = 'canvas.mat'

    f_in = os.path.join(folder, canvas_name)


    def logit(a):
        return np.log((a)/(1-a))


    if not os.path.exists(folder):
        os.makedirs(folder)


    data = sio.loadmat(f_in)
    canvas_data = data['canvas']
    glimpses = canvas_data.shape[1]



    for j in range(glimpses):
            epoch_str = "%03d_glimpse_%03d" % (epoch, j)
            center_y, center_x, delta, sigma, gamma, muX, muY = read_params(data['att_write'][:,j, :],5, [28,28])
            c = create_reading_square(muX, muY)
            img_digit = plot_n_by_n_images(1.0/(1+np.exp(-canvas_data[:,j,:])), n=20)
            img_reading = plot_n_by_n_images(c.reshape(400, 784), n=20)
            img_digit = Image.fromarray(np.asarray(np.dstack((img_digit, img_digit, img_digit)), dtype=np.uint8))
            img_reading = Image.fromarray(np.asarray(np.dstack((np.zeros_like(img_reading), img_reading, np.zeros_like(img_reading))), dtype=np.uint8))

            new_img = Image.blend(img_digit, img_reading, 0.5)
            new_img.save(os.path.join(folder, epoch_str+".png"),"PNG")

    #plot_n_by_n_images(canvas_data,epoch,folder, n = 20, shp=[28,28])

    os.system('convert -delay 10 -loop 0 %s/*.png %s/animaion.gif' % (folder, folder))
