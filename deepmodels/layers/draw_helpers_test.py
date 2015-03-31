from __future__ import division
from PIL import Image

import numpy as np
import theano
import theano.tensor as T
from theano import tensor
from draw_helpers import  *
import matplotlib
import lasagne.init as init
ini = init.Normal(std=0.01, avg=0.0)
zero = init.Constant(0.)
import matplotlib.cm as cm

nlstm = 256
W_wr = ini((nlstm, 5))
b_wr = zero((5))
hid_dec = T.tanh(ini((1,nlstm))).eval()

att_vals_read = np.dot(hid_dec, W_wr) + b_wr


USE_CAT = False
if USE_CAT:
    N = 20
    height = 480
    width =  640
    I = Image.open("cat.jpg")
    I = I.resize((width, height)).convert('L')

    I = np.asarray(I).reshape( (width*height) )
    I = I / 255.
    cmap = None
else: # use digit
    N = 12
    height = 28
    width =  28
    I = np.load('exampledigit.npy')
    I = I.reshape((28*28))
    cmap = None


# #------------------------------------------------------------------------
# att = ZoomableAttentionWindow(height, width, N)
#
I_ = T.matrix()
I_err_ = T.matrix()
center_y_ = T.vector()
center_x_ = T.vector()
delta_ = T.vector()
sigma_ = T.vector()
att_read = {}
att_read['center_y'] = center_y_
att_read['center_x'] = center_x_
att_read['delta'] = delta_
att_read['sigma'] = sigma_

l_ = T.matrix()
att_ = nn2att(l_, N,  [height, width])
W_, W_err = read(I_,I_err_, att_, N, [height, width])
do_read = theano.function(inputs=[I_, I_err_, l_],
                          outputs=[W_, W_err], allow_input_downcast=True)
I_wrt = write(W_, att_, N, [height, width])
do_write = theano.function(inputs=[W_, l_],
outputs=I_wrt, allow_input_downcast=True)

delta_read = 1
delta_write = 1
print delta_write, "DELTA"
gamma = 1
sigma_read = np.max([width,height])
sigma_write = np.max([width,height])
center_y = 0.
center_x = 0.

def vectorize(*args):
    return [a.reshape((1,)+a.shape) for a in args]

I, center_y, center_x, delta_read, sigma = \
    vectorize(I, np.array(center_y), np.array(center_x), np.array(delta_read), np.array(sigma_read))

# #import ipdb; ipdb.set_trace()


b_read = np.array([center_y, center_x, np.log(delta_read), np.log(sigma_read), np.log(gamma)]).astype('float32').reshape((1,5))*0
b_write = np.array([center_y, center_x, np.log(delta_write), np.log(sigma_write), np.log(gamma)]).astype('float32').reshape((1,5))*0
W_read = ini((nlstm, 5))
W_write = ini((nlstm, 5))
print delta_read, gamma
att_read = np.dot(hid_dec, W_read) + b_read
att_write = np.dot(hid_dec, W_write) + b_write
#
W_omega = ini((nlstm, N*N))

W_write = np.dot(hid_dec,W_omega )
W, W_err  = do_read(I, I, att_read)
I2 = do_write(W, att_write)

import pylab
pylab.figure()
pylab.gray()
pylab.imshow(I.reshape([height, width]), interpolation='nearest', cmap=cmap)
#
pylab.figure()
pylab.gray()
pylab.imshow(W.reshape([N, N]), interpolation='nearest', cmap=cmap)

print("")

pylab.figure()
pylab.gray()
pylab.imshow(I2.reshape([height, width]), interpolation='nearest',cmap=cmap)
pylab.show(block=True)
pylab.show(block=True)
#
# pylab.figure()
# pylab.gray()



#pylab.imshow(I2.reshape([height, width]), interpolation='nearest')

#
# import ipdb; ipdb.set_trace()
