## The attention code is copied from https://github.com/jbornschein/draw
from __future__ import division, print_function
import theano
import theano.tensor as T
import numpy as np

def my_batched_dot(A, B):
    """
    This is copied from jbornschein code:
    https://github.com/jbornschein/draw/blob/master/draw/attention.py

    Batched version of dot-product.
    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this
    is \approx equal to:
    for i in range(dim_1):
    C[i] = tensor.dot(A, B)
    Returns
    -------
    C : shape (dim_1 \times dim_2 \times dim_4)
    """
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)

def filterbank_matrices(center_y, center_x, delta, sigma, N, imgshp):
    """Create a Fy and a Fx

    Parameters
    ----------
    center_y : T.vector (shape: batch_size)
    center_x : T.vector (shape: batch_size)
        Y and X center coordinates for the attention window
    delta : T.vector (shape: batch_size)
    sigma : T.vector (shape: batch_size)

    Returns
    -------
        FY, FX
    """
    tol = 1e-4
    img_height, img_width = imgshp
    muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)
    muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)

    a = T.arange(img_width)
    b = T.arange(img_height)

    FX = T.exp( -(a-muX.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
    FY = T.exp( -(b-muY.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
    FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
    FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

    return FY, FX


def read(images,images_err, att_read, N, imgshp):
    """
    Parameters
    ----------
    images : T.matrix    (shape: batch_size x img_size)
        Batch of images. Internally it will be reshaped to be a
        (batch_size, img_height, img_width)-shaped stack of images.
    center_y : T.vector (shape: batch_size)
    center_x : T.vector (shape: batch_size)
    delta : T.vector    (shape: batch_size)
    sigma : T.vector    (shape: batch_size)

    Returns
    -------
    window : T.matrix   (shape: batch_size x N**2)
    """
    center_y = att_read['center_y']
    center_x = att_read['center_x']
    delta = att_read['delta']
    sigma = att_read['sigma']
    A, B = imgshp
    batch_size = images.shape[0]

    # Reshape input into proper 2d images
    I = images.reshape( (batch_size, A, B) )
    I_err = images_err.reshape( (batch_size, A, B) )

    # Get separable filterbank
    FY, FX = filterbank_matrices(center_y, center_x, delta, sigma, N, imgshp)

    # apply to the batch of images

    W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0,2,1]))
    W_err = my_batched_dot(my_batched_dot(FY, I_err), FX.transpose([0,2,1]))

    return W.reshape((batch_size, N*N)), W_err.reshape((batch_size, N*N))


def read_single(images, att_read, N, imgshp):
    """
    Parameters
    ----------
    images : T.matrix    (shape: batch_size x img_size)
        Batch of images. Internally it will be reshaped to be a
        (batch_size, img_height, img_width)-shaped stack of images.
    center_y : T.vector (shape: batch_size)
    center_x : T.vector (shape: batch_size)
    delta : T.vector    (shape: batch_size)
    sigma : T.vector    (shape: batch_size)

    Returns
    -------
    window : T.matrix   (shape: batch_size x N**2)
    """
    center_y = att_read['center_y']
    center_x = att_read['center_x']
    delta = att_read['delta']
    sigma = att_read['sigma']
    A, B = imgshp
    batch_size = images.shape[0]

    # Reshape input into proper 2d images
    I = images.reshape( (batch_size, A, B) )

    # Get separable filterbank
    FY, FX = filterbank_matrices(center_y, center_x, delta, sigma, N, imgshp)

    # apply to the batch of images

    W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0,2,1]))

    return W.reshape((batch_size, N*N))

def write(windows, att_write, N, imgshp):
    img_height,img_width = imgshp
    batch_size = windows.shape[0]
    center_y = att_write['center_y']
    center_x = att_write['center_x']
    delta = att_write['delta']
    sigma = att_write['sigma']

    # Reshape input into proper 2d windows
    W = windows.reshape( (batch_size, N, N) )

    # Get separable filterbank
    FY, FX = filterbank_matrices(center_y, center_x, delta, sigma, N, imgshp)

    # apply...
    I = my_batched_dot(my_batched_dot(FY.transpose([0,2,1]), W), FX)

    return I.reshape( (batch_size, img_height*img_width) )

def nn2att(l, N,  imgshp):
    """Convert neural-net outputs to attention parameters

    Parameters
    ----------
    l : tensor (batch_size x 5)

    Returns
    -------
    center_y : vector (batch_size)
    center_x : vector (batch_size)
    delta : vector (batch_size)
    sigma : vector (batch_size)
    gamma : vector (batch_size)
    """
    A,B = imgshp
    center_y  = l[:, 0]
    center_x  = l[:, 1]   # sigmoid all thee
    log_delta = l[:, 2]   # sigmoid all these
    log_sigma2 = l[:, 3]
    log_gamma = l[:, 4]

    delta = T.exp(log_delta)
    sigma = T.exp(log_sigma2/2.)
    gamma = T.exp(log_gamma).dimshuffle(0, 'x')

    # normalize coordinates
    delta = (max(A, B)-1) / (N-1) * delta
    center_x = (center_x+1.)* (B+1)*0.5
    center_y = (center_y+1.)* (A+1)*0.5




    return {'center_y':center_y, 'center_x':center_x, 'delta':delta,
            'sigma':sigma, 'gamma':gamma}


