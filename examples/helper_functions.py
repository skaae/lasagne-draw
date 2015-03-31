# l2_norm and step_clipping copied/adapted from blocks
from theano import tensor as T
import theano
from theano import ifelse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.pyplot import plot, draw, show, ion, figure, savefig, imshow, matshow


def adam_return_steps(all_grads, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,gamma=1-1e-8):
    updates, steps = [], []
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        step = (alpha * m_hat) / (T.sqrt(v_hat) + e)
        theta = theta_previous -  step   #(Update parameters)
        steps.append(step)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))

    return updates, steps

def shared_floatx(value, name=None, borrow=False, dtype=None):
    """Transform a value into a shared variable of type floatX.
    Parameters
    ----------
    value : :class:`~numpy.ndarray`
    The value to associate with the Theano shared.
    name : :obj:`str`, optional
    The name for the shared variable. Defaults to `None`.
    borrow : :obj:`bool`, optional
    If set to True, the given `value` will not be copied if possible.
    This can save memory and speed. Defaults to False.
    dtype : :obj:`str`, optional
    The `dtype` of the shared variable. Default value is
    :attr:`config.floatX`.
    Returns
    -------
    :class:`tensor.TensorSharedVariable`
    A Theano shared variable with the requested value and `dtype`.
    """
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
            name=name,
            borrow=borrow)

def l2_norm(tensors):
    """Computes the total L2 norm of a set of tensors.

    Converts all operands to :class:`~tensor.TensorVariable`
    (see :func:`~tensor.as_tensor_variable`).

    Parameters
    ----------
    tensors : iterable of :class:`~tensor.TensorVariable` (or compatible)
        The tensors.

    """
    flattened = [T.as_tensor_variable(t).flatten() for t in tensors]
    flattened = [(t if t.ndim > 0 else t.dimshuffle('x'))
                 for t in flattened]
    joined = T.join(0, *flattened)
    return T.sqrt(T.sqr(joined).sum())



def step_clipping(steps, threshold, to_zero=False):
    """Rescales an entire step if its L2 norm exceeds a threshold.

    When the previous steps are the gradients, this step rule performs
    gradient clipping.

    Parameters
    ----------
    threshold : float, optional
        The maximum permitted L2 norm for the step. The step
        will be rescaled to be not higher than this quanity.
        If ``None``, no rescaling will be applied.

    Attributes
    ----------
    threshold : :class:`.tensor.TensorSharedVariable`
        The shared variable storing the clipping threshold used.

    """
    # calculate the step size as the previous values minus the
    threshold = shared_floatx(threshold)

    norm = l2_norm(steps)   # return total norm
    if to_zero:
        print("clipping to zero")
        scale = 1e-8  # smallstep
    else:
        scale = threshold / norm
    multiplier = T.switch(norm < threshold,
                                1.0, scale)

    return [step*multiplier for step in steps], norm, multiplier



# copied from https://raw.githubusercontent.com/bartvm/blocks/master/blocks/algorithms/__init__.py
# to reproduce draw results...
def adam(all_grads, all_params,learning_rate=0.002,
                 beta1=0.1, beta2=0.001, epsilon=1e-8,
                 decay_factor=1e-8):
    """Adam optimizer as described in [King2014]_.

    .. [King2014] Diederik Kingma, Jimmy Ba,
       *Adam: A Method for Stochastic Optimization*,
       http://arxiv.org/abs/1412.6980

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.0002.
    beta_1 : float, optional
        Exponential decay rate for the first moment estimates.
        Default value is set to 0.1.
    beta_2 : float, optionaldddd
        Exponential decay rate for the second moment estimates.
        Default value is set to 0.001.
    epsilon : float, optional
        Default value is set to 1e-8.
    decay_factor : float, optional
        Default value is set to 1e-8.

    """

    time = shared_floatx(0., 'time')
    updates, steps = [], []
    for param, g in zip(all_params, all_grads):
        mean = shared_floatx(param.get_value() * 0., 'mean')
        variance = shared_floatx(param.get_value() * 0., 'variance')


        t1 = time + 1
        learning_rate = (learning_rate *
                         T.sqrt((1. - (1. - beta2)**t1)) /
                         (1. - (1. - beta1)**t1))
        beta_1t = 1 - (1 - beta1) * decay_factor ** (t1 - 1)
        mean_t = beta_1t * g + (1. - beta_1t) * mean
        variance_t = (beta2 * T.sqr(g) +
                      (1. - beta2) * variance)
        step = (learning_rate * mean_t /
                (T.sqrt(variance_t) + epsilon))

        updates.append((mean, mean_t))
        updates.append((variance, variance_t))
        updates.append((param, param - step))
        steps.append(step)

    updates.append((time, t1))
    return updates, steps


def plot_n_by_n_images(images,epoch=None,folder=None, n = 10, shp=[28,28]):
    """ Plot 100 MNIST images in a 10 by 10 table. Note that we crop
    the images so that they appear reasonably close together. The
    image is post-processed to give the appearance of being continued."""
    #image = np.concatenate(images, axis=1)
    i = 0
    a,b = shp
    img_out = np.zeros((a*n, b*n))
    for x in range(n):
        for y in range(n):
            xa,xb = x*a, (x+1)*b
            ya,yb = y*a, (y+1)*b
            im = np.reshape(images[i], (a,b))
            img_out[xa:xb, ya:yb] = im
            i+=1
    #matshow(img_out*100.0, cmap = matplotlib.cm.binary)
    img_out = (255*img_out).astype(np.uint8)
    img_out = Image.fromarray(img_out)
    if folder is not None and epoch is not None:
        img_out.save(os.path.join(folder,epoch + ".png"))
    return img_out

def one_hot(s, n_classes):
    y_one_of_k = np.zeros((len(s), n_classes)).astype(theano.config.floatX)
    for row,col in enumerate(s):
        y_one_of_k[row,col] = 1
    return y_one_of_k

def read_params(l,N,imgshp):
    A,B = imgshp
    center_y  = l[:,0]
    center_x  = l[:,1]   # sigmoid all thee
    log_delta = l[:,2]   # sigmoid all these
    log_sigma2 = l[:,3]
    log_gamma = l[:,4]

    delta = np.exp(log_delta)
    sigma = np.exp(log_sigma2/2.)
    gamma = np.exp(log_gamma)

    # normalize coordinates
    delta = (max(A, B)-1) / (N-1) * delta
    center_x = (center_x+1.)* (B+1)*0.5
    center_y = (center_y+1.)* (A+1)*0.5

    muX = center_x.reshape(-1,1) + delta.reshape(-1,1)*(np.arange(N).reshape(1,-1)-N/2-0.5)
    muY = center_y.reshape(-1,1)+ delta.reshape(-1,1)*(np.arange(N).reshape(1,-1)-N/2-0.5)

    return center_y, center_x, delta, sigma, gamma, muX, muY

def create_reading_square(muX, muY, imgshp = [28,28], fill=True):
        muX = np.clip(muX, 0, imgshp[1]-1)
        muY = np.clip(muY, 0, imgshp[0]-1)
        l = muX.shape[0]
        c = np.zeros([l,imgshp[0],imgshp[1]])
        muX = np.floor(muX)
        muY = np.floor(muY)
        for i in range(l):
            for y in muY[i]:
                for x in muX[i]:
                    c[i,y,x] = 1


        if fill:
            maxX, minX = np.max(muX,axis=1), np.min(muX, axis=1)
            maxY, minY = np.max(muY,axis=1), np.min(muY, axis=1)
            for i in range(l):
                c[i, minY[i]:maxY[i], minX[i]:maxX[i]] = 1

        return c