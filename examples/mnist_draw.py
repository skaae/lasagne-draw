#####
from deepmodels import batchiterator
import deepmodels
import time
import gzip
import pickle
import sys
import lasagne
from helper_functions import *
import theano
theano.optimizer_including='cudnn'
#theano.config.compute_test_value = 'raise'

DATA_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = 'mnist.pkl.gz'

NUM_EPOCHS = 50000
BATCH_SIZE = 100
NUM_EPOCHS = 1000
DIMZ = 100
GLIMPSES = 64
ENC_DEC_SIZE = 256
USE_Y = False
N_CLASSES = 10

def _load_data(url=DATA_URL, filename=DATA_FILENAME):
    PY2 = sys.version_info[0] == 2

    if PY2:
        from urllib import urlretrieve
        pickle_load = lambda f, encoding: pickle.load(f)
    else:
        from urllib.request import urlretrieve
        pickle_load = lambda f, encoding: pickle.load(f, encoding=encoding)

    if not os.path.exists(filename):
        print("Downloading MNIST")
        urlretrieve(url, filename)

    with gzip.open(filename, 'rb') as f:
        return pickle_load(f, encoding='latin-1')

print "Loading binarized data"
data = np.load('mnist_binarized.npz')
X_train = data['X_train']
X_valid = data['X_valid']
X_test = data['X_test']
print ""


N_SAMPLES_TRAIN, N_FEATURES = X_train.shape
N_SAMPLES_VAL, _ = X_valid.shape
N_SAMPLES_TEST, _ = X_test.shape

xx = X_train[101].reshape((28,28))
for row in xx:
    for col in row:
        print col,
    print ""

# build model
l_inp = lasagne.layers.InputLayer((BATCH_SIZE, N_FEATURES))
l_vae = deepmodels.layers.DrawLayer(l_inp,
                                num_units_encoder_and_decoder=ENC_DEC_SIZE,
                                glimpses = GLIMPSES,
                                dimz=DIMZ,
                                nonlinearities_out_decoder=lasagne.nonlinearities.tanh, #T.nnet.softplus,
                                nonlinearities_out_encoder=lasagne.nonlinearities.tanh, #T.nnet.softplus,
                                x_distribution='bernoulli',
                                pz_distribution='gaussianmarg',
                                qz_distribution='gaussianmarg',
                                imgshp = [28,28],
                                N_filters_read=2,
                                N_filters_write=5,
                                peepholes=True,
                                learn_hid_init=True,
                                learn_canvas_init=False,
                                n_classes=N_CLASSES,
                                use_y=USE_Y,
                                grad_clip_vals_in=[-10.0,10.0],
                                grad_clip_vals_out=[-1.0,1.0])

sym_x = T.matrix('sym_x')
sym_y = T.matrix('sym_y')
sym_x.tag.test_value = np.zeros((BATCH_SIZE, N_FEATURES), dtype='float32')
sym_y.tag.test_value = np.zeros((BATCH_SIZE, N_CLASSES), dtype='float32')
sh_x = theano.shared(np.zeros((BATCH_SIZE, N_FEATURES), dtype='float32'))
sh_y = theano.shared(np.zeros((BATCH_SIZE, N_CLASSES), dtype='float32'))


all_params = lasagne.layers.get_all_params(l_vae)
i = 0
for p in all_params:
    print i, p.name, p.shape
    i+=1
print "len all params 22", len(all_params)
print ""

givens = [(sym_x, sh_x),
          (sym_y, sh_y)]
# cost = costfun(sym_x, lower_bound)
import theano.gradient
print "Calculating updates"
x_clip = theano.gradient.grad_clip(sym_x, -10.0, 10.0) # see graves generating sequences
cost = l_vae.get_cost(x_clip, testing=False)
all_grads = theano.grad(cost,all_params)


#clip these if too big
all_grads, step_norm, multiplier = step_clipping(all_grads, threshold=10, to_zero=False)

updates,steps = adam(all_grads, all_params, learning_rate=0.0003)


sym_att_read, sym_att_write = l_vae.get_att_vals()
print "Compiling training function"
outputs =  [cost, l_vae.get_canvas(), sym_att_read, sym_att_write, l_vae.get_logx(), l_vae.get_KL(), step_norm]



train = theano.function([],outputs+ [multiplier],
                          givens=givens, updates=updates,
                           on_unused_input="warn")

eval_cost = theano.function([],outputs,
                          givens=givens,
                           on_unused_input="warn")


n_batches_train = N_SAMPLES_TRAIN / BATCH_SIZE
batchitertrain = batchiterator.BatchIterator(range(N_SAMPLES_TRAIN), BATCH_SIZE,
                           data=(X_train))
batchitertrain = batchiterator.threaded_generator(batchitertrain,3)

batchiterval = batchiterator.BatchIterator(range(X_valid.shape[0]), BATCH_SIZE,
                           data=(X_valid))
batchitertval = batchiterator.threaded_generator(batchiterval,3)

batchitertest = batchiterator.BatchIterator(range(X_test.shape[0]), BATCH_SIZE,
                           data=(X_test))
batchitertest = batchiterator.threaded_generator(batchitertest,3)
n_batches_val = N_SAMPLES_VAL / BATCH_SIZE
batches_val = [range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
               for i in range(n_batches_val)]

n_batches_test = N_SAMPLES_TEST / BATCH_SIZE
batches_test = [range(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
               for i in range(n_batches_test)]


print "Training"
load_count = 0

canvas_out, att_read_out, att_write_out = [None, None,None,None],\
                                          [None, None,None,None],\
                                          [None, None,None,None],

with open("output.log", "w") as f:
    f.write("Experiment Log\n")

for epoch in range(NUM_EPOCHS):
    conf = deepmodels.confusionmatrix.ConfusionMatrix(N_CLASSES)
    # single epoch training
    start_time = time.time()
    c = 0
    dkl_tot = 0
    logx_tot = 0
    folder = str(epoch) + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(n_batches_train):
        # update shared variables
        if USE_Y:
            X_train_batch, y_train_batch = batchitertrain.next()
            sh_y.set_value(y_train_batch, borrow=True)
        else:
            X_train_batch = batchitertrain.next()[0]
        sh_x.set_value(X_train_batch, borrow=True)

        cc, canvas, att_read, att_write, logx, dkl, total_norm, multiplier = train()
        canvas_out[i % 4] = canvas
        att_read_out[ i %4] = att_read
        att_write_out[ i %4] = att_write
        print cc, load_count, logx, dkl, total_norm, multiplier
        out_str = "cost %f logx %f dkl %f total_norm %f multiplier %f" %(cc, logx, dkl, total_norm, multiplier)
        with open("output.log", "a") as f:
            f.write(out_str + "\n")
        c += cc*BATCH_SIZE
        dkl_tot += dkl*BATCH_SIZE
        logx_tot += logx_tot*BATCH_SIZE

    c_val_tot = 0
    for i in range(n_batches_val):
        if USE_Y:
            X_val_batch, y_val_batch = batchiterval.next()
            sh_y.set_value(y_val_batch, borrow=True)
        else:
            X_val_batch = batchiterval.next()[0]
        sh_x.set_value(X_val_batch, borrow=True)
        cc_val, canvas_val, att_read_val, att_write_val, logx_val, dkl_val, total_norm_val = eval_cost()
        print cc_val
        c_val_tot += cc_val*BATCH_SIZE


    try:
        import scipy.io as sio
        sio.savemat(folder+'canvas.mat',
                    {'canvas': np.vstack(canvas_out),
                     'att_read': np.vstack(att_read_out),
                     'att_write': np.vstack(att_write_out)})
    except Exception:
        pass

    for j in range(GLIMPSES):
        epoch_str = "%03d_glimpse_%03d" % (epoch, j)
        img = plot_n_by_n_images(np.vstack(canvas_out)[:,j,:], epoch_str, folder, n=20)
    out_str = "EPOCH %i: Avg epoch cost %f KL %f logx %f cost val %f" % \
              (epoch, c/N_SAMPLES_TRAIN, dkl_tot/N_SAMPLES_TRAIN, logx_tot/N_SAMPLES_TRAIN, c_val_tot/X_valid.shape[0])
    print out_str

    with open("output.log", "a") as f:
        f.write(out_str + "\n")
