from __future__ import division, print_function
import theano
import theano.tensor as T
import numpy as np
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init
from lasagne.layers.base import *
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from draw_helpers import *
_srng = RandomStreams()


__all__ = [
    "DrawLayer"
]


class DrawLayer(Layer):
    '''
    Implements the draw model.

    The input to the model should be flattened images. Set the original
    image shape with img shp

    nb. Glorot init will not work



    REFS
    Gregor, K., Danihelka, I., Graves, A., & Wierstra, D. (2015).
    DRAW: A Recurrent Neural Network For Image Generation.
    arXiv Preprint arXiv:1502.04623.
    '''
    ini = init.Normal(std=0.01, mean=0.0)
    zero = init.Constant(0.)
    ortho = init.Orthogonal(np.sqrt(2))
    def __init__(self, input_layer, num_units_encoder_and_decoder,
                 glimpses, dimz, imgshp, N_filters_read,
                 N_filters_write,
                 W_x_to_gates=ini,
                 W_cell_to_gates=zero,
                 b_gates=zero,
                 W_read=ini,
                 b_read=zero,
                 W_write=ini,
                 b_write=zero,
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 nonlinearity_modulationgate=nonlinearities.tanh,
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearities_out_encoder = nonlinearities.tanh,
                 nonlinearities_out_decoder = nonlinearities.tanh,
                 cell_init=zero,
                 hid_init=zero,
                 canvas_init=zero,
                 W_dec_to_canvas=ini,
                 W_enc_to_mu_z=ini,
                 learn_hid_init=False,
                 learn_canvas_init=True,
                 peepholes=False,
                 x_distribution = 'bernoulli',
                 qz_distribution = 'gaussian',
                 pz_distribution = 'gaussian',
                 read_init=None,
                 n_classes=None,
                 use_y = False,
                 grad_clip_vals_out=[-1.0,1.0],
                 grad_clip_vals_in = [-10,10]
                    ):
        """
        :param input_layer: Lasagne input layer
        :param num_units_encoder_and_decoder:  Number of units in encoder and
               decoder
        :param glimpses: Number of times the networks sees and tries to
                         reconstruct the image
        :param dimz: Size of latent layer
        :param imgshp: list, [height, width]
        :param N_filters_read:  int
        :param N_filters_write: int
        :param W_x_to_gates:   function or np.ndarray or theano.shared
        :param W_cell_to_gates: function or np.ndarray or theano.shared
        :param b_gates: function or np.ndarray or theano.shared
        :param W_read:  function or np.ndarray or theano.shared
        :param b_read:  function or np.ndarray or theano.shared
        :param W_write: function or np.ndarray or theano.shared
        :param b_write: function or np.ndarray or theano.shared
        :param nonlinearity_ingate: function
        :param nonlinearity_forgetgate: function
        :param nonlinearity_modulationgate: function
        :param nonlinearity_outgate: function
        :param nonlinearities_out_encoder: function
        :param nonlinearities_out_decoder: function
        :param cell_init: function or np.ndarray or theano.shared
        :param hid_init:  function or np.ndarray or theano.shared
        :param canvas_init:  function or np.ndarray or theano.shared
        :param W_dec_to_canvas: function or np.ndarray or theano.shared
        :param W_enc_to_mu_z:   function or np.ndarray or theano.shared
        :param learn_hid_init:  boolean. If true cell and hid inits are learned
        :param learn_canvas_init: boolean. Learn canvas init. To start with a
                                 blank canvas set this to False
        :param peepholes: boolean. LSTM with or without peepholes
        :param x_distribution: str. Distribution of input data. Only supports
                                'bernoulli'
        :param qz_distribution: distribution of q(z|x), only supports
                                'gaussianmarg'
        :param pz_distribution: prior on z, p(z), only supports 'gaussianmarg'
        :param read_init: None or nd.array of length 5 with initial values
                          for reading operation. If you want to change this
                          you should probly change it so the models sees a
                          blurry version of the entire image.
        :param n_classes: int, Number if classes. required if use_y=True
        :param use_y: boolean. If true models p(x,y) else p(x)
        :param grad_clip_vals_out: Clipping of gradients with grad_clip
        :param grad_clip_vals_in: Clipping of gradients with grad_clip
        """

        # Initialize parent layer
        super(DrawLayer, self).__init__(input_layer)
        # For any of the nonlinearities, if None is supplied, use identity
        if nonlinearity_ingate is None:
            self.nonlinearity_ingate = nonlinearities.identity
        else:
            self.nonlinearity_ingate = nonlinearity_ingate

        if nonlinearity_forgetgate is None:
            self.nonlinearity_forgetgate = nonlinearities.identity
        else:
            self.nonlinearity_forgetgate = nonlinearity_forgetgate

        if nonlinearity_modulationgate is None:
            self.nonlinearity_modulationgate = nonlinearities.identity
        else:
            self.nonlinearity_modulationgate = nonlinearity_modulationgate

        if nonlinearity_outgate is None:
            self.nonlinearity_outgate = nonlinearities.identity
        else:
            self.nonlinearity_outgate = nonlinearity_outgate
        if x_distribution not in ['bernoulli']:
            raise NotImplementedError
        if pz_distribution not in ['gaussianmarg']:
            raise NotImplementedError
        if qz_distribution not in ['gaussianmarg']:
            raise NotImplementedError

        if use_y is True and n_classes is None:
            raise ValueError('n_classes must be given when use_y is true')
        self.learn_hid_init = learn_hid_init
        self.learn_canvas_init = learn_canvas_init
        self.num_units_encoder_and_decoder = num_units_encoder_and_decoder
        self.peepholes = peepholes
        self.glimpses = glimpses
        self.dimz = dimz
        self.nonlinearity_out_encoder  = nonlinearities_out_encoder
        self.nonlinearity_out_decoder = nonlinearities_out_decoder
        self.x_distribution  = x_distribution
        self.qz_distribution = qz_distribution
        self.pz_distribution = pz_distribution
        self.N_filters_read = N_filters_read
        self.N_filters_write = N_filters_write
        self.imgshp = imgshp
        self.n_classes = n_classes
        self.use_y = use_y
        self.grad_clip_vals_out = grad_clip_vals_out
        self.grad_clip_vals_in = grad_clip_vals_in


        # Input dimensionality is the output dimensionality of the input layer
        num_batch, num_inputs = self.input_layer.output_shape
        self.num_batch = num_batch
        self.num_inputs = num_inputs

        if self.peepholes:
            self.W_cellenc_to_enc_gates =  self.add_param(
                W_cell_to_gates, [3*num_units_encoder_and_decoder])
            self.W_celldec_to_dec_gates =  self.add_param(
                W_cell_to_gates, [3*num_units_encoder_and_decoder])
            self.W_cellenc_to_enc_gates.name = "DrawLayer: W_cellenc_to_enc_gates"
            self.W_celldec_to_dec_gates.name = "DrawLayer: W_celldec_to_dec_gates"
        else:
            self.W_cellenc_to_enc_gates = []
            self.W_celldec_to_dec_gates = []

        # enc
        self.b_gates_enc =  self.add_param(
            b_gates, [4*num_units_encoder_and_decoder])

        # extra input applies to both encoder and decoder
        if self.use_y:
            # if y is modelled its concatenated to the x input to the encoder
            # and the z input to the decoder. We need to expand the
            # corresponding matrices to handle this.
            extra_input = self.n_classes
        else:
            extra_input = 0

        self.W_enc_gates =  self.add_param(
            W_x_to_gates,
            [2*N_filters_read*N_filters_read+num_units_encoder_and_decoder + extra_input,
             4*num_units_encoder_and_decoder])

        self.W_hid_to_gates_enc =  self.add_param(
            W_x_to_gates, [num_units_encoder_and_decoder,
                           4*num_units_encoder_and_decoder])



        self.b_gates_dec =  self.add_param(
            b_gates, [4*num_units_encoder_and_decoder])
        self.W_z_to_gates_dec =  self.add_param(
            W_x_to_gates, [dimz + extra_input, 4*num_units_encoder_and_decoder])
        self.W_hid_to_gates_dec =  self.add_param(
            W_x_to_gates, [num_units_encoder_and_decoder, 4*num_units_encoder_and_decoder])


        # Setup initial values for the cell and the lstm hidden units
        if self.learn_hid_init:
            self.cell_init_enc = self.add_param(
                cell_init, (1, num_units_encoder_and_decoder))
            self.hid_init_enc = self.add_param(
                hid_init, (1, num_units_encoder_and_decoder))
            self.cell_init_dec = self.add_param(
                cell_init, (1, num_units_encoder_and_decoder))
            self.hid_init_dec = self.add_param(
                hid_init, (1, num_units_encoder_and_decoder))

        else:  # init at zero + they will not be returned as parameters
            self.cell_init_enc = T.zeros((1, num_units_encoder_and_decoder))
            self.hid_init_enc = T.zeros((1, num_units_encoder_and_decoder))
            self.cell_init_dec = T.zeros((1, num_units_encoder_and_decoder))
            self.hid_init_dec = T.zeros((1, num_units_encoder_and_decoder))

        if self.learn_canvas_init:
            self.canvas_init = self.add_param(canvas_init, (1, num_inputs))
        else:
            self.canvas_init = T.zeros((1, num_inputs))

        # decoder to canvas
        self.W_dec_to_canvas_patch = self.add_param(
            W_dec_to_canvas, (num_units_encoder_and_decoder,
                              N_filters_write*N_filters_write))


        # variational weights
        # TODO: Make the sizes more flexible, they are not required to be equal

        self.W_enc_to_z_mu = self.add_param(
            W_enc_to_mu_z, (self.num_units_encoder_and_decoder, self.dimz))
        self.b_enc_to_z_mu = self.add_param(b_gates, (self.dimz,))
        self.W_enc_to_z_sigma = self.add_param(
            W_enc_to_mu_z, (self.num_units_encoder_and_decoder, self.dimz))
        self.b_enc_to_z_sigma = self.add_param(b_gates, (self.dimz,))

        self.b_gates_enc.name = "DrawLayer: b_gates_enc"
        self.b_gates_dec.name = "DrawLayer: b_gates_dec"
        self.W_enc_gates.name = "DrawLayer: W_x_to_gates_enc"
        self.W_hid_to_gates_enc.name = "DrawLayer: W_hid_to_gates_enc"
        self.W_z_to_gates_dec.name = "DrawLayer: W_z_to_gates_dec"
        self.W_hid_to_gates_dec.name = "DrawLayer: W_hid_to_gates_dec"
        self.W_enc_to_z_mu.name = "DrawLayer: W_enc_to_z_mu"
        self.b_enc_to_z_mu.name = "DrawLayer: b_enc_to_z_mu"
        self.W_enc_to_z_sigma.name = "DrawLayer: W_enc_to_z_sigma"
        self.b_enc_to_z_sigma.name = "DrawLayer: b_enc_to_z_sigma"
        self.W_dec_to_canvas_patch.name = "DrawLayer: W_dec_to_canvas"

        self.cell_init_enc.name = "DrawLayer: cell_init_enc"
        self.hid_init_enc.name = "DrawLayer: hid_init_enc"
        self.cell_init_dec.name = "DrawLayer: cell_init_dec"
        self.hid_init_dec.name = "DrawLayer: hid_init_dec"
        self.canvas_init.name = "DrawLayer: canvas_init"

        # init values for read operation.
        delta_read = 1  #
        gamma = 1.0
        sigma_read = 1.0
        center_y = 0.
        center_x = 0.
        if read_init is None:
            read_init = np.array([[center_y,
                               center_x,
                               np.log(delta_read),
                               np.log(sigma_read),
                               np.log(gamma)]])
            read_init = read_init.astype(theano.config.floatX)
        print("Read init is", read_init)

        self.W_read = self.add_param(W_read,
                                        (num_units_encoder_and_decoder, 5))
        self.W_write = self.add_param(W_write,
                                         (num_units_encoder_and_decoder, 5))
        self.b_read = self.add_param(b_read, (5,))
        self.b_write = self.add_param(b_write, (5,))
        self.read_init = self.add_param(read_init, (1,5))
        self.W_read.name = "DrawLayer: W_read"
        self.W_write.name = "DrawLayer: W_write"
        self.b_read.name = "DrawLayer: b_read"
        self.b_write.name = "DrawLayer: b_write"

    def get_read_init(self):
        return self.read_init

    def get_params(self):
        '''
        Get all parameters of this layer.

        :returns:
            - params : list of theano.shared
                List of all parameters
        '''
        params = self.get_weight_params() + self.get_bias_params()
        if self.peepholes:
            params.extend(self.get_peephole_params())

        if self.learn_hid_init:
            params.extend(self.get_init_params())

        if self.learn_canvas_init:
            params += [self.canvas_init]

        return params

    def get_weight_params(self):
        '''
        Get all weights of this layer
        :returns:
            - weight_params : list of theano.shared
                List of all weight parameters
        '''
        return [self.W_enc_gates,
                self.W_hid_to_gates_enc,
                self.W_z_to_gates_dec,
                self.W_hid_to_gates_dec,
                self.W_dec_to_canvas_patch,
                self.W_enc_to_z_mu,
                self.W_enc_to_z_sigma,
                self.W_read,
                self.W_write
        ]

    def get_peephole_params(self):
        '''
        Get all peephole parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all peephole parameters
        '''
        return [self.W_cellenc_to_enc_gates, self.W_celldec_to_dec_gates]

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        if self.learn_hid_init:
            params =  [self.hid_init_enc, self.cell_init_enc,
                self.hid_init_dec, self.cell_init_dec]
        else:
            params = []
        return params

    def get_bias_params(self):
        '''
        Get all bias parameters of this layer.

        :returns:
            - bias_params : list of theano.shared
                List of all bias parameters
        '''
        params =  [self.b_gates_enc,self.b_gates_dec,
                   self.b_enc_to_z_mu,
                   self.b_enc_to_z_sigma,
                   self.b_read, self.b_write]

        return params

    def get_output_shape_for(self, input_shape):
        '''
        Compute the expected output shape given the input.

        :parameters:
            - input_shape : tuple
                Dimensionality of expected input

        :returns:
            - output_shape : tuple
                Dimensionality of expected outputs given input_shape
        '''
        return self.input_shape

    def _lstm(self, gates, cell_previous,W_cell_to_gates, nonlinearity_out):
        # LSTM step
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def slice_w(x, n):
            start = n*self.num_units_encoder_and_decoder
            stop = (n+1)*self.num_units_encoder_and_decoder
            return x[:, start:stop]

        def slice_c(x, n):
            start = n*self.num_units_encoder_and_decoder
            stop = (n+1)*self.num_units_encoder_and_decoder
            return x[start:stop]

        def clip(x):
            return theano.gradient.grad_clip(
                x, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

        ingate = slice_w(gates, 0)
        forgetgate = slice_w(gates, 1)
        modulationgate = slice_w(gates, 2)
        outgate = slice_w(gates, 3)

        if self.peepholes:
            ingate += cell_previous*slice_c(W_cell_to_gates, 0)
            forgetgate += cell_previous*slice_c(W_cell_to_gates, 1)

        if self.grad_clip_vals_in is not None:
            print('STEP: CLipping gradients IN', self.grad_clip_vals_in)
            ingate = clip(ingate)
            forgetgate = clip(forgetgate)
            modulationgate = clip(modulationgate)
        ingate = self.nonlinearity_ingate(ingate)
        forgetgate = self.nonlinearity_forgetgate(forgetgate)
        modulationgate = self.nonlinearity_modulationgate(modulationgate)
        if self.grad_clip_vals_in is not None:
            ingate = clip(ingate)
            forgetgate = clip(forgetgate)
            modulationgate = clip(modulationgate)

        cell = forgetgate*cell_previous + ingate*modulationgate
        if self.peepholes:
            outgate += cell*slice_c(W_cell_to_gates, 2)

        if self.grad_clip_vals_in is not None:
            outgate = clip(outgate)

        outgate = self.nonlinearity_outgate(outgate)
        if self.grad_clip_vals_in is not None:
            outgate = clip(outgate)

        hid = outgate*nonlinearity_out(cell)
        return [cell, hid]

    def get_cost(self, x, y=None, *args, **kwargs):
        """
        Compute layer cost.

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        """



        if y is None and self.use_y is True:
                raise ValueError('y must be given when use_y is true')

        def step(eps_n,
                 ######### REUCCRENT
                 cell_previous_enc, hid_previous_enc,
                 cell_previous_dec, hid_previous_dec,
                 canvas_previous, mu_z_previous, log_sigma_z_previous,
                 z_previous, l_read_previous, l_write_previous,
                 #kl_previous,
                 ######### x and WEIGHTS
                 x, y,
                 W_enc_gates,
                 W_hid_to_gates_enc,
                 b_gates_enc,
                 W_cellenc_to_enc_gates,
                 W_read, b_read,
                 W_z_to_gates_dec, b_gates_dec,
                 W_hid_to_gates_dec,

                 W_celldec_to_dec_gates,
                 W_enc_to_z_mu, b_enc_to_z_mu,
                 W_enc_to_z_sigma, b_enc_to_z_sigma,
                 W_dec_to_canvas_patch, W_write, b_write,
                 ):
            # calculate gates pre-activations and slice
            N_read = self.N_filters_read
            N_write = self.N_filters_write
            img_shp = self.imgshp

            x_err = x - T.nnet.sigmoid(canvas_previous)
            att_read = nn2att(l_read_previous, N_read,  img_shp)
            x_org_in, x_err_in = read(x,x_err, att_read, N_read, img_shp)

            x_org_in = att_read['gamma']*x_org_in
            x_err_in = att_read['gamma']*x_err_in

            if self.use_y:
                in_gates_enc = T.concatenate([y, x_org_in, x_err_in,hid_previous_dec], axis=1)
            else:
                in_gates_enc = T.concatenate([x_org_in, x_err_in,hid_previous_dec], axis=1)

            # equation (5)~ish
            #slice_gates_idx = 4*self.num_units_encoder_and_decoder
            # ENCODER
            gates_enc = T.dot(in_gates_enc, W_enc_gates) + b_gates_enc
            gates_enc += T.dot(hid_previous_enc, W_hid_to_gates_enc)
            #gates_enc +=T.dot(hid_previous_enc, W_hidenc_to_enc_gates)
            cell_enc, hid_enc = self._lstm(gates_enc,
                                     cell_previous_enc,
                                     W_cellenc_to_enc_gates,
                                     self.nonlinearity_out_encoder)

            # VARIATIONAL
            # eq 6
            mu_z = T.dot(hid_enc, W_enc_to_z_mu) + b_enc_to_z_mu
            log_sigma_z = 0.5*(T.dot(hid_enc, W_enc_to_z_sigma) + b_enc_to_z_sigma)
            z = mu_z + T.exp(log_sigma_z)*eps_n

            if self.use_y:
                print('STEP: using Y')
                in_gates_dec = T.concatenate([y, z], axis=1)
            else:
                print('STEP: Not using Y')
                in_gates_dec = z


            # DECODER
            gates_dec = T.dot(in_gates_dec, W_z_to_gates_dec) + b_gates_dec  # i_dec
            gates_dec += T.dot(hid_previous_dec, W_hid_to_gates_dec)
            # equation (7)
            cell_dec, hid_dec = self._lstm(gates_dec,
                                     cell_previous_dec,
                                     W_celldec_to_dec_gates,
                                     self.nonlinearity_out_decoder)

            # WRITE
            l_write = T.dot(hid_dec, W_write) + b_write
            w = T.dot(hid_dec, W_dec_to_canvas_patch)
            att_write = nn2att(l_write, N_write,  img_shp)
            canvas_upd = write(w, att_write, N_write, img_shp)
            canvas_upd = 1.0/(att_write['gamma']+1e-4) * canvas_upd
            canvas = canvas_previous + canvas_upd

            l_read = T.dot(hid_dec, W_read) + b_read

            # Todo: some of the (all?) gradient clips are redundant
            # + I'm unsure if I use grad_clip correct and in correct places...
            # The description of gradient clipping is in
            # Generating sequences with recurrent neural networks
            # section: 2.1 Long Short-Term Memory
            #
            #if self.grad_clip_vals_out is not None:
            #    print('STEP: CLipping gradients Out', self.grad_clip_vals_out)
            #    cell_enc = theano.gradient.grad_clip(cell_enc, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
            #    hid_enc = theano.gradient.grad_clip(hid_enc, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
            #    cell_dec = theano.gradient.grad_clip(cell_dec, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
            #    hid_dec = theano.gradient.grad_clip(hid_dec, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])

            return [cell_enc, hid_enc, cell_dec, hid_dec, canvas,
                    mu_z, log_sigma_z, z, l_read, l_write]


        ones = T.ones((self.num_batch,1))
        mu_z_init = T.zeros((self.num_batch, self.dimz))
        log_sigma_z_init = T.zeros((self.num_batch, self.dimz))
        z_init = T.zeros((self.num_batch, self.dimz))
        att_vals_write_init = T.zeros((self.num_batch, 5))

        if theano.config.compute_test_value is 'off':
            eps = _srng.normal((self.glimpses,self.num_batch))
        else:
            # for testing
            print("draw.py: is not using random generator"+"!#>"*30)
            eps = T.ones((self.glimpses,self.num_batch), theano.config.floatX) * 0.3

        if y is None:
            y = T.zeros((1))


        # Todo: cleanup this somehow
        # Todo: Will it slow down theano optimization if I dont pass in
        # non seqs as arguments, but just call them with self.XXXX?
        seqs = [eps]
        init = [T.dot(ones, self.cell_init_enc), T.dot(ones, self.hid_init_enc),
                T.dot(ones, self.cell_init_dec), T.dot(ones, self.hid_init_dec),
                T.dot(ones, self.canvas_init), mu_z_init, log_sigma_z_init,
                z_init, T.dot(ones, self.read_init), att_vals_write_init]
        nonseqs_input = [x, y]
        nonseqs_enc = [self.W_enc_gates,
                       self.W_hid_to_gates_enc,
                       self.b_gates_enc,
                       self.W_cellenc_to_enc_gates,
                       self.W_read, self.b_read]
        nonseqs_dec = [self.W_z_to_gates_dec, self.b_gates_dec,
                       self.W_hid_to_gates_dec,
                       self.W_celldec_to_dec_gates]
        nonseqs_variational = [self.W_enc_to_z_mu, self.b_enc_to_z_mu,
                               self.W_enc_to_z_sigma, self.b_enc_to_z_sigma]
        nonseqs_other = [self.W_dec_to_canvas_patch, self.W_write, self.b_write]
        non_seqs = nonseqs_input +  nonseqs_enc + nonseqs_dec + nonseqs_variational \
                   + nonseqs_other

        output_scan = theano.scan(step, sequences=seqs,
                             outputs_info=init,
                             non_sequences=non_seqs,
                             go_backwards=False)[0]


        cell_enc, hid_enc, cell_dec, hid_dec, canvas, mu_z, log_sigma_z, \
        z, l_read, l_write = output_scan

        # because we model the output as bernoulli we take sigmoid to ensure
        # range (0,1)
        last_reconstruction = T.nnet.sigmoid(canvas[-1, :, :])
        # select distribution of p(x|z)

        # LOSS
        # The loss is the negative loglikelihood of the data plus the
        # KL divergence between the the variational approximation to z and
        # the prior on z:
        # Loss = -logD(x) + D_kl(Q(z|h)||p(z))
        # If we assume that x is bernoulli then
        # -logD(x) = -(t*log(o) +(1-t)*log(1-o)) = cross_ent(t,o)
        # D_kl(Q(z|h)||p(z)) can in some cases be solved analytically as
        # D_kl(Q(z|h)||p(z)) = 0.5(sum_T(mu^2 + sigma^2 - 1 -log(sigma^2)))
        # We add these terms and return minus the cost, i.e return the
        # lowerbound

        L_x = T.nnet.binary_crossentropy(last_reconstruction, x).sum()
        #L_x = cross_ent(last_reconstruction, x).sum()
        L_z =  T.sum(0.5*( mu_z**2 + T.exp(log_sigma_z*2) - 1 - log_sigma_z*2))
        self.L_x = L_x
        self.L_z = L_z
        L =  L_x + L_z

        self.canvas = canvas
        self.att_vals_read = l_read
        self.att_vals_write = l_write

        return L / self.num_batch

    def get_canvas(self):
        return T.nnet.sigmoid(self.canvas.dimshuffle(1,0,2))

    def get_att_vals(self):
        return self.att_vals_read.dimshuffle(1,0,2), \
               self.att_vals_write.dimshuffle(1,0,2)

    def get_logx(self):
        return self.L_x / self.num_batch

    def get_KL(self):
        return self.L_z / self.num_batch

    def generate(self, n_digits, y=None, *args, **kwargs):
        '''

        Generate digits see http://arxiv.org/abs/1502.04623v1 section 2.3

        '''

        if y is None and self.use_y is True:
                raise ValueError('y must be given when use_y is true')
        def step(z,
                 cell_previous_dec, hid_previous_dec,
                 canvas_previous, l_write_previous,
                 y,
                 W_z_to_gates_dec, b_gates_dec,
                 W_hid_to_gates_dec,
                 W_celldec_to_dec_gates,
                 W_dec_to_canvas_patch, W_write, b_write
                 ):
            N_write =self.N_filters_write
            img_shp = self.imgshp


            # DECODER
            if self.use_y:
                print('STEP: using Y')
                in_gates_dec = T.concatenate([y, z], axis=1)
            else:
                print('STEP: Not using Y')
                in_gates_dec = z


            gates_dec = T.dot(in_gates_dec, W_z_to_gates_dec) + b_gates_dec
            gates_dec += T.dot(hid_previous_dec, W_hid_to_gates_dec)
            # equation (7)
            cell_dec, hid_dec = self._lstm(gates_dec,
                                     cell_previous_dec,
                                     W_celldec_to_dec_gates,
                                     self.nonlinearity_out_decoder)

            # WRITE
            l_write = T.dot(hid_dec, W_write) + b_write
            w = T.dot(hid_dec, W_dec_to_canvas_patch)
            att_write = nn2att(l_write, N_write,  img_shp)
            canvas_upd = write(w, att_write, N_write, img_shp)
            canvas_upd = 1.0/(att_write['gamma']+1e-4) * canvas_upd
            canvas = canvas_previous + canvas_upd

            return [cell_dec, hid_dec, canvas, l_write]


        ones = T.ones((n_digits,1))
        if theano.config.compute_test_value is 'off':
            z_samples = _srng.normal((self.glimpses,n_digits,self.dimz))
        else:
            print("draw.py: is not using random generator"+"!#>"*30)
            z_samples = T.ones(
                (self.glimpses,n_digits,self.dimz), theano.config.floatX) * 0.3

        if y is None:
            y = T.zeros((1))
        att_vals_write_init = T.zeros((n_digits, 5))
        seqs = [z_samples]
        init = [T.dot(ones, self.cell_init_dec), T.dot(ones, self.hid_init_dec),
                T.dot(ones, self.canvas_init),att_vals_write_init]
        non_seqs = [y, self.W_z_to_gates_dec, self.b_gates_dec,
                 self.W_hid_to_gates_dec,
                 self.W_celldec_to_dec_gates,
                 self.W_dec_to_canvas_patch,
                 self.W_write, self.b_write]

        output_scan = theano.scan(step, sequences=seqs,
                             outputs_info=init,
                             non_sequences=non_seqs,
                             go_backwards=False)[0]


        canvas = output_scan[2]
        l_write = output_scan[3]
        return T.nnet.sigmoid(canvas.dimshuffle(1,0,2)), l_write.dimshuffle(1,0,2)
