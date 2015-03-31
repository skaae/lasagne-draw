from __future__ import division, print_function
import theano
import theano.tensor as T
import numpy as np
import lasagne.nonlinearities as nonlinearities
import lasagne.init as init
from .base import Layer
from .. import logdists
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from draw_helpers import *
_srng = RandomStreams()


__all__ = [
    "DrawLayer",
    "DRAMLayer"
]


class DrawLayer(Layer):
    '''
    DRAW model from the paper:
    Gregor, K., Danihelka, I., Graves, A., & Wierstra, D. (2015).
    DRAW: A Recurrent Neural Network For Image Generation.
    arXiv Preprint arXiv:1502.04623.
    '''
    #ini = init.Uniform((-0.05, 0.05))
    ini = init.Normal(std=0.01, avg=0.0)
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
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_gates : function or np.ndarray or theano.shared
            - W_hid_to_gates : function or np.ndarray or theano.shared
            - W_cell_to_gates : function or np.ndarray or theano.shared
            - b_gates : function or np.ndarray or theano.shared
            - nonlinearity_ingate : function
            - nonlinearity_forgetgate : function
            - nonlinearity_modulationgate : function
            - nonlinearity_outgate : function
            - nonlinearity_out : function
            - cell_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - highforgetbias: if true forget gates will be initalized to
            range [20, 25] i.e large positive value
        '''

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
        if pz_distribution not in ['gaussian', 'gaussianmarg']:
            raise NotImplementedError
        if qz_distribution not in ['gaussian', 'gaussianmarg']:
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
        num_batch, num_inputs = self.input_layer.get_output_shape()
        self.num_batch = num_batch
        self.num_inputs = num_inputs

        if self.peepholes:
            self.W_cellenc_to_enc_gates =  self.create_param(
                W_cell_to_gates, [3*num_units_encoder_and_decoder])
            self.W_celldec_to_dec_gates =  self.create_param(
                W_cell_to_gates, [3*num_units_encoder_and_decoder])
            self.W_cellenc_to_enc_gates.name = "DrawLayer: W_cellenc_to_enc_gates"
            self.W_celldec_to_dec_gates.name = "DrawLayer: W_celldec_to_dec_gates"
        else:
            self.W_cellenc_to_enc_gates = []
            self.W_celldec_to_dec_gates = []

        # enc
        self.b_gates_enc =  self.create_param(b_gates, [4*num_units_encoder_and_decoder])

        # extra input applies to both encoder and decoder
        if self.use_y:
            extra_input = self.n_classes
        else:
            extra_input = 0

        self.W_enc_gates =  self.create_param(
            W_x_to_gates,
            [2*N_filters_read*N_filters_read+num_units_encoder_and_decoder + extra_input,
             4*num_units_encoder_and_decoder])

        self.W_hid_to_gates_enc =  self.create_param(
            W_x_to_gates, [num_units_encoder_and_decoder, 4*num_units_encoder_and_decoder])



        self.b_gates_dec =  self.create_param(b_gates, [4*num_units_encoder_and_decoder])
        self.W_z_to_gates_dec =  self.create_param(
            W_x_to_gates, [dimz + extra_input, 4*num_units_encoder_and_decoder])
        self.W_hid_to_gates_dec =  self.create_param(
            W_x_to_gates, [num_units_encoder_and_decoder, 4*num_units_encoder_and_decoder])


        # Setup initial values for the cell and the lstm hidden units
        if self.learn_hid_init:
            self.cell_init_enc = self.create_param(cell_init, (1, num_units_encoder_and_decoder))
            self.hid_init_enc = self.create_param(hid_init, (1, num_units_encoder_and_decoder))
            self.cell_init_dec = self.create_param(cell_init, (1, num_units_encoder_and_decoder))
            self.hid_init_dec = self.create_param(hid_init, (1, num_units_encoder_and_decoder))

        else:
            self.cell_init_enc = T.zeros((1, num_units_encoder_and_decoder))
            self.hid_init_enc = T.zeros((1, num_units_encoder_and_decoder))
            self.cell_init_dec = T.zeros((1, num_units_encoder_and_decoder))
            self.hid_init_dec = T.zeros((1, num_units_encoder_and_decoder))

        if self.learn_canvas_init:
            self.canvas_init = self.create_param(canvas_init, (1, num_inputs))
        else:
            self.canvas_init = T.zeros((1, num_inputs))

        # decoder to canvas
        self.W_dec_to_canvas_patch = self.create_param(W_dec_to_canvas, (num_units_encoder_and_decoder, N_filters_write*N_filters_write))


        # variational weights
        # TODO: Make the sizes more flexible, they are not required to be equal

        self.W_enc_to_z_mu = self.create_param(W_enc_to_mu_z, (self.num_units_encoder_and_decoder, self.dimz))
        self.b_enc_to_z_mu = self.create_param(b_gates, (self.dimz))
        self.W_enc_to_z_sigma = self.create_param(W_enc_to_mu_z, (self.num_units_encoder_and_decoder, self.dimz))
        self.b_enc_to_z_sigma = self.create_param(b_gates, (self.dimz))

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

        self.W_read = self.create_param(W_read, (num_units_encoder_and_decoder, 5))
        self.W_write = self.create_param(W_write, (num_units_encoder_and_decoder, 5))
        self.b_read = self.create_param(b_read, (5))
        self.b_write = self.create_param(b_write, (5))
        self.read_init = self.create_param(read_init, (1,5))
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

    def get_cost(self, x, y=None, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''
        if y is None and self.use_y is True:
            raise ValueError('y must be given when use_y is true')

        def slice_w(x, n):
            return x[:, n*self.num_units_encoder_and_decoder:(n+1)*self.num_units_encoder_and_decoder]

        def slice_c(x, n):
            return x[n*self.num_units_encoder_and_decoder:(n+1)*self.num_units_encoder_and_decoder]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th row of the input dot W multiplication
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        #
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def lstm(gates, cell_previous,W_cell_to_gates,nonlinearity_out):
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            modulationgate = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*slice_c(W_cell_to_gates, 0)
                forgetgate += cell_previous*slice_c(W_cell_to_gates, 1)

            if self.grad_clip_vals_in is not None:
                print('STEP: CLipping gradients IN', self.grad_clip_vals_in)
                ingate = theano.gradient.grad_clip(ingate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                forgetgate = theano.gradient.grad_clip(forgetgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                modulationgate = theano.gradient.grad_clip(modulationgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            modulationgate = self.nonlinearity_modulationgate(modulationgate)
            if self.grad_clip_vals_in is not None:
                ingate = theano.gradient.grad_clip(ingate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                forgetgate = theano.gradient.grad_clip(forgetgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                modulationgate = theano.gradient.grad_clip(modulationgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            cell = forgetgate*cell_previous + ingate*modulationgate
            if self.peepholes:
                outgate += cell*slice_c(W_cell_to_gates, 2)

            if self.grad_clip_vals_in is not None:
                outgate = theano.gradient.grad_clip(outgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            outgate = self.nonlinearity_outgate(outgate)
            if self.grad_clip_vals_in is not None:
                outgate = theano.gradient.grad_clip(outgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            hid = outgate*nonlinearity_out(cell)
            return [cell, hid]


        def step(eps_n,                                  # standardnormal values
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




            # TODO: add read (eq 4) current: "read" whole input

            # equation (5)~ish
            #slice_gates_idx = 4*self.num_units_encoder_and_decoder
            # ENCODER
            gates_enc = T.dot(in_gates_enc, W_enc_gates) + b_gates_enc
            gates_enc += T.dot(hid_previous_enc, W_hid_to_gates_enc)
            #gates_enc +=T.dot(hid_previous_enc, W_hidenc_to_enc_gates)
            cell_enc, hid_enc = lstm(gates_enc,
                                     cell_previous_enc,
                                     W_cellenc_to_enc_gates,
                                     self.nonlinearity_out_encoder)

            # VARIATIONAL
            # eq 6
            mu_z = T.dot(hid_enc, W_enc_to_z_mu) + b_enc_to_z_mu
            log_sigma_z = 0.5*T.dot(hid_enc, W_enc_to_z_sigma) + b_enc_to_z_sigma
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
            cell_dec, hid_dec = lstm(gates_dec,
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
            if self.grad_clip_vals_out is not None:
                print('STEP: CLipping gradients Out', self.grad_clip_vals_out)
                cell_enc = theano.gradient.grad_clip(cell_enc, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                hid_enc = theano.gradient.grad_clip(hid_enc, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                cell_dec = theano.gradient.grad_clip(cell_dec, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                hid_dec = theano.gradient.grad_clip(hid_dec, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])

            return [cell_enc, hid_enc, cell_dec, hid_dec, canvas,
                    mu_z, log_sigma_z, z, l_read, l_write]


        ones = T.ones((self.num_batch,1))
        mu_z_init = T.zeros((self.num_batch, self.dimz))
        log_sigma_z_init = T.zeros((self.num_batch, self.dimz))
        z_init = T.zeros((self.num_batch, self.dimz))
        att_vals_write_init = T.zeros((self.num_batch, 5))

        #kl_init = T.zeros((self.num_batch))
        if theano.config.compute_test_value is 'off':
            eps = _srng.normal((self.glimpses,self.num_batch))
        else:
            print("lstm.py: is not using random generator")
            eps = T.ones((self.glimpses,self.num_batch), theano.config.floatX) * 0.3

        if y is None:
            y = T.zeros((1))

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
        nonseqs_variational = [self.W_enc_to_z_mu, self.b_enc_to_z_mu, self.W_enc_to_z_sigma, self.b_enc_to_z_sigma]
        nonseqs_other = [self.W_dec_to_canvas_patch, self.W_write, self.b_write]
        non_seqs = nonseqs_input +  nonseqs_enc + nonseqs_dec + nonseqs_variational \
                   + nonseqs_other

        output_scan = theano.scan(step, sequences=seqs,
                             outputs_info=init,
                             non_sequences=non_seqs,
                             go_backwards=False)[0]


        cell_enc, hid_enc, cell_dec, hid_dec, canvas, mu_z, log_sigma_z, z, l_read, l_write = output_scan

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


        #def cross_ent(output, target, tol=1e-8):
        #    # -log(p(x|z))
        #    return -(target * T.log(output+tol) +
        #             (1.0 - target) * T.log(1.0+tol - output))
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
        return self.att_vals_read.dimshuffle(1,0,2), self.att_vals_write.dimshuffle(1,0,2)

    def get_logx(self):
        return self.L_x / self.num_batch

    def get_KL(self):
        return self.L_z / self.num_batch

    def generate(self, n_digits, *args, **kwargs):
        '''

        Generate digits see http://arxiv.org/abs/1502.04623v1 section 2.3
        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''


        def slice_w(x, n):
            return x[:, n*self.num_units_encoder_and_decoder:(n+1)*self.num_units_encoder_and_decoder]

        def slice_c(x, n):
            return x[n*self.num_units_encoder_and_decoder:(n+1)*self.num_units_encoder_and_decoder]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th row of the input dot W multiplication
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        #
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def lstm(gates, cell_previous,W_cell_to_gates,nonlinearity_out):

            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            modulationgate = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*slice_c(W_cell_to_gates, 0)
                forgetgate += cell_previous*slice_c(W_cell_to_gates, 1)

            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            modulationgate = self.nonlinearity_modulationgate(modulationgate)

            cell = forgetgate*cell_previous + ingate*modulationgate
            if self.peepholes:
                outgate += cell*slice_c(W_cell_to_gates, 2)
            outgate = self.nonlinearity_outgate(outgate)
            hid = outgate*nonlinearity_out(cell)
            return [cell, hid]


        def step(z_sample,
                 cell_previous_dec, hid_previous_dec,
                 canvas_previous,
                 W_z_to_gates_dec, b_gates_dec,
                 W_hid_to_gates_dec,
                 W_celldec_to_dec_gates,
                 W_dec_to_canvas_patch, W_write, b_write
                 ):
            N_write =self.N_filters_write
            img_shp = self.imgshp


            # DECODER
            gates_dec = T.dot(z_sample, W_z_to_gates_dec) + b_gates_dec  # i_dec
            gates_dec += T.dot(hid_previous_dec, W_hid_to_gates_dec)
            # equation (7)
            cell_dec, hid_dec = lstm(gates_dec,
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
            z_samples = _srng.uniform((self.glimpses,n_digits,self.dimz))
        else:
            print("lstm.py: is not using random generator")
            z_samples = T.ones((self.glimpses,self.dimz), theano.config.floatX) * 0.3



        seqs = [z_samples]
        init = [T.dot(ones, self.cell_init_dec), T.dot(ones, self.hid_init_dec),
                T.dot(ones, self.canvas_init)]
        non_seqs = [ self.W_z_to_gates_dec, self.b_gates_dec,
                 self.W_hid_to_gates_dec,
                 self.W_celldec_to_dec_gates,
                 self.W_dec_to_canvas_patch,
                 self.W_write, self.b_write]



        output_scan = theano.scan(step, sequences=seqs,
                             outputs_info=init,
                             non_sequences=seqs + init + non_seqs,
                             go_backwards=False)[0]


        canvas = output_scan[2]
        return T.nnet.sigmoid(canvas.dimshuffle(1,0,2))


class DRAMLayer(Layer):
    '''
        Differentiable RAM model as it is described in the paper:

        Gregor, K., Danihelka, I., Graves, A., & Wierstra, D. (2015).
        DRAW: A Recurrent Neural Network For Image Generation.
        arXiv Preprint arXiv:1502.04623.
    '''
    #ini = init.Uniform((-0.05, 0.05))
    ini = init.Normal(std=0.01, avg=0.0)
    zero = init.Constant(0.)
    ortho = init.Orthogonal(np.sqrt(2))
    def __init__(self, input_layer, num_units, n_classes,
                 glimpses, imgshp,
                 N_filters_read,
                 W_x_to_gates=ini,
                 W_hid_to_gates=ini,
                 W_cell_to_gates=zero,
                 b_gates=zero,
                 W_read=ini,
                 b_read=zero,
                 W_y=ini,
                 b_y = ini,
                 nonlinearity_ingate=nonlinearities.sigmoid,
                 nonlinearity_forgetgate=nonlinearities.sigmoid,
                 nonlinearity_modulationgate=nonlinearities.tanh,
                 nonlinearity_outgate=nonlinearities.sigmoid,
                 nonlinearities_out = nonlinearities.tanh,
                 cell_init=zero,
                 hid_init=zero,
                 learn_hid_init=False,
                 peepholes=False,
                 return_sequence=False,
                 two_layer=True,
                 read_init=None,
                 n_units_input_net=200,
                 use_loc_as_input=True,
                 use_memory_as_output=True,
                 mem_size = 20,
                 grad_clip_vals_out=[-1.0,1.0],
                 grad_clip_vals_in = [-10,10]
                    ):
        '''
        Initialize an LSTM layer.  For details on what the parameters mean, see
        (7-11) from [#graves2014generating]_.

        :parameters:
            - input_layer : layers.Layer
                Input to this recurrent layer
            - num_units : int
                Number of hidden units
            - W_in_to_gates : function or np.ndarray or theano.shared
            - W_hid_to_gates : function or np.ndarray or theano.shared
            - W_cell_to_gates : function or np.ndarray or theano.shared
            - b_gates : function or np.ndarray or theano.shared
            - nonlinearity_ingate : function
            - nonlinearity_forgetgate : function
            - nonlinearity_modulationgate : function
            - nonlinearity_outgate : function
            - nonlinearity_out : function
            - cell_init : function or np.ndarray or theano.shared
                :math:`c_0`
            - hid_init : function or np.ndarray or theano.shared
                :math:`h_0`
            - learn_init : boolean
                If True, initial hidden values are learned
            - peepholes : boolean
                If True, the LSTM uses peephole connections.
                When False, W_cell_to_ingate, W_cell_to_forgetgate and
                W_cell_to_outgate are ignored.
            - highforgetbias: if true forget gates will be initalized to
            range [20, 25] i.e large positive value
        '''

        # Initialize parent layer
        super(DRAMLayer, self).__init__(input_layer)
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


        self.learn_hid_init = learn_hid_init
        self.num_units = num_units
        self.peepholes = peepholes
        self.glimpses = glimpses
        self.nonlinearity_out  = nonlinearities_out
        self.N_filters_read = N_filters_read
        self.imgshp = imgshp
        self.n_classes = n_classes
        self.return_sequence=return_sequence
        self.two_layer = two_layer
        self.n_units_input_net = n_units_input_net
        self.use_loc_as_input = use_loc_as_input
        self.use_memory_as_output = use_memory_as_output
        self.mem_size = mem_size
        self.grad_clip_vals_out = grad_clip_vals_out
        self.grad_clip_vals_in = grad_clip_vals_in

        # Input dimensionality is the output dimensionality of the input layer
        num_batch, num_inputs = self.input_layer.get_output_shape()
        self.num_batch = num_batch
        self.num_inputs = num_inputs

        if self.peepholes:
            self.W_cell_to_gates1 =  self.create_param(
                W_cell_to_gates, [3*num_units])
            self.W_cell_to_gates1.name = "DRAMLayer: W_cell_to_gates1"
        else:
            self.W_cell_to_gates1 = []

        # enc
        if self.use_loc_as_input:
            #add_input_x = self.n_units_input_net
            add_input_x = 0  # we new use additive interaction
        else:
            add_input_x = 0


        self.b_gates1 =  self.create_param(b_gates, [4*num_units])
        self.W_x_to_gates1 =  self.create_param(
            W_x_to_gates, [n_units_input_net+add_input_x, 4*num_units])
        self.W_hid_to_gates1 =  self.create_param(
            W_hid_to_gates, [num_units, 4*num_units])


        if self.use_memory_as_output:
            self.W_y =  self.create_param(
                W_y, [self.mem_size, n_classes])
            self.b_y =  self.create_param(
                b_y, [self.mem_size])
        else:
            self.W_y =  self.create_param(
                W_y, [num_units, n_classes])
            self.b_y =  self.create_param(
                b_y, [n_classes])

        self.b_gates2 = []
        self.W_hid_to_gates2 = []
        self.W_x_to_gates2 = [],
        self.W_cell_to_gates2 = []
        if self.two_layer:
            self.b_gates2 =  self.create_param(b_gates, [4*num_units])
            self.W_x_to_gates2 =  self.create_param(W_x_to_gates, [self.num_units, 4*num_units])
            self.W_hid_to_gates2 =  self.create_param(
            W_hid_to_gates, [num_units, 4*num_units])
            if self.peepholes:
                self.W_cell_to_gates2 =  self.create_param(
                    W_cell_to_gates, [3*num_units])
                self.W_cell_to_gates2.name = "DRAMLayer: W_cell_to_gates2"
            else:
                self.W_cell_to_gates2 = []
            self.b_gates2.name = "DRAMLayer(two_layer): b_gates2"
            self.W_x_to_gates2.name = "DRAMLayer(two_layer): W_x_to_gates2"
            self.W_hid_to_gates2.name = "DRAMLayer(two_layer): W_hid_to_gates2"

        # Setup initial values for the cell and the lstm hidden units
        if self.learn_hid_init:
            self.cell1_init = self.create_param(cell_init, (1, num_units))
            self.hid1_init = self.create_param(hid_init, (1, num_units))
            self.cell2_init = self.create_param(cell_init, (1, num_units))
            self.hid2_init = self.create_param(hid_init, (1, num_units))

        else:
            self.cell1_init = T.zeros((1, num_units))
            self.hid1_init = T.zeros((1, num_units))
            self.cell2_init = T.zeros((1, num_units))
            self.hid2_init = T.zeros((1, num_units))


        # variational weights
        # TODO: Make the sizes more flexible, they are not required to be equal

        self.b_gates1.name = "DRAMLayer(core1): b_gates1"
        self.W_x_to_gates1.name = "DRAMLayer(core1): W_x_to_gates1"
        self.W_hid_to_gates1.name = "DRAMLayer(core1): W_hid_to_gates1"
        self.W_y.name = "DRAMLayer: W_y"
        self.b_y.name = "DRAMLayer: b_y"
        self.cell1_init.name = "DRAMLayer(core1): cell1_init"
        self.hid1_init.name = "DRAMLayer(core1): hid1_init"
        if self.two_layer:
            self.cell2_init.name = "DRAMLayer(two_layer): cell2_init"
            self.hid2_init.name = "DRAMLayer(two_layer): hid2_init"

        # INIT READ AND WRITE BAISES
        # We assume that the innitial hiden output from the LSTMs are zero or
        # near zero. Then this init will
        #
        delta_read = 0.8  #
        gamma = 1.0
        sigma_read = 1.0
        center_y = 0.0
        center_x = 0.0
        if read_init is None:
            read_init = np.array([[center_y,
                               center_x,
                               np.log(delta_read),
                               np.log(sigma_read),
                               np.log(gamma)]])
            read_init = read_init.astype(theano.config.floatX)
        print("Read init is", read_init)

        self.W_read = self.create_param(W_read, (num_units, 5))
        self.b_read = self.create_param(b_read, (5))
        self.read_init = self.create_param(read_init, (1,5))
        self.W_read.name = "DRAMLayer: W_read"
        self.b_read.name = "DRAMLayer: b_read"

        #input network
        if self.use_loc_as_input:
            self.W_loc_to_lstm1 = self.create_param(
            W_x_to_gates, (2, 128))
            self.b_loc_to_lstm1 = self.create_param(b_gates, (128))
            self.W_loc_to_lstm1.name = "DRAMLayer(use_loc_as_input): W_loc_to_lstm1"
            self.b_loc_to_lstm1.name = "DRAMLayer(use_loc_as_input): b_loc_to_lstm1"

            self.W_loc_to_lstm2 = self.create_param(
            W_x_to_gates, (128, self.n_units_input_net))
            self.b_loc_to_lstm2 = self.create_param(b_gates, (self.n_units_input_net))
            self.W_loc_to_lstm2.name = "DRAMLayer(use_loc_as_input): W_loc_to_lstm2"
            self.b_loc_to_lstm2.name = "DRAMLayer(use_loc_as_input): b_loc_to_lstm2"
        else:
            self.b_loc_to_lstm1 = []
            self.W_loc_to_lstm1 = []
            self.b_loc_to_lstm2 = []
            self.W_loc_to_lstm2 = []



        self.W_patch_to_lstm1 = self.create_param(
            W_x_to_gates, (N_filters_read*N_filters_read, 128))
        self.b_patch_to_lstm1 = self.create_param(b_gates, (128))
        self.W_patch_to_lstm1.name = "DRAMLayer: W_patch_to_lstm1"
        self.b_patch_to_lstm1.name = "DRAMLayer: b_patch_to_lstm1"

        self.W_patch_to_lstm2 = self.create_param(
        W_x_to_gates, (128, self.n_units_input_net))
        self.b_patch_to_lstm2 = self.create_param(b_gates, (self.n_units_input_net))
        self.W_patch_to_lstm2.name = "DRAMLayer: W_patch_to_lstm2"
        self.b_patch_to_lstm2.name = "DRAMLayer: b_patch_to_lstm2"

        if self.use_memory_as_output:
            print("Using memory")
            self.W_mem_mem = self.create_param(
            W_x_to_gates, (self.num_units, self.mem_size))
            self.b_mem_mem = self.create_param(b_gates, (self.mem_size))
            self.W_mem_pos = self.create_param(
            W_x_to_gates, (self.num_units+self.mem_size, self.mem_size))
            self.b_mem_pos = self.create_param(b_gates, (self.mem_size))
            self.W_mem_pos.name = "DRAMLayer: W_mem_pos"
            self.b_mem_pos.name = "DRAMLayer: b_mem_pos"
            self.W_mem_mem.name = "DRAMLayer: W_mem_mem"
            self.b_mem_mem.name = "DRAMLayer: b_mem_mem"
        else:
            self.b_mem_pos = []
            self.W_mem_pos = []
            self.b_mem_mem = []
            self.W_mem_mem = []

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

        if self.two_layer:
            params += [self.b_gates2, self.W_x_to_gates2, self.W_hid_to_gates2,
                       self.W_cell_to_gates2]

        params += [self.W_patch_to_lstm1, self.b_patch_to_lstm1, self.W_patch_to_lstm2, self.b_patch_to_lstm2]

        if self.use_loc_as_input:
            params += [self.W_loc_to_lstm1, self.b_loc_to_lstm1, self.W_loc_to_lstm2, self.b_loc_to_lstm2]

        if self.use_memory_as_output:
            params += [self.W_mem_pos, self.b_mem_pos, self.W_mem_mem, self.b_mem_mem]
        return params

    def get_weight_params(self):
        '''
        Get all weights of this layer
        :returns:
            - weight_params : list of theano.shared
                List of all weight parameters
        '''
        return [self.W_x_to_gates1, self.W_hid_to_gates1, self.W_read, self.W_y]

    def get_peephole_params(self):
        '''
        Get all peephole parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all peephole parameters
        '''
        return [self.W_cell_to_gates1]

    def get_init_params(self):
        '''
        Get all initital parameters of this layer.
        :returns:
            - init_params : list of theano.shared
                List of all initial parameters
        '''
        if self.learn_hid_init:
            params =  [self.hid1_init, self.cell1_init]
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
        params =  [self.b_gates1, self.b_read, self.b_y]

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
        if self.return_sequence:
            return (input_shape[0], input_shape[1], self.n_classes)
        else:
            return (input_shape[0], self.n_classes)

    def get_output_for(self, x, *args, **kwargs):
        '''
        Compute this layer's output function given a symbolic input variable

        :parameters:
            - input : theano.TensorType
                Symbolic input variable

        :returns:
            - layer_output : theano.TensorType
                Symbolic output variable
        '''


        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        def slice_c(x, n):
            return x[n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_dot_W_n is the n'th row of the input dot W multiplication
        # The step function calculates the following:
        #
        # i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + W_{ci}c_{t-1} + b_i)
        # f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + W_{cf}c_{t-1} + b_f)
        # c_t = f_tc_{t - 1} + i_t\tanh(W_{xc}x_t + W_{hc}h_{t-1} + b_c)
        # o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + W_{co}c_t + b_o)
        # h_t = o_t \tanh(c_t)
        #
        # Gate names are taken from http://arxiv.org/abs/1409.2329 figure 1
        def lstm(gates, cell_previous,W_cell_to_gates,nonlinearity_out):
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            modulationgate = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                ingate += cell_previous*slice_c(W_cell_to_gates, 0)
                forgetgate += cell_previous*slice_c(W_cell_to_gates, 1)

            if self.grad_clip_vals_in is not None:
                ingate = theano.gradient.grad_clip(ingate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                forgetgate = theano.gradient.grad_clip(forgetgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                modulationgate = theano.gradient.grad_clip(modulationgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            modulationgate = self.nonlinearity_modulationgate(modulationgate)
            if self.grad_clip_vals_in is not None:
                ingate = theano.gradient.grad_clip(ingate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                forgetgate = theano.gradient.grad_clip(forgetgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])
                modulationgate = theano.gradient.grad_clip(modulationgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            cell = forgetgate*cell_previous + ingate*modulationgate
            if self.peepholes:
                outgate += cell*slice_c(W_cell_to_gates, 2)

            if self.grad_clip_vals_in is not None:
                outgate = theano.gradient.grad_clip(outgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            outgate = self.nonlinearity_outgate(outgate)
            if self.grad_clip_vals_in is not None:
                outgate = theano.gradient.grad_clip(outgate, self.grad_clip_vals_in[0], self.grad_clip_vals_in[1])

            hid = outgate*nonlinearity_out(cell)
            return [cell, hid]


        def step(cell1_previous, hid1_previous,
                 cell2_previous, hid2_previous,
                 l_read_previous, preds_previous, mem_previous,
            x, W_x_to_gates, W_hid_to_gates, W_read, W_y,
            W_cell_to_gates, b_gates, b_read, b_y,
            W_x_to_gates2, W_hid_to_gates2, W_cell_to_gates2, b_gates2,
            W_patch_to_lstm1, b_patch_to_lstm1,
            W_patch_to_lstm2, b_patch_to_lstm2,
            W_loc_to_lstm1, b_loc_to_lstm1,
            W_loc_to_lstm2, b_loc_to_lstm2,
                 W_mem_pos, b_mem_pos, W_mem_mem, b_mem_mem):

            # calculate gates pre-activations and slice
            att_read = nn2att(l_read_previous, self.N_filters_read,  self.imgshp)
            x_patch = read_single(x, att_read, self.N_filters_read, self.imgshp)
            x_patch = att_read['gamma']*x_patch

            ##input network
            # see eq 1 from http://arxiv.org/pdf/1412.7755v1.pdf
            g_image1 = nonlinearities.rectify(T.dot(x_patch, W_patch_to_lstm1) + b_patch_to_lstm1)
            g_image2 = T.dot(g_image1, W_patch_to_lstm2) + b_patch_to_lstm2
            if self.use_loc_as_input:
                print("STEP: using loc as input")
                # see http://arxiv.org/pdf/1406.6247v1.pdf
                ll_xy = l_read_previous[:,T.arange(2)]
                g_loc1 = nonlinearities.rectify(T.dot(ll_xy, W_loc_to_lstm1) + b_loc_to_lstm1)
                g_loc2 = T.dot(g_loc1, W_loc_to_lstm2) + b_loc_to_lstm2
                g_in = nonlinearities.rectify(g_loc2 + g_image2)  # eq
            else:
                print("STEP: Not using loc as input")
                g_in = nonlinearities.rectify(g_image2)

            # equation (5)~ish
            #slice_gates_idx = 4*self.num_units_encoder_and_decoder
            # ENCODER
            gates = T.dot(g_in, W_x_to_gates) + b_gates
            gates += T.dot(hid1_previous, W_hid_to_gates)
            cell1, hid1 = lstm(gates, cell1_previous, W_cell_to_gates,
                             self.nonlinearity_out)


            ### use memeory or not
            # todo: move preds softmax outside of scan
            if self.use_memory_as_output:
                print("STEP: Memory output")
                mem_in = T.concatenate([hid1, mem_previous], axis=1)
                mem_pos = T.nnet.softmax(T.dot(mem_in, W_mem_pos) + b_mem_pos)
                mem_vec = nonlinearities.tanh(T.dot(hid1, W_mem_mem) + b_mem_mem)
                mem = mem_previous + mem_pos*mem_vec
                preds = T.nnet.softmax(T.dot(mem, W_y) + b_y)
            else:
                print("STEP: Output from layer1")
                preds = T.nnet.softmax(T.dot(hid1, W_y) + b_y)
                mem = mem_previous

            if self.two_layer:
                # make location output from second layer
                print('STEP: TWO layer')
                gates2 = T.dot(hid1, W_x_to_gates2) + b_gates2
                gates2 += T.dot(hid2_previous, W_hid_to_gates2)
                cell2, hid2 = lstm(gates2, cell2_previous, W_cell_to_gates2,
                             self.nonlinearity_out)
                l_read = T.dot(hid2, W_read) + b_read
            else:
                # make location output from first layer
                print('STEP: ONE layer')
                l_read = T.dot(hid1, W_read) + b_read
                hid2 = hid2_previous   # dummy
                cell2 = cell2_previous # dummy

            # clip gradients
            if self.grad_clip_vals_out is not None:
                cell1 = theano.gradient.grad_clip(cell1, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                hid1 = theano.gradient.grad_clip(hid1, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                cell2 = theano.gradient.grad_clip(cell2, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
                hid2 = theano.gradient.grad_clip(hid2, self.grad_clip_vals_out[0], self.grad_clip_vals_out[1])
            return [cell1, hid1, cell2, hid2, l_read, preds, mem]


        non_seqs = [x, self.W_x_to_gates1, self.W_hid_to_gates1, self.W_read,
                   self.W_y, self.W_cell_to_gates1, self.b_gates1,
                   self.b_read, self.b_y,
                   self.W_x_to_gates2, self.W_hid_to_gates2, self.W_cell_to_gates2, self.b_gates2,
                   self.W_patch_to_lstm1, self.b_patch_to_lstm1,
                   self.W_patch_to_lstm2, self.b_patch_to_lstm2,
                   self.W_loc_to_lstm1, self.b_loc_to_lstm1,
                   self.W_loc_to_lstm2, self.b_loc_to_lstm2,
                   self.W_mem_pos, self.b_mem_pos,
                   self.W_mem_mem, self.b_mem_mem]
        ones = T.ones((self.num_batch,1))
        init = [T.dot(ones, self.cell1_init),
                T.dot(ones, self.hid1_init),
                T.dot(ones, self.cell2_init),
                T.dot(ones, self.hid2_init),
                T.dot(ones, self.read_init),
                T.zeros((self.num_batch, self.n_classes)),
                T.zeros((self.num_batch, self.mem_size))]

        output_scan = theano.scan(step,
                             outputs_info=init,
                             non_sequences=non_seqs,
                             n_steps=self.glimpses,
                             go_backwards=False)[0]


        cell1, hid1, cell2, hid2, l_read, preds, mem = output_scan
        self.att_vals_read = l_read
        if self.return_sequence:
            preds = preds.dimshuffle(1,0,2)
        else:
            preds = preds[-1]


        return preds

    def get_att_vals(self):
        return self.att_vals_read.dimshuffle(1,0,2)
