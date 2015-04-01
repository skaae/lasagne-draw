DRAW implmentation [Gregor 2015]
=======
Implementation of the draw network in Lasagne (https://github.com/benanne/Lasagne)



The Read/write operation is based on https://github.com/jbornschein/draw/tree/master/draw


This is work in progress and is poorly documented


Animation of the DRAW network reconstructing images. Modelling p(x).


.. image:: https://raw.githubusercontent.com/skaae/lasagne-draw/master/animaion.gif
    :alt: DRAW animation
    :width: 679
    :height: 781
    :align: center


Animation of reconstructing images modelling p(x,y). The two first rows generate
0's, next two rows 1's etc. The model is not fully converged

.. image:: https://raw.githubusercontent.com/skaae/lasagne-draw/master/animaion_cond.gif
    :alt: DRAW animation
    :width: 679
    :height: 781
    :align: center


Install
=========
First install Lasagne and Theano.


    git clone https://github.com/skaae/lasagne-draw.git

    cd lasagne-draw

    python setup.py develop

    cd ..

    python -c 'import deepmodels'


References
=========


* Gregor, K., Danihelka, I., Graves, A., & Wierstra, D. (2015). DRAW: A Recurrent Neural Network For Image Generation. arXiv Preprint arXiv:1502.04623.
