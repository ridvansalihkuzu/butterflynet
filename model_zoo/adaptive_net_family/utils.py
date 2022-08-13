import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Concatenate, Conv2D
from tensorflow.keras import backend as K

def context_module_2D(input_layer, n_filters, pool_size = (2,2), n_convs = 2, activation = 'prelu'):
    """
    [2D]; Channels_last
    Context module (Downsampling compartment of the U-Net): `n_convs` Convs (w/ LeakyReLU and BN) -> MaxPooling
    Args:
        input_layer:
        n_filters:
        pool_size: if None, there will be no pooling (default: (2,2))
        n_convs: Number of convolutions in the module
        activation: the activation name; the only advanced activation supported is 'LeakyReLU' (default: 'LeakyReLU')
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        maxpooled output
    """
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            if activation == 'prelu':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(input_layer)
                act = PReLU(tf.keras.initializers.Constant(0.3))(conv)
            elif not activation == 'prelu':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(input_layer)
            bn = BatchNormalization(axis = -1)(act)
        else:
            if activation == 'prelu':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(bn)
                act = PReLU(tf.keras.initializers.Constant(0.3))(conv)
            elif not activation == 'prelu':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(bn)
            bn = BatchNormalization(axis = -1)(act)
    if pool_size is not None:
        pool = MaxPooling2D(pool_size)(bn)
        return bn, pool
    elif pool_size is None:
        return bn

def localization_module_2D(input_layer, skip_layer, n_filters, upsampling_size = (2,2), n_convs = 2, activation = 'prelu', transposed_conv = False, second_skip_layer=None, is_attention1=True, is_attention2=True):
    """
    [2D]; Channels_last
    Localization module (Downsampling compartment of the U-Net): UpSampling3D -> `n_convs` Convs (w/ LeakyReLU and BN)
    Args:
        input_layer:
        skip_layer: layer with the corresponding skip connection (same depth)
        n_filters:
        upsampling_size:
        n_convs: Number of convolutions in the module
        activation: the activation name; the only advanced activation supported is 'LeakyReLU' (default: 'LeakyReLU')
        transposed_conv: boolean on whether you want transposed convs or UpSampling2D (default: False, which is UpSampling2D)
    Returns:
        keras layer after double convs w/ LeakyReLU and BN in-between
        upsampled output
    """
    if transposed_conv:
        upsamp = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding = 'same')(input_layer)
    elif not transposed_conv:
        upsamp = UpSampling2D(upsampling_size, interpolation='bilinear')(input_layer)
    # Tensor concatenation
    if is_attention1:
        skip_layer = attention_gate(X=skip_layer, g=upsamp, channel=skip_layer.shape[-1] // 2,attention='add')
    if second_skip_layer==None:
        concat = Concatenate(axis = -1)([upsamp, skip_layer])
    else:
        if is_attention2:
            second_skip_layer = attention_gate(X=second_skip_layer, g=upsamp, channel=second_skip_layer.shape[-1] // 2, attention='multiply')
        concat = Concatenate(axis=-1)([upsamp, skip_layer,second_skip_layer])
    for conv_idx in range(n_convs):
        if conv_idx == 0:
            if activation == 'prelu':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(concat)
                act = PReLU(tf.keras.initializers.Constant(0.3))(conv)
            elif not activation == 'prelu':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(concat)
            bn = BatchNormalization(axis = -1)(act)
        else:
            if activation == 'prelu':
                conv = Conv2D(n_filters, kernel_size = (3,3), padding = 'same')(bn)
                act = PReLU(tf.keras.initializers.Constant(0.3))(conv)
            elif not activation == 'prelu':
                act = Conv2D(n_filters, kernel_size = (3,3), padding = 'same', activation = activation)(bn)
            bn = BatchNormalization(axis = -1)(act)
    return bn

def attention_gate(X, g, channel,attention='add'):
    '''
    Self-attention gate modified from Oktay et al. 2018.

    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')

    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.

    Output
    ----------
        X_att: output tensor.

    '''

    shape_x = K.int_shape(X)
    shape_g = K.int_shape(g)

    attention_func = eval(attention)

    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True)(X)

    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True)(g)

    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g])

    # nonlinear activation
    f = Activation(tf.nn.swish)(query)

    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True)(f)
    # ------------------------------ #

    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid')(psi_f)

    #shape_sigmoid = K.int_shape(coef_att)
    #upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(coef_att)  # 32

    #upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),arguments={'repnum': shape_x[3]})(upsample_psi) #EXPAND

    # multiplicative attention masking
    X_att = multiply([X, coef_att])

    return X_att



