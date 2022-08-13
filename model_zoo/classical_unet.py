import tensorflow as tf
from tensorflow.keras.layers import *
from model_zoo.adaptive_net_family.utils import context_module_2D,localization_module_2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D
from functools import partial

class Unet_A(tf.keras.Model):
    def __init__(self, input_shape, include_top = True, starting_filters = 32, depth = 4, n_convs = 2, activation = 'relu', weights=None):
        '''
        THIS IS THE CONSTRUCTOR OF UNET CLASS
        :param input_shape: (width,height,channel) shape of the input images
        :param freeze_backbone: it determines if the backbone model will be frozen or not, it is redundant in this class
        :param weights: a file path for pretrained initialization of the model weights
        :return: returns nothing
                      '''


        context_mod = partial(context_module_2D, n_convs=n_convs, activation=activation)
        localization_mod = partial(localization_module_2D, n_convs=n_convs, activation=activation, transposed_conv=True)
        filter_list = [starting_filters * (2 ** level) for level in range(0, depth)]
        pool_size = (2, 2)
        max_pools = depth - 1
        input_layer = Input(shape=input_shape, name='x')
        skip_layers = []
        level = 0
        # context pathway (downsampling) [level 0 to (depth - 1)]
        while level < max_pools:
            if level == 0:
                skip, pool = context_mod(input_layer, filter_list[level], pool_size=pool_size)
            elif level > 0:
                skip, pool = context_mod(pool, filter_list[level], pool_size=pool_size)
            skip_layers.append(skip)
            level += 1
        convs_bottom = context_mod(pool, filter_list[level],
                                   pool_size=None)  # No downsampling;  level at (depth) after the loop
        convs_bottom = context_mod(convs_bottom, filter_list[level], pool_size=None)  # happens twice
        # localization pathway (upsampling with concatenation) [level (depth - 1) to level 1]
        while level > 0:  # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == max_pools:
                upsamp = localization_mod(convs_bottom, skip_layers[current_depth], filter_list[current_depth], upsampling_size=pool_size,is_attention1=False, is_attention2=False)
            elif not level == max_pools:
                upsamp = localization_mod(upsamp, skip_layers[current_depth], filter_list[current_depth], upsampling_size=pool_size,is_attention1=False, is_attention2=False)
            level -= 1
        conv_transition = Conv2D(starting_filters, (1, 1), activation=activation)(upsamp)
        # return feature maps
        if not include_top:
            super(Unet_A, self).__init__(inputs=input_layer, outputs=conv_transition, name='classical_unet')

        # return the segmentation
        elif include_top:
            # inferring the number of classes
            n_class = input_shape[-1]
            # setting activation function based on the number of classes
            if n_class > 1:  # multiclass
                conv_seg = Conv2D(n_class, (1, 1), activation='softmax')(conv_transition)
            elif n_class == 1:  # binary
                conv_seg = Conv2D(1, (1, 1), activation='sigmoid')(conv_transition)

            super(Unet_A, self).__init__(inputs=input_layer, outputs=conv_seg, name='classical_unet')

        if weights is not None:
            self.load_weights(filepath=weights)

            print('PRE-TIRAINED MODEL IS LOADED: {}'.format(weights))


class Unet_B(tf.keras.Model):
    def __init__(self, input_shape, freeze_backbone=False, weights=None):
        '''
        THIS IS THE CONSTRUCTOR OF UNET CLASS DEVELOPED BY Darshan Kishorbai Thummar.
        :param input_shape: (width,height,channel) shape of the input images
        :param freeze_backbone: it determines if the backbone model will be frozen or not, it is redundant in this class
        :param weights: a file path for pretrained initialization of the model weights
        :return: returns nothing
                      '''

        inputs = Input(input_shape)
        if   input_shape[0] == 512: nu_layer = 8
        elif input_shape[0] == 256: nu_layer = 4
        elif input_shape[0] == 128: nu_layer = 2
        else: raise NotImplementedError

        neurons = int(input_shape[0] /nu_layer)

        conv1 = Conv2D(neurons, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # conv1 = BatchNormalization()(conv1)
        conv1 = Conv2D(neurons, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        # conv1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        # conv2 = BatchNormalization()(conv2)
        conv2 = Conv2D(neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        # conv2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        # conv3 = BatchNormalization()(conv3)
        conv3 = Conv2D(neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        # conv3 = BatchNormalization()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        # conv4 = BatchNormalization()(conv4)
        conv4 = Conv2D(neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        # conv4 = BatchNormalization()(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(neurons * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        # conv5 = BatchNormalization()(conv5)
        conv5 = Conv2D(neurons * 16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        # conv5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(neurons * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        # up6 = BatchNormalization()(up6)
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        # conv6 = BatchNormalization()(conv6)
        conv6 = Conv2D(neurons * 8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        # conv6 = BatchNormalization()(conv6)

        up7 = Conv2D(neurons * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        # up7 = BatchNormalization()(up7)
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        # conv7 = BatchNormalization()(conv7)
        conv7 = Conv2D(neurons * 4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        # conv7 = BatchNormalization()(conv7)

        up8 = Conv2D(neurons * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        # up8 = BatchNormalization()(up8)

        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        # conv8 = BatchNormalization()(conv8)
        conv8 = Conv2D(neurons * 2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        # conv8 = BatchNormalization()(conv8)


        last = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')
        x = last(conv8)

        super(Unet_B, self).__init__(inputs=inputs, outputs=x, name='classical_unet')

        if weights is not None:
            self.load_weights(filepath=weights)

            print('PRE-TIRAINED MODEL IS LOADED: {}'.format(weights))
