from functools import partial
from model_zoo.adaptive_net_family.utils import localization_module_2D,context_module_2D
from model_zoo.adaptive_net_family.adaptive_network import AdaptiveNetwork
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


class AdaptiveAttentionButterfly(tf.keras.Model):

    def __init__(self, input_shape, n_classes=1, max_pools=6, starting_filters=16, weights=None):
        '''
        THIS IS THE CONSTRUCTOR OF Adaptive Attention Butterfly UNET CLASS.
        :param input_shape: (width,height,channel) shape of the input images
        :param n_classes: number of output classes
        :param max_pools: maximum number of pools
        :param starting_filters: number of filters to start extending the network
        :param weights: a file path for pretrained initialization of the model weights
        :return: returns nothing
               '''

        inputs, outputs =AdaptiveButterFlyBack(2, input_shape, n_classes=n_classes, max_pools=max_pools, starting_filters=starting_filters).build_model()

        super(AdaptiveAttentionButterfly, self).__init__(inputs=inputs, outputs=outputs, name='adaptive_attention_butterfly')

        if weights is not None:
            self.load_weights(filepath=weights)

            print('PRE-TIRAINED MODEL IS LOADED: {}'.format(weights))


########################################################################################################


class AdaptiveButterFlyBack(AdaptiveNetwork):
    """
    Attributes:
        n_convs: number of convolutions per module
        input_shape: The shape of the input including the number of input channels; (z, x, y, n_channels)
        n_classes: number of output classes (default: 1, which is binary segmentation)
            * Make sure that it doesn't include the background class (0)
        max_pools:
        starting_filters:
    """
    def __init__(self, n_convs, input_shape, n_classes = 1, max_pools = 6, starting_filters = 30):
        super().__init__(input_shape, max_pools, starting_filters, base_pool_size = 2)
        self.n_convs = n_convs
        self.n_classes = n_classes
        if self.ndim == 2:
            self.context_mod = partial(context_module_2D, n_convs = n_convs)
            self.localization_mod = partial(localization_module_2D, n_convs = n_convs)
        # automatically reassigns the max number of pools in a model (for cases where the actual max pools < inputted one)
        self.max_pools = max(self._pool_statistics())

    def _build_predictor(self, input_layer):
        """
        Takes a keras layer as input and returns the subsequent classifer Convolution with kernel_size = 1
        * Activations: sigmoid for binary and softmax for multiclass
        Args:
            input_layer:
        Returns:
            conv_seg: output tensor segmentation
        """
        # sigmoid for binary and softmax for multiclass
        kernel_size = tuple([1 for dim in range(self.ndim)])
        if self.ndim == 2:
            if self.n_classes == 1:
                conv_seg = Conv2D(self.n_classes, kernel_size = kernel_size, activation = 'sigmoid')(input_layer)
            elif self.n_classes > 1:
                conv_seg = Conv2D(self.n_classes, kernel_size = kernel_size, activation = 'softmax')(input_layer)
        return conv_seg



    def encoder1(self, inputs):
        level = 0
        skip_connections = []
        while level < self.max_pools:
            if level == 0:
                skip, pool = self.context_mod(inputs, self.filter_list[level], pool_size=self.pool_list[0])
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size=self.pool_list[level])
            skip_connections.append(skip)
            level += 1
        convs_bottom = self.context_mod(pool, self.filter_list[level],pool_size=None)  # No downsampling;  level at (depth) after the loop

        return convs_bottom, skip_connections,level

    def encoder2(self, inputs):
        level = 0
        skip_connections = []
        while level < self.max_pools:
            if level == 0:
                skip, pool = self.context_mod(inputs, self.filter_list[level], pool_size=self.pool_list[0])
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size=self.pool_list[level])
            skip_connections.append(skip)
            level += 1
        convs_bottom = self.context_mod(pool, self.filter_list[level],pool_size=None)  # No downsampling;  level at (depth) after the loop

        return convs_bottom, skip_connections,level


    def decoder1(self, inputs, skip_connections, level):

        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(inputs, skip_connections[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])

            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_connections[current_depth], self.filter_list[current_depth],\
                                               upsampling_size = self.pool_list[current_depth])
            level -= 1

        return upsamp


    def decoder2(self, inputs, skip_1, skip_2, level):

        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(inputs, skip_1[current_depth], self.filter_list[current_depth],upsampling_size = self.pool_list[current_depth],second_skip_layer=skip_2[current_depth])
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_1[current_depth], self.filter_list[current_depth],upsampling_size = self.pool_list[current_depth],second_skip_layer=skip_2[current_depth])
            level -= 1

        return upsamp


    def build_model(self):
        """
        Returns a keras.models.Model instance.
        Args:
            input_layer: keras layer
                * if None, then defaults to a regular input layer based on the shape
            extractor: boolean on whether or not to use the U-Net as a feature extractor or not
        """
        inputs = Input(shape = self.input_shape)

        x, skip_1, level_1 = self.encoder1(inputs)
        x = self.ASPP(x,64,name='aspp1')
        x = self.decoder1(x,skip_1,level_1)
        outputs1 = self._build_predictor(x)

        x = inputs * outputs1

        x, skip_2, level_2 = self.encoder2(x)
        x = self.ASPP(x, 64, name='aspp2')
        x = self.decoder2(x, skip_1, skip_2,level_2)
        outputs2 = self._build_predictor(x)

        return inputs, outputs2

    def ASPP(self,x, filter, name='ASPP'):

        shape=x.shape


        y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
        y1 = Conv2D(filter, 1, padding="same")(y1)
        y1 = PReLU(tf.keras.initializers.Constant(0.3))(y1)
        y1 = BatchNormalization()(y1)
        y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

        y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
        y2 = PReLU(tf.keras.initializers.Constant(0.3))(y2)
        y2 = BatchNormalization()(y2)

        y3 = Conv2D(filter, 3, dilation_rate=3, padding="same", use_bias=False)(x)
        y3 = PReLU(tf.keras.initializers.Constant(0.3))(y3)
        y3 = BatchNormalization()(y3)

        y4 = Conv2D(filter, 3, dilation_rate=5, padding="same", use_bias=False)(x)
        y4 = PReLU(tf.keras.initializers.Constant(0.3))(y4)
        y4 = BatchNormalization()(y4)

        y5 = Conv2D(filter, 3, dilation_rate=7, padding="same", use_bias=False)(x)
        y5 = PReLU(tf.keras.initializers.Constant(0.3))(y5)
        y5 = BatchNormalization()(y5)

        y = Concatenate()([y1, y2, y3, y4, y5])

        y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
        y = PReLU(tf.keras.initializers.Constant(0.3))(y)
        y = BatchNormalization(name=name)(y)

        return y


