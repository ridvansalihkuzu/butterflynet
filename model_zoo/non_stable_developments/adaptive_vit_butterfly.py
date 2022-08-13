from functools import partial
from model_zoo.butterfly_family import adaptive_attention_butterfly_utils
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from model_zoo.vision_transformer import *

class AdaptiveVITButterfly(adaptive_attention_butterfly_utils.AdaptiveNetwork):
    """
    Isensee's 2D and 3D U-Nets for Heart Segmentation from the MSD that follows the conditions:
        * pools until the feature maps axes are all of at <= 8
        * max # of pools = 5 for 3D and max # of pools = 6 for 2D
    Augmented to allow for use as a feature extractor
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
            self.context_mod = partial(adaptive_attention_butterfly_utils.context_module_2D, n_convs = n_convs)
            self.localization_mod = partial(adaptive_attention_butterfly_utils.localization_module_2D, n_convs = n_convs)
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

    def vit_encoder(self,inputs, patch_size=8, projection_dim=64):
        # Define constants
        num_patches = (inputs.shape[-2] // patch_size) ** 2
        num_heads = 16
        transformer_units = [projection_dim * 2, projection_dim]
        patches = Patches(patch_size)(inputs)

        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        level = 0
        skip_connections1 = []
        skip_connections2 = []
        while level < self.max_pools:
            if level == 0:
                skip, pool = self.context_mod(inputs, self.filter_list[level], pool_size=self.pool_list[0])
            elif level > 0:
                skip, pool = self.context_mod(pool, self.filter_list[level], pool_size=self.pool_list[level])
            encoded_patches = VisionTransfomer.transformer_encoder(encoded_patches, num_heads, projection_dim, transformer_units)

            skip_connections1.append(skip)
            skip_connections2.append(encoded_patches)

            shape=[pool.shape[1], pool.shape[2], encoded_patches.shape[1]*encoded_patches.shape[2]//(pool.shape[1]*pool.shape[2])]
            reshaped=tf.keras.layers.Reshape(shape)(encoded_patches)
            pool=Concatenate()([pool, reshaped])
            level += 1

        pool = self.context_mod(pool, self.filter_list[level], pool_size=None)
        encoded_patches = VisionTransfomer.transformer_encoder(encoded_patches, num_heads, projection_dim, transformer_units)
        #encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #bottom = Concatenate()([pool, reshaped])

        return pool, encoded_patches, skip_connections1, skip_connections2, level

    def vit_decoder1(self, inputs, decoded_patches, skip_1, skip_2, level, projection_dim=64):

        transformer_units = [projection_dim * 2, projection_dim]
        num_heads = 16

        shape = [inputs.shape[1], inputs.shape[2], decoded_patches.shape[1] * decoded_patches.shape[2] // (inputs.shape[1] * inputs.shape[2])]
        reshaped = tf.keras.layers.Reshape(shape)(decoded_patches)
        inputs = Concatenate()([inputs, reshaped])

        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(inputs, skip_1[current_depth], self.filter_list[current_depth], upsampling_size = self.pool_list[current_depth],is_attention1=False, is_attention2=False)
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_1[current_depth], self.filter_list[current_depth], upsampling_size = self.pool_list[current_depth],is_attention1=False,is_attention2=False)

            decoded_patches = VisionTransfomer.transformer_decoder(decoded_patches, skip_2[current_depth], num_heads, projection_dim, transformer_units)

            shape = [upsamp.shape[1], upsamp.shape[2],decoded_patches.shape[1] * decoded_patches.shape[2] // (upsamp.shape[1] * upsamp.shape[2])]
            reshaped = tf.keras.layers.Reshape(shape)(decoded_patches)
            upsamp = Concatenate()([upsamp, reshaped])


            level -= 1

        return upsamp

    def vit_decoder2(self,inputs, decoded_patches, skip_1, skip_21, skip_22,level, projection_dim=64):

        transformer_units = [projection_dim * 2, projection_dim]
        num_heads = 16

        shape = [inputs.shape[1], inputs.shape[2],
                 decoded_patches.shape[1] * decoded_patches.shape[2] // (inputs.shape[1] * inputs.shape[2])]
        reshaped = tf.keras.layers.Reshape(shape)(decoded_patches)
        inputs = Concatenate()([inputs, reshaped])

        while level > 0:  # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(inputs, skip_1[current_depth], self.filter_list[current_depth],
                                               upsampling_size=self.pool_list[current_depth], second_skip_layer=skip_21[current_depth], is_attention1=False,
                                               is_attention2=True)
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_1[current_depth], self.filter_list[current_depth],
                                               upsampling_size=self.pool_list[current_depth], second_skip_layer=skip_21[current_depth], is_attention1=False,
                                               is_attention2=True)

            decoded_patches = VisionTransfomer.transformer_decoder(decoded_patches, skip_22[current_depth], num_heads, projection_dim,
                                                  transformer_units)

            shape = [upsamp.shape[1], upsamp.shape[2],
                     decoded_patches.shape[1] * decoded_patches.shape[2] // (upsamp.shape[1] * upsamp.shape[2])]
            reshaped = tf.keras.layers.Reshape(shape)(decoded_patches)
            upsamp = Concatenate()([upsamp, reshaped])

            level -= 1

        return upsamp



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

    def encoder11(self, inputs):
        level = 0
        skip_connections = []

        model = EfficientNetB7(include_top=False, weights=None, input_tensor=inputs)
        names = ['block2a_expand_activation', 'block3a_expand_activation', 'block4a_expand_activation', 'block6a_expand_activation']
        for name in names:
            skip_connections.append(model.get_layer(name).output)
            level += 1

        output = model.get_layer("top_activation").output
        return output, skip_connections, level

    def encoder12(self, inputs):
        level = 0
        skip_connections = []

        model = VGG19(include_top=False, weights=None, input_tensor=inputs)
        names = ["block1_conv2", "block2_conv2", "block3_conv4", "block4_conv4","block5_conv4"]
        for name in names:
            skip_connections.append(model.get_layer(name).output)
            level += 1

        output = model.get_layer("block5_pool").output
        return output, skip_connections, level

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


    def decoder3(self, inputs, skip_1, skip_2, level):

        while level > 0: # (** level = depth - 1 at the start of the loop)
            current_depth = level - 1
            if level == self.max_pools:
                upsamp = self.localization_mod(inputs, skip_1[current_depth], self.filter_list[current_depth],upsampling_size = self.pool_list[current_depth],second_skip_layer=skip_2[current_depth],is_attention2=False)
            elif not level == self.max_pools:
                upsamp = self.localization_mod(upsamp, skip_1[current_depth], self.filter_list[current_depth],upsampling_size = self.pool_list[current_depth],second_skip_layer=skip_2[current_depth],is_attention2=False)
            level -= 1

        return upsamp

    def build_model(self, fly=False):
        """
        Returns a keras.models.Model instance.
        Args:
            input_layer: keras layer
                * if None, then defaults to a regular input layer based on the shape
            extractor: boolean on whether or not to use the U-Net as a feature extractor or not
        """
        inputs = Input(shape = self.input_shape)

        x, encoded_patches, skip_11, skip_12, level_1 = self.vit_encoder(inputs)
        x = self.ASPP(x.shape[1:], 64)(x)

        x = self.vit_decoder1(x, encoded_patches, skip_11, skip_12, level_1) #TODO

        outputs1 = self._build_predictor(x)

        if not fly:
            return Model(inputs, outputs1)
        else:

            x = inputs * outputs1

            x, skip_2, level_2 = self.encoder2(x)
            aspp_model2 = self.ASPP(x.shape[1:], 64)
            x = aspp_model2(x)
            x = self.decoder3(x, skip_11, skip_2,level_2) # TODO: replace with decoder 2 for comparison
            outputs2 = self._build_predictor(x)

            #x = Concatenate()([outputs1, outputs2])
            #outputs = self._build_predictor(x)

            return Model(inputs, outputs2)

    def build_best_model(self):
        """
        Returns a keras.models.Model instance.
        Args:
            input_layer: keras layer
                * if None, then defaults to a regular input layer based on the shape
            extractor: boolean on whether or not to use the U-Net as a feature extractor or not
        """
        inputs = Input(shape = self.input_shape)

        x, skip_1, level_1 = self.encoder1(inputs)
        aspp_model1 = self.ASPP(x.shape[1:],64)
        x = aspp_model1(x)
        x = self.decoder1(x,skip_1,level_1)
        outputs1 = self._build_predictor(x)

        x = inputs * outputs1

        x, skip_2, level_2 = self.encoder2(x)
        aspp_model2 = self.ASPP(x.shape[1:], 64)
        x = aspp_model2(x)
        x = self.decoder2(x, skip_1, skip_2,level_2)
        outputs2 = self._build_predictor(x)

        #x = Concatenate()([outputs1, outputs2])
        #outputs = self._build_predictor(x)

        model = Model(inputs, outputs2)

        return model

    def build_model3(self):
            """
            Returns a keras.models.Model instance.
            Args:
                input_layer: keras layer
                    * if None, then defaults to a regular input layer based on the shape
                extractor: boolean on whether or not to use the U-Net as a feature extractor or not
            """
            inputs = Input(shape=self.input_shape)

            x, skip_1, level_1 = self.encoder1(inputs)
            aspp_model1 = self.ASPP(x.shape[1:], 64)
            x = aspp_model1(x)
            x = self.decoder1(x, skip_1, level_1)
            outputs1 = self._build_predictor(x)

            x = inputs * outputs1

            x, encoded_patches, skip_21, skip_22, level_1  = self.vit_encoder(x)
            aspp_model2 = self.ASPP(x.shape[1:], 64)
            x = aspp_model2(x)
            x = self.vit_decoder2(x, encoded_patches, skip_1, skip_21, skip_22,level_1)
            outputs2 = self._build_predictor(x)

            # x = Concatenate()([outputs1, outputs2])
            # outputs = self._build_predictor(x)

            model = Model(inputs, outputs2)

            return model

    @staticmethod
    def ASPP(shape, filter):

        x = Input(shape)

        y1 = AveragePooling2D(pool_size=(shape[0], shape[1]))(x)
        y1 = Conv2D(filter, 1, padding="same")(y1)
        y1 = BatchNormalization()(y1)
        y1 = Activation("relu")(y1)
        y1 = UpSampling2D((shape[0], shape[1]), interpolation='bilinear')(y1)

        y2 = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(x)
        y2 = BatchNormalization()(y2)
        y2 = Activation("relu")(y2)

        y3 = Conv2D(filter, 3, dilation_rate=6, padding="same", use_bias=False)(x)
        y3 = BatchNormalization()(y3)
        y3 = Activation("relu")(y3)

        y4 = Conv2D(filter, 3, dilation_rate=12, padding="same", use_bias=False)(x)
        y4 = BatchNormalization()(y4)
        y4 = Activation("relu")(y4)

        y5 = Conv2D(filter, 3, dilation_rate=18, padding="same", use_bias=False)(x)
        y5 = BatchNormalization()(y5)
        y5 = Activation("relu")(y5)

        y = Concatenate()([y1, y2, y3, y4, y5])

        y = Conv2D(filter, 1, dilation_rate=1, padding="same", use_bias=False)(y)
        y = BatchNormalization()(y)
        y = Activation("relu")(y)

        return Model(x,y)


