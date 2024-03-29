import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

#############################################################################################################

class MobileUnet(tf.keras.Model):

    def __init__(self, input_shape, freeze_backbone=False, weights=None):
        '''
        THIS IS THE CONSTRUCTOR OF MOBILE UNET CLASS.
        :param input_shape: (width,height,channel) shape of the input images
        :param freeze_backbone: it determines if the backbone model will be frozen or not
        :param weights: a file path for pretrained initialization of the model weights
        :return: returns nothing
               '''
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
        down_stack.trainable = not freeze_backbone

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=input_shape)
        x = inputs

        # Downsampling through the model
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')  # 64x64 -> 128x128 ,
        x = last(x)

        super(MobileUnet, self).__init__(inputs=inputs, outputs=x, name='mobile_unet')

        if weights is not None:
            self.load_weights(filepath=weights)

            print('PRE-TIRAINED MODEL IS LOADED: {}'.format(weights))


########################################################################################################