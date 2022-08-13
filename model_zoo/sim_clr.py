import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D
from tensorflow.keras import backend as K
#####################################################################################################################

class SimCLR(tf.keras.Model):

    def __init__(self, base_model, input_shape, order=0):
        '''
        THIS IS THE CONSTRUCTOR OF SIM-CLR CLASS.
        :param base_model: backbone UNET model; it may not workwith some of the base models, model type 0, 5, and 6 are valid choices
        :param input_shape: (width,height,channel) shape of the input images
        :param order: it determines where to cut (from which layer order) UNET model for putting CNN + MLP header
        :return: returns nothing
        '''
        input_a = Input(shape=input_shape)
        input_b = Input(shape=input_shape)

        # CUT UNET MODEL FROM A CERTAIN POINT
        if base_model.name=='attunet_model':
            layer_name='attunet_up{}_conv_after_concat_4_activation'.format(order) #'attunet_up0_conv_after_concat_0_bn'
            neurons = base_model.get_layer(layer_name).output_shape[1:][2]
            header_shape=base_model.get_layer(layer_name).output_shape[1:]
            header_input=base_model.get_layer(layer_name).output

        elif base_model.name=='mobile_unet':
            order=order+2
            layer_name = 'concatenate_{}'.format(order)
            neurons = base_model.get_layer(layer_name).output_shape[1:][2]
            header_shape=base_model.get_layer(layer_name).output_shape[1:]
            header_input = base_model.get_layer(layer_name).output

        elif base_model.name=='capsule_network':
            layer_name = 'conv_cap_4_1'
            neurons = base_model.get_layer(layer_name).output_shape[1:][2]
            header_input=tf.reduce_mean(base_model.get_layer(layer_name).output,axis=-1)
            header_shape = header_input.shape[1:]

        elif base_model.name=='adaptive_attention_butterfly':
            layer_name = 'aspp2'
            neurons = base_model.get_layer(layer_name).output_shape[1:][2]
            header_shape = base_model.get_layer(layer_name).output_shape[1:]
            header_input = base_model.get_layer(layer_name).output
        else: #Unet 3+
            layer_name='unet3plus_down4_conv_1_activation'
            header_shape = base_model.get_layer(layer_name).output_shape[1:]
            header_input = base_model.get_layer(layer_name).output



        for layer in base_model.layers: layer.trainable = False

        for ind, layer in enumerate(base_model.layers):
            base_model.layers[ind].trainable=True
            if layer.name == layer_name:
                break


        model_h = Sequential([
            Input(shape=header_shape),
            Flatten(),
            Dense(2048, activation='swish'),
            Dropout(0.1),
            Dense(1024, activation='swish'),
            Dropout(0.1),
            Lambda(lambda x: K.l2_normalize(x, axis=-1))

         ])
        # CNN + MLP HEADER DEFINITION
        #model_h = Sequential([
        #    Input(shape=header_shape),
        #    MaxPooling2D(pool_size=(2, 2)),
        #    Conv2D(neurons, 3, activation='swish', padding='same', kernel_initializer='he_normal'),
        #    MaxPooling2D(pool_size=(2, 2)),
        #    Conv2D(neurons//2, 3, activation='swish', padding='same', kernel_initializer='he_normal'),
        #    MaxPooling2D(pool_size=(2, 2)),
        #    Flatten(),
        #    Dense(2048, activation='swish'),
        #    Dropout(0.5),
        #    Dense(512, activation='swish'),
        #    Lambda(lambda x: K.l2_normalize(x, axis=-1))
        #
        #])



        composed_model = Model(inputs=[base_model.input], outputs=[model_h(header_input)])

        out_a = composed_model(input_a)
        out_b = composed_model(input_b)

        # CONCATENATED OUTPUT FOR SIM-CLR MODEL
        con_out = tf.keras.layers.Concatenate(axis=-1,name='contrastive')([out_a,out_b])

        # BINARY OUTPUT FOR SELF-SUPERVISED MODEL
        out = tf.math.subtract(out_a, out_b, name=None)
        out = Lambda(lambda out: tf.norm(out, ord='euclidean', keepdims=True, axis=-1))(out)
        binary =Dense(1, activation='sigmoid', name='binary')(out)


        super(SimCLR, self).__init__(inputs=[input_a, input_b], outputs=[con_out,binary])
        self.base_model = base_model
