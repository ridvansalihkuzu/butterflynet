import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#############################################################################################################

class VisionTransfomer(tf.keras.Model):

    def __init__(self, input_shape, num_classes, patch_size = 6, projection_dim = 64, transformer_layers = 8, num_heads = 4, mlp_head_units=None, weights=None):
        '''
        THIS IS THE CONSTRUCTOR OF VISION TRANSFORMET CLASS.
        :param input_shape: (width,height,channel) shape of the input images
        :param num_classes: it sets number of classes
        :param patch_size: it sets patch size for splitting the images
        :param projection_dim: it sets projection dimension for self-attention gate outputs
        :param transformer_layers: it sets number of consecutive self-attention layers
        :param num_heads: it sets number of heads for each attention
        :param mlp_head_units: it is a list of numbers to set number of layers and number of elements in each layer for MLP
        :param weights: a file path for pretrained initialization of the model weights
        :return: returns nothing
               '''

        # Define constants
        if mlp_head_units is None:
            mlp_head_units = [2048, 1024]
        inputs = tf.keras.layers.Input(shape=input_shape)

        num_patches = (inputs.shape[-2] // patch_size) ** 2

        transformer_units = [projection_dim * 2, projection_dim]
        patches = Patches(patch_size)(inputs)

        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = VisionTransfomer.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = VisionTransfomer.mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)


        super(VisionTransfomer, self).__init__(inputs=inputs, outputs=logits, name='vision_transformer')

        if weights is not None:
            self.load_weights(filepath=weights)

            print('PRE-TIRAINED MODEL IS LOADED: {}'.format(weights))


    @staticmethod
    def mlp(x, hidden_units, dropout_rate):
        for units in hidden_units:
            x = layers.Dense(units, activation=tf.nn.gelu)(x)
            x = layers.Dropout(dropout_rate)(x)
        return x

    @staticmethod
    def transformer_encoder(input, num_heads, projection_dim, transformer_units):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(input)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, input])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = VisionTransfomer.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        return layers.Add()([x3, x2])

    @staticmethod
    def transformer_decoder(x11, x12, num_heads, projection_dim, transformer_units):
        # Layer normalization 1.
        x11 = layers.LayerNormalization(epsilon=1e-6)(x11)
        x12 = layers.LayerNormalization(epsilon=1e-6)(x12)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x11, x12)
        # Skip connection 1
        x2 = layers.Add()([attention_output, x11])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = VisionTransfomer.mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        return layers.Add()([x3, x2])


########################################################################################################
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

########################################################################################################
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


