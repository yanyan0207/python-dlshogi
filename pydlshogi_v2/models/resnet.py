from pydlshogi_v2.models.bias_layer import BiasLayer
from tensorflow.keras.layers import Input, Conv2D, Permute, Flatten, BatchNormalization, Activation
from tensorflow.keras import layers
from tensorflow.keras.models import Model


def residualBlock(input):
    ch = 192
    x = Conv2D(ch, kernel_size=(3, 3), use_bias=False, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(ch, kernel_size=(3, 3), use_bias=False, padding='same')(x)
    x = BatchNormalization()(x)
    x = layers.add([input, x])
    x = Activation('relu')(x)
    return x


def createModel(blocks=5) -> Model:
    ch = 192
    inputs = Input(shape=(9, 9, 104), name="digits")
    x = Conv2D(ch, kernel_size=(3, 3),
               activation='relu', padding='same')(inputs)
    for i in range(blocks):
        x = residualBlock(x)
    x = Conv2D(27, kernel_size=(1, 1),
               activation='relu', use_bias=False)(x)
    x = Permute((3, 1, 2))(x)
    x = Flatten()(x)
    x = BiasLayer()(x)
    outputs = x
    model = Model(inputs=inputs, outputs=outputs)

    return model
