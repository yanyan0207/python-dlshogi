from pydlshogi_v2.models.bias_layer import BiasLayer
from tensorflow.keras.layers import Input, Conv1D, Conv2D, Permute, Flatten, Reshape, BatchNormalization, Activation
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
    # 盤上のコマ
    ch = 192
    inputs_board = Input(shape=(9, 9, 28), name="digits")
    x_board = Conv2D(ch, kernel_size=(3, 3),
                     activation='relu', padding='same')(inputs_board)

    # 駒台
    inputs_hands = Input(shape=(76,))
    x_hands = Reshape((-1, 1, 76))(inputs_hands)
    x_hands = Conv2D(ch, kernel_size=(1, 1),
                     activation='relu', padding='same')(x_hands)

    # 盤上と駒台を加算
    x = layers.add([x_board, x_hands])

    for i in range(blocks):
        x = residualBlock(x)
    x = Conv2D(27, kernel_size=(1, 1),
               activation='relu', use_bias=False)(x)
    x = Flatten()(x)
    x = BiasLayer()(x)
    outputs = x
    model = Model(inputs=[inputs_board, inputs_hands], outputs=outputs)

    return model
