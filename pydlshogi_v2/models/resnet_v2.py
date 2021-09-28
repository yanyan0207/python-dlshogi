from pydlshogi_v2.models.bias_layer import BiasLayer
from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, Permute, Flatten, Reshape, BatchNormalization, Activation
from tensorflow.keras import layers
from tensorflow.keras.models import Model

ch = 192
fcl = 256


def residualBlock(input):
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
    inputs_board = Input(shape=(9, 9, 28), name="input_board")
    x_board = Conv2D(ch, kernel_size=(3, 3),
                     activation='relu', padding='same')(inputs_board)

    # 駒台
    inputs_hands = Input(shape=(76,), name="input_hands")
    x_hands = Reshape((-1, 1, 76))(inputs_hands)
    x_hands = Conv2D(ch, kernel_size=(1, 1),
                     activation='relu', padding='same')(x_hands)

    # 盤上と駒台を加算
    x = layers.add([x_board, x_hands])

    for i in range(blocks):
        x = residualBlock(x)

    # policy
    x_policy = Conv2D(27, kernel_size=(1, 1),
                      activation='relu', use_bias=False, name="policy_conv2d")(x)
    x_policy = Flatten()(x_policy)
    x_policy = BiasLayer(name="policy_output")(x_policy)
    outputs_policy = x_policy

    # value
    x_value = Conv2D(27, kernel_size=(1, 1),
                     activation='relu', use_bias=False, name="value_conv2d")(x)
    x_value = BatchNormalization()(x_value)
    x_value = Flatten()(x_value)
    x_value = Dense(fcl, activation='relu')(x_value)
    x_value = Dense(1, name="value_output")(x_value)
    outputs_value = x_value
    model = Model(inputs=[inputs_board, inputs_hands],
                  outputs=[outputs_policy, outputs_value])

    return model
