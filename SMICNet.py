# SMICNet
import numpy
numpy.float = float
numpy.int = numpy.int_
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from neuclid.transformations import superimposition_matrix
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    DepthwiseConv2D,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    LeakyReLU,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
import tensorflow as tf
 
def SMICNet_build():
    inputs = Input(shape=(81, 81, 1))# 2D image for the input 

    x = DepthwiseConv2D((3, 3), padding="same", depth_multiplier=1)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(16, (1, 1))(x)  # Pointwise convolution
    x = LeakyReLU()(x)
    print(f"x LeakyReLU is {x.shape}")

    x = MaxPooling2D(pool_size=(2, 2))(x)
    print(f"x MaxPooling2D is {x.shape}")


    res = x
    for filters in [32, 64, 128, 256]:
        x = DepthwiseConv2D((3, 3), padding="same", depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters, (1, 1))(x)  
        x = LeakyReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        print(f"x is {x.shape}")
        if filters == 256:
            res = Conv2D(filters, (1, 1), strides=(3, 3), padding="same")(res)
        else:
            res = Conv2D(filters, (1, 1), strides=(2, 2), padding="same")(res)
        print(f"res is {res.shape}")
        x = Add()([x, res])
        res = x

    x = GlobalAveragePooling2D()(x)
    x = Dense(
        384,
        kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
    )(  # here I made the same amount channels as in the dense layer and GAP
        x
    )
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    outputs = Dense(4, activation="softmax")(x)

    SMICNet = Model(inputs, outputs)



    optimizer = tf.keras.optimizers.legacy.RMSprop(
        learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0
    )
    SMICNet.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return SMICNet

SMICNet = SMICNet_build()
 