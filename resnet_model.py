# TensorFlow 2.x provide you with three methods to implement your own neural network architectures:
# Subclass, Functional API, Sequential
# Here, we'll use Functional API method to impliment Resnet
# https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py


from tensorflow.keras import Model
from resnet_model_blocks import identity_block, conv_block

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D

# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.initializers import RandomNormal


def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):

    """Instantiates the ResNet50 architecture.
    Args:
        batch_size: Size of the batches for each step.
        use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.

    Returns:
        A Keras model instance.
    """

    # 0. define block config
    block_config = dict(use_l2_regularizer=use_l2_regularizer, \
                        batch_norm_decay=batch_norm_decay, \
                        batch_norm_epsilon=batch_norm_epsilon)

    # 1. define input
    input_shape = (224, 224, 3)
    img_input = Input(shape=input_shape, batch_size=batch_size)

    # 2. Setting axis of batchnormalization
    # Since we might not use Theano as backend
    # we can set batchnormalization axis directly into 3
    bn_axis=3

    # 3. model
    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', \
        use_bias=False, kernel_initializer='he_normal', kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        name='conv1')(x)        
    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name='bn_conv1')(x)        
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', **block_config)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', **block_config)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', **block_config)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', **block_config)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', **block_config)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', **block_config)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', **block_config)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', **block_config)

    #
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, kernel_initializer=RandomNormal(stddev=0.01), \
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        bias_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        name='fc1000')(x) # fc1000=1000 dimension fully connected layer's output

    # in float16 due to numeric issues. So we pass dtype=float32.
    x = Activation('softmax', dtype='float32')(x)

    output = Model(img_input, x, name='resnet50')

    return output