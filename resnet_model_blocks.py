# TensorFlow 2.x provide you with three methods to implement your own neural network architectures:
# Subclass, Functional API, Sequential
# Here, we'll use Functional API method to impliment Resnet
# https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py


from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation


# Difference between convolution_block and identity_block:
# 1. the residule path: 
# in identity_block, the residule merely equlas to input
# in convolution_block, use convolution in the path
# 2. the stride:
# in convolution_block, both the 2nd conv layer in the main path 
# and the conv layer in the residule path is (2,2)


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(
        l2_weight_decay) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   batch_norm_decay,
                   batch_norm_epsilon,
                   use_bias=False,
                   kernel_initializer='he_normal',
                   use_l2_regularizer=True):
    """
    The identity block is the block that has no conv layer at shortcut.
    Args:
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: the number of output filters in the convolution
        block: 'a','b'..., current block label, used for generating layer names

    Returns:
        Output tensor for the block.
    """

    # 1. Setting axis of batchnormalization
    # Since we might not use Theano as backend
    # we can set batchnormalization axis directly into 3
    bn_axis=3

    # 2. Setting argument to accept different filter input
    filters1, filters2, filters3 = filters

    # 2. Setting name of each convolution and batchnormalization
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 3. the main path
    x = Conv2D(filters=filters1, kernel_size=(1,1), use_bias=use_bias, kernel_initializer=kernel_initializer, \
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2a')(x)    

    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size, padding='same', use_bias=use_bias, \
        kernel_initializer=kernel_initializer, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        name=conv_name_base + '2b')(x)
    
    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer, \
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        name=conv_name_base + '2c')(x)

    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2c')(x)

    # 4 the residule
    residule=input_tensor

    # 5. combine the convolutional path and residule path
    x = add([x, residule])
    x = Activation('relu')(x)


    return x



def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               batch_norm_decay,
               batch_norm_epsilon,
               strides=(2, 2),
               use_bias=False,
               kernel_initializer='he_normal',
               use_l2_regularizer=True):
    """
    A block that has a conv layer at residule path.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: the number of output filters in the convolution
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
        
    Returns:
        Output tensor for the block.
    """
    # 1. Setting axis of batchnormalization
    # Since we might not use Theano as backend
    # we can set batchnormalization axis directly into 3
    bn_axis=3

    # 2. Setting argument to accept different filter input
    filters1, filters2, filters3 = filters

    # 2. Setting name of each convolution and batchnormalization
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    # 3. the main path
    x = Conv2D(filters=filters1, kernel_size=(1, 1), use_bias=use_bias, kernel_initializer=kernel_initializer, \
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
        name=conv_name_base + '2a')(input_tensor)

    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2a')(x)

    x = Activation('relu')(x)

    x = Conv2D(filters=filters2, kernel_size=kernel_size, strides=strides, padding='same', use_bias=use_bias, \
        kernel_initializer=kernel_initializer, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
            name=conv_name_base + '2b')(x)

    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2b')(x)

    x = Activation('relu')(x)

    x = Conv2D(filters=filters3, kernel_size=(1, 1), use_bias=use_bias, \
        kernel_initializer=kernel_initializer, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
            name=conv_name_base + '2c')(x)

    x = BatchNormalization(axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
        name=bn_name_base + '2c')(x)

    
    # 4 the residule
    residule = Conv2D(filters=filters3, kernel_size=(1, 1), strides=strides, use_bias=use_bias, \
                kernel_initializer=kernel_initializer, kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer), \
                name=conv_name_base + '1')(input_tensor)

    residule = BatchNormalization( axis=bn_axis, momentum=batch_norm_decay, epsilon=batch_norm_epsilon, \
                name=bn_name_base + '1')(residule)

    
    # 5. combine the convolutional path and residule path
    x = add([x, residule])
    x = Activation('relu')(x)


    return x
