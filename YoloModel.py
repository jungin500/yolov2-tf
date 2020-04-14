from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, Dense, BatchNormalization, Input, \
    LeakyReLU, Reshape, Flatten, Softmax, Lambda, Concatenate, UpSampling3D

import tensorflow as tf

def LeakyNormConv2D(input, filter_size, kernel_size, strides=(1, 1), block_name=''):
    # bias를 쓰지 않는 이유? batch_norm?
    x = Conv2D(
        filter_size,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation=None,
        # use_bias=False,
        name=block_name + '_conv'
    )(input)
    x = BatchNormalization(name=block_name + '_bn')(x)
    return LeakyReLU(alpha=0.1, name=block_name + '_lru')(x)


def SpaceToDepth(x):
    return tf.nn.space_to_depth(x, block_size=2)


def ResBlock(prev_input, skip_connection, filter_size, kernel_size, block_name=''):
    x = LeakyNormConv2D(skip_connection, filter_size=filter_size, kernel_size=kernel_size, block_name=block_name + '_leaky')
    x = Lambda(SpaceToDepth, name=block_name + '_sp2dep')(x)  # output tensor: [B, 7, 7, 2048]
    return Concatenate(name=block_name + '_cat')([x, prev_input]) # concatenate [[None, 7, 7, 2048], [None, 7, 7, 1024]]


'''
    - [ ] Global Average Pooling
    - [ ] Batch Normalization
    - [ ] Stochastic Gradient Descent
    
    - [ ] lr = 0.1
    - [ ] polynomial rate decay = 4
    - [ ] weight decay = 0.0005
    - [ ] momentum = 0.9
    
    - [ ] random crops
    - [ ] rotations
    - [ ] hue
    - [ ] saturation
    - [ ] exposure shifts
'''
def Yolov2Model():
    inputs = Input(shape=(416, 416, 3))

    # L1
    x = LeakyNormConv2D(inputs, filter_size=32, kernel_size=3, block_name='leaky_l1')

    # L2
    x = MaxPool2D(name='l2_maxpool')(x)
    x = LeakyNormConv2D(x, filter_size=64, kernel_size=3, block_name='leaky_l2')

    # L3
    x = MaxPool2D(name='l3_maxpool')(x)
    x = LeakyNormConv2D(x, filter_size=128, kernel_size=3, block_name='leaky_l3_s1')
    x = LeakyNormConv2D(x, filter_size=64, kernel_size=1, block_name='leaky_l3_s2')
    x = LeakyNormConv2D(x, filter_size=128, kernel_size=3, block_name='leaky_l3_s3')

    # L4
    x = MaxPool2D(name='l4_maxpool')(x)
    x = LeakyNormConv2D(x, filter_size=256, kernel_size=3, block_name='leaky_l4_s1')
    x = LeakyNormConv2D(x, filter_size=128, kernel_size=1, block_name='leaky_l4_s2')
    x = LeakyNormConv2D(x, filter_size=256, kernel_size=3, block_name='leaky_l4_s3')

    # L5
    x = MaxPool2D(name='l5_maxpool')(x)
    x = LeakyNormConv2D(x, filter_size=512, kernel_size=3, block_name='leaky_l5_s1')
    x = LeakyNormConv2D(x, filter_size=256, kernel_size=1, block_name='leaky_l5_s2')
    x = LeakyNormConv2D(x, filter_size=512, kernel_size=3, block_name='leaky_l5_s3')
    x = LeakyNormConv2D(x, filter_size=256, kernel_size=1, block_name='leaky_l5_s4')
    x = LeakyNormConv2D(x, filter_size=512, kernel_size=3, block_name='leaky_l5_s5')

    # L5-Out
    skip_connection = x

    # L6
    x = MaxPool2D(name='l6_maxpool')(x)
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l6_s1')
    x = LeakyNormConv2D(x, filter_size=512, kernel_size=1, block_name='leaky_l6_s2')
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l6_s3')
    x = LeakyNormConv2D(x, filter_size=512, kernel_size=1, block_name='leaky_l6_s4')
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l6_s5')
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l6_s6')  # Impl by github
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l6_s7')

    # L7
    x = ResBlock(x, skip_connection=skip_connection, filter_size=64, kernel_size=1, block_name='res_l7')
    x = LeakyNormConv2D(x, filter_size=1024, kernel_size=3, block_name='leaky_l7')

    # L8 (Final)
    # 5 * Box[{ t_x, t_y, t_w, t_h, t_c } + { class_0 ~ class_19 }] = 5 * (5 + 20) = 5 * 25 = 125
    x = Conv2D(filters=125, kernel_size=1, padding='same', name='conv_l8', use_bias=False)(x)
    x = Reshape((13, 13, 5, 25))(x)  # 결과값: 13 * 13 * 5 * 25

    model = Model(inputs=inputs, outputs=x)
    return model
