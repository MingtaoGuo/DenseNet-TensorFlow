import tensorflow as tf
import numpy as np

def conv(name, inputs, nums_out, k_size, strides=1):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        kernel = tf.get_variable("weights", [k_size, k_size, nums_in, nums_out], initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2/(k_size*k_size*nums_in))))
        bias = tf.get_variable("bias", [nums_out], initializer=tf.constant_initializer(0.))
        inputs = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias
    return inputs

def relu(inputs):
    return tf.nn.relu(inputs)

def max_pooling(inputs, k_size):
    return tf.nn.max_pool(inputs, [1, k_size, k_size, 1], [1, 2, 2, 1], "SAME")

def avg_pooling(inputs, k_size):
    return tf.nn.avg_pool(inputs, [1, k_size, k_size, 1], [1, 2, 2, 1], "SAME")

def global_avg_pooling(inputs):
    mean, _ = tf.nn.moments(inputs, [1, 2])
    return mean

def drop_out(inputs, train_phase):
    return tf.layers.dropout(inputs, 0.2, training=train_phase)

def batchnorm(x, train_phase, scope_bn):
    #Batch Normalization
    #Ioffe S, Szegedy C. Batch normalization: accelerating deep network training by reducing internal covariate shift[J]. 2015:448-456.
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def composite_fun(name, inputs, growth_rate=12, train_phase=True):
    temp = inputs * 1.0
    with tf.variable_scope(name):
        inputs = batchnorm(inputs, train_phase, "BN")
        inputs = relu(inputs)
        inputs = conv("conv2", inputs, growth_rate, 3)
        inputs = drop_out(inputs, train_phase)
    return tf.concat([temp, inputs], axis=3)


def DenseBlock(name, inputs, nums_blocks, growth_rate=12, train_phase=True):
    with tf.variable_scope(name):
        for h in range(nums_blocks):
            inputs = composite_fun("block_"+str(h), inputs, growth_rate, train_phase)
    return inputs

def Transition(name, inputs, nums_out, train_phase):
    with tf.variable_scope(name):
        inputs = batchnorm(inputs, train_phase, "BN")
        inputs = relu(inputs)
        inputs = conv("conv", inputs, nums_out=nums_out, k_size=1)
        inputs = drop_out(inputs, train_phase)
        inputs = relu(inputs)
        inputs = avg_pooling(inputs, 2)
    return inputs

def fully_connected(name, inputs, nums_out):
    nums_in = int(inputs.shape[-1])
    with tf.variable_scope(name):
        W = tf.get_variable("weights", [inputs.shape[-1], nums_out], initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / nums_in)))
        b = tf.get_variable("bias", [nums_out])
    return tf.matmul(inputs, W) + b

def softmax(inputs):
    return tf.nn.softmax(inputs)

def to_OneHot(label, class_nums):
    return tf.one_hot(label, class_nums)

def preprocess(inputs):
    mean, var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    return (inputs - mean) / tf.sqrt(var + 1e-10)


