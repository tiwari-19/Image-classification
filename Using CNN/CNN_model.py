import tensorflow as tf
import _pickle
import os

# disable the warnings by tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoint_dir')
label_dict = _pickle.load((open(os.path.join(CHECKPOINT_PATH, 'labels_dict.pkl'), 'rb')))
num_classes = len(label_dict)
img_size = 128
num_channels = 3

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))


def create_bias(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolution_layer(input, num_channels, conv_filter_size, num_filter):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_channels, num_filter])
    bias = create_bias(size=num_filter)
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer = tf.add(layer, bias)
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer =tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_bias(num_outputs)
    layer = tf.add(tf.matmul(input, weights), biases)
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='X')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_label = tf.argmax(y_true, axis=1)

layer_conv1 = create_convolution_layer(input=X,
                                       num_channels=3,
                                       conv_filter_size=3,
                                       num_filter=32)
layer_conv2 = create_convolution_layer(input=layer_conv1,
                                       num_channels=32,
                                       conv_filter_size=3,
                                       num_filter=32)
layer_conv3 = create_convolution_layer(input=layer_conv2,
                                       num_channels=32,
                                       conv_filter_size=3,
                                       num_filter=64)
layer_flat = create_flatten_layer(layer_conv3)
layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=128,
                            use_relu=True)
layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=128,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')
y_pred_label = tf.argmax(y_pred, axis=1)
# y_pred_prob = [y_pred[i][y_pred_label[i].eval()].eval() for i in range(len(y_pred.eval()))]

global_step = tf.Variable(0, trainable=False)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true))

is_correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_step = optimizer.minimize(loss, global_step=global_step)

predict = tf.argmax(tf.nn.softmax(y_pred), axis=1)