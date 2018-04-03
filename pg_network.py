import state
import math
import numpy as np
import os
import tensorflow as tf

class GradientClippingOptimizer(tf.train.Optimizer):
    def __init__(self, optimizer, use_locking=False, name="GradientClipper"):
        super(GradientClippingOptimizer, self).__init__(use_locking, name)
        self.optimizer = optimizer

    def compute_gradients(self, *args, **kwargs):
        grads_and_vars = self.optimizer.compute_gradients(*args, **kwargs)
        clipped_grads_and_vars = []
        for (grad, var) in grads_and_vars:
            if grad is not None:
                clipped_grads_and_vars.append((tf.clip_by_value(grad, -1, 1), var))
            else:
                clipped_grads_and_vars.append((grad, var))
        return clipped_grads_and_vars

    def apply_gradients(self, *args, **kwargs):
        return self.optimizer.apply_gradients(*args, **kwargs)

class PolicyGradientNetwork:
    def __init__(self, numActions, baseDir, args):
        
        self.numActions = numActions
        self.baseDir = baseDir
        self.saveModelFrequency = args.save_model_freq
        self.normalizeWeights = args.normalize_weights

        self.staleSess = None

        tf.set_random_seed(123456)
        
        self.sess = tf.Session()
        
        assert (len(tf.global_variables()) == 0),"Expected zero variables"
        self.x, self.y = self.buildNetwork('policy', True, numActions)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.global_variables()) == 10),"Expected 10 total variables"
        self.x_target, self.y_target = self.buildNetwork('target', False, numActions)
        assert (len(tf.trainable_variables()) == 10),"Expected 10 trainable_variables"
        assert (len(tf.global_variables()) == 20),"Expected 20 total variables"

        self.selected_action = tf.placeholder(tf.float32, shape=[None, numActions])
        print('selected_action %s' % (self.selected_action.get_shape()))

        good_probabilities = tf.reduce_sum(tf.multiply(self.y, self.selected_action), reduction_indices=[1])
        print('good_probabilities %s' % (good_probabilities.get_shape()))
        log_probabilities = tf.log(good_probabilities)
        self.loss = -tf.reduce_sum(log_probabilities)
        
        if args.use_rms_prop:
            #optimizer = tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01)
            optimizer = GradientClippingOptimizer(tf.train.RMSPropOptimizer(args.learning_rate, decay=.95, epsilon=.01))
        else:
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
        
        self.train_step = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver(max_to_keep=25)

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())

        self.summary_writer = tf.summary.FileWriter(self.baseDir + '/tensorboard', self.sess.graph)

        if args.model is not None:
            print('Loading from model file %s' % (args.model))
            self.saver.restore(self.sess, args.model)

    def buildNetwork(self, name, trainable, numActions):
        
        print("Building network for %s trainable=%s" % (name, trainable))

        # First layer takes a screen, and shrinks by 2x
        x = tf.placeholder(tf.uint8, shape=[None, 84, 84, 4], name="screens")
        print(x)

        x_normalized = tf.to_float(x) / 255.0
        print(x_normalized)

        # Second layer convolves 32 8x8 filters with stride 4 with relu
        with tf.variable_scope("cnn1_" + name):
            W_conv1, b_conv1 = self.makeLayerVariables([8, 8, 4, 32], trainable, "conv1")

            h_conv1 = tf.nn.relu(tf.nn.conv2d(x_normalized, W_conv1, strides=[1, 4, 4, 1], padding='VALID') + b_conv1, name="h_conv1")
            print(h_conv1)

        # Third layer convolves 64 4x4 filters with stride 2 with relu
        with tf.variable_scope("cnn2_" + name):
            W_conv2, b_conv2 = self.makeLayerVariables([4, 4, 32, 64], trainable, "conv2")

            h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID') + b_conv2, name="h_conv2")
            print(h_conv2)

        # Fourth layer convolves 64 3x3 filters with stride 1 with relu
        with tf.variable_scope("cnn3_" + name):
            W_conv3, b_conv3 = self.makeLayerVariables([3, 3, 64, 64], trainable, "conv3")

            h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID') + b_conv3, name="h_conv3")
            print(h_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64], name="h_conv3_flat")
        print(h_conv3_flat)

        # Fifth layer is fully connected with 512 relu units
        with tf.variable_scope("fc1_" + name):
            W_fc1, b_fc1 = self.makeLayerVariables([7 * 7 * 64, 512], trainable, "fc1")

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1, name="h_fc1")
            print(h_fc1)

        # Sixth (Output) layer is fully connected softmax layer
        with tf.variable_scope("fc2_" + name):
            W_fc2, b_fc2 = self.makeLayerVariables([512, numActions], trainable, "fc2")

            y = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
            print(y)
            
        return x, y

    def makeLayerVariables(self, shape, trainable, name_suffix):
        if self.normalizeWeights:
            # This is my best guess at what DeepMind does via torch's Linear.lua and SpatialConvolution.lua (see reset methods).
            # np.prod(shape[0:-1]) is attempting to get the total inputs to each node
            stdv = 1.0 / math.sqrt(np.prod(shape[0:-1]))
            weights = tf.Variable(tf.random_uniform(shape, minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.random_uniform([shape[-1]], minval=-stdv, maxval=stdv), trainable=trainable, name='W_' + name_suffix)
        else:
            weights = tf.Variable(tf.truncated_normal(shape, stddev=0.01), trainable=trainable, name='W_' + name_suffix)
            biases  = tf.Variable(tf.fill([shape[-1]], 0.1), trainable=trainable, name='W_' + name_suffix)
        return weights, biases
        
    def inference(self, screens):
        y = self.sess.run([self.y], {self.x: screens})
        y = np.squeeze(y)
        return np.random.choice(y.size, p=y)

    def train(self, x, selected_action):
        # one hot encode the selected actions
        selected_action = [np.eye(self.numActions)[i] for i in selected_action]

        _, loss = self.sess.run([self.train_step, self.loss], feed_dict={
            self.x: x,
            self.selected_action: selected_action,
        })
        return loss
