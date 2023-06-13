from config import *

import tensorflow as tf



class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding="same", activation=True):
        """ filters     : int
            kernel_size : int
            strides     : int
        """
        assert padding == "same", "padding = '%s' not implemented?" % padding
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        if strides == 1:
            self.conv = tf.keras.layers.SeparableConv1D(self.filters, self.kernel_size, self.strides, self.padding)
        else:
            self.conv = tf.keras.layers.SeparableConv1D(self.filters, self.kernel_size, self.strides, "valid")
        test = 4
        self.beta = tf.Variable(tf.zeros(self.filters), trainable=True)
        self.gamma = tf.Variable(tf.ones(self.filters), trainable=True)

        self.population_mean = tf.Variable(tf.zeros(self.filters), trainable=False)
        self.population_variance = tf.Variable(tf.ones(self.filters), trainable=False)

    def _batch_norm(self, x, x_len, momentum=0.99, epsilon=0.001, training=None):
        if training:
            num_timesteps = tf.cast(tf.reduce_sum(x_len), dtype=tf.float32)
            batch_size = tf.shape(x)[0]

            x_reshape = tf.reshape(x, [-1, self.filters])

            # TODO Any gradient issue here? Need to check paper
            batch_mean = tf.math.reduce_sum(x_reshape, axis=0) / num_timesteps
            batch_variance = tf.math.squared_difference(x_reshape, batch_mean)
            batch_variance = batch_variance * tf.reshape(tf.sequence_mask(x_len, dtype=tf.float32), [-1, 1])
            batch_variance = tf.math.reduce_sum(batch_variance, axis=0) / num_timesteps

            # Update population statistics
            self.population_mean.assign(self.population_mean * momentum + batch_mean * (1 - momentum))
            self.population_variance.assign(self.population_variance * momentum + batch_variance * (1 - momentum))
            x = tf.nn.batch_normalization(x, batch_mean, batch_variance, self.beta, self.gamma, epsilon)
        else:
            # TODO Can population mean and variance be issue for initial validation steps?
            x = tf.nn.batch_normalization(x, self.population_mean, self.population_variance,
                                          self.beta, self.gamma, epsilon)
        return x * tf.expand_dims(tf.sequence_mask(x_len, dtype=tf.float32), -1)

    def _convolution(self, x, x_len):
        """ SeparableConv1D for padded input and "same" padding
            Not verified / tested for "valid" padding
        """
        if self.strides > 1:
            final_timesteps = tf.cast(tf.math.ceil(x_len / self.strides), dtype="int32")
            required_length = self.strides * (final_timesteps - 1) + self.kernel_size
            num_padding = required_length - x_len
            left_padding = num_padding // 2
            right_padding = num_padding - left_padding
            max_left_padding = self.kernel_size // 2
            max_right_padding = self.kernel_size - max_left_padding

            # Zero padding
            batch_size = tf.shape(x)[0]
            feat_dim = tf.shape(x)[-1]
            x_max_padded = tf.concat([tf.zeros([batch_size, max_left_padding, feat_dim]),
                                     x,
                                     tf.zeros([batch_size, max_right_padding, feat_dim])], 1)
            max_required_length = tf.math.reduce_max(required_length)
            start_timesteps = max_left_padding - left_padding

            # Work around for x = x_max_padded[:, start_timesteps:, :]
            idx = tf.expand_dims(tf.range(max_required_length), 0) + tf.expand_dims(start_timesteps, 1)
            batch_id = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1]), [1, max_required_length, 1])
            idx = tf.concat((batch_id, tf.reshape(idx, [batch_size, -1, 1])), -1)
            x = tf.gather_nd(x_max_padded, idx)

        x_len = tf.cast(tf.math.ceil(x_len / self.strides), dtype="int32")
        mask = tf.expand_dims(tf.sequence_mask(x_len, dtype=tf.float32), -1)
        x = mask * self.conv(x)
        return x, x_len

    def call(self, x, x_len, training=None):
        """ x : (B, T, F) """
        x, x_len = self._convolution(x, x_len)
        x = self._batch_norm(x, x_len, training=training)
        if self.activation:
            x = tf.nn.swish(x)
        return x, x_len



class SE(tf.keras.layers.Layer):
    """
    Making Squeeze-and-Excitation layer
    original paper: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, units):
        super(SE, self).__init__()
        self.units = units
        self.full_conv_layers = []

        for unit in units:
            self.full_conv_layers.append(tf.keras.layers.Dense(unit))

    def call(self, x, x_len):
        copy_x = x
        x = tf.reduce_sum(x, axis=1) / tf.expand_dims(tf.cast(x_len, tf.float32), 1)
        for i in range(len(self.units)):
            x = tf.nn.swish(self.full_conv_layers[i](x))
        x = tf.expand_dims(tf.nn.sigmoid(x), 1)
        return x * copy_x


class MakeBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        se_units,
        count_layers: int,
        filters: int,
        kernel_size: int,
        strides=1,
        residual=True,
    ):
        super(MakeBlock, self).__init__()
        self.num_layers = count_layers
        self.se = SE(se_units) if se_units else None
        self.residual = (
            ConvLayer(filters, kernel_size, strides, activation=False)
            if residual
            else None
        )

        self.conv_layers = []
        strides = [strides] + [1] * (count_layers - 1)
        for stride in strides:
            self.conv_layers.append(ConvLayer(filters, kernel_size, stride))

    def call(self, x, x_len, training=None):
        x_orig = x
        x_len_orig = x_len
        for conv_layer in self.conv_layers:
            x, x_len = conv_layer(x, x_len, training=training)

        if self.residual is None and self.se is None:
            return x
        if self.se is not None:
            x = self.se(x, x_len)
        if self.residual is not None:
            x = x + self.residual(x_orig, x_len_orig)[0]

        return tf.nn.swish(x), x_len


class AudioEncoder(tf.keras.layers.Layer):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        # Making Convolution blocks
        blocks = []
        blocks += [
            MakeBlock([int(256 * alpha), 256], 1, 256, 5, 1, residual=False)
        ]  # C0
        blocks += [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 1)]  # C1
        blocks += [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 1)]  # C2
        blocks += [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 2)]  # C3
        blocks.extend(
            [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 1) for i in range(3)]
        )  # C4-6
        blocks += [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 2)]  # C7
        blocks.extend(
            [MakeBlock([int(256 * alpha), 256], 5, 256, 5, 1) for i in range(3)]
        )  # C8-10
        blocks.extend(
            [MakeBlock([int(512 * alpha), 512], 5, 512, 5, 1) for i in range(3)]
        )  # C11-13
        blocks += [MakeBlock([int(512 * alpha), 512], 5, 512, 5, 2)]  # C14
        blocks.extend(
            [MakeBlock([int(512 * alpha), 512], 5, 512, 5, 1) for i in range(7)]
        )  # C15-21
        blocks += [
            MakeBlock([int(640 * alpha), 640], 1, 640, 5, 1, residual=False)
        ]  # C22

        self.blocks = blocks

    def call(self, x, x_len, training=None):
        for block in self.blocks:
            x, x_len = block(x, x_len, training=training)
        return x, x_len


class LabelEncoder(tf.keras.layers.Layer):
    def __init__(self, count_layers, units, output_dim, num_vocab, embedding_dim):
        super(LabelEncoder, self).__init__()
        self.num_layers = count_layers
        self.num_units = units
        self.out_dim = output_dim

        self.embedding = tf.keras.layers.Embedding(num_vocab, embedding_dim)
        self.lstms, self.projection = [], []
        for i in range(count_layers):
            self.lstms.append(tf.keras.layers.LSTM(units, return_sequences=True))
            self.projection.append(tf.keras.layers.Dense(output_dim))

    def call(self, y, y_len):
        y = tf.pad(y, [[0, 0], [1, 0]])
        y_len += 1
        y = self.embedding(y)
        for i in range(self.num_layers):
            mask = tf.sequence_mask(y_len)
            y = self.projection[i](self.lstms[i](y, mask=mask))
        return y


class ASR(tf.keras.Model):
    def __init__(self, num_units, num_vocabulary, count_lstms, lstm_units, output_dim):
        super(ASR, self).__init__()
        self.num_units = num_units
        self.num_vocab = num_vocabulary

        self.audio_encoder = AudioEncoder()
        self.label_encoder = LabelEncoder(
            count_lstms, lstm_units, output_dim, num_vocabulary, num_units
        )

        # Joiner
        self.projection = tf.keras.layers.Dense(num_units)
        self.output_layer = tf.keras.layers.Dense(num_vocabulary + 1)

    def call(self, x, y, x_len, y_len, training=None):
        x, x_len = self.audio_encoder(x, x_len, training=training)
        y = self.label_encoder(y, y_len)

        x = tf.expand_dims(x, 2)
        y = tf.expand_dims(y, 1)

        z = tf.nn.tanh(self.projection(x + y))
        return self.output_layer(z), x_len, y_len
