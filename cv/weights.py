from config import *
from model import YoloV3

import os
import subprocess

import tensorflow as tf


def load_darknet_weights(model, weights_file):
    with open(weights_file, 'rb') as wf:
        _ = np.fromfile(wf, dtype=np.int32, count=5)
        custom_layer_names = YOLO_V3_LAYERS

        for custom_layer_name in custom_layer_names:
            custom_sub_model = model.get_layer(custom_layer_name)
            for custom_idx, custom_layer in enumerate(custom_sub_model.layers):
                if isinstance(custom_layer, tf.keras.layers.Conv2D):
                    custom_batch_norm = None
                    custom_num_filters = custom_layer.get_config()['filters']
                    custom_kernel_size = custom_layer.kernel_size[0]
                    custom_input_dim = custom_layer.input_shape[-1]
                    if custom_idx + 1 < len(custom_sub_model.layers) and \
                            isinstance(custom_sub_model.layers[custom_idx + 1], tf.keras.layers.BatchNormalization):
                        custom_batch_norm = custom_sub_model.layers[custom_idx + 1]

                    if not custom_batch_norm:
                        custom_conv_bias = np.fromfile(wf, dtype=np.float32, count=custom_num_filters)
                    else:
                        custom_bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * custom_num_filters)
                        custom_bn_weights = custom_bn_weights.reshape((4, custom_num_filters))[[1, 0, 2, 3]]

                    custom_conv_shape = (custom_num_filters, custom_input_dim, custom_kernel_size, custom_kernel_size)
                    custom_conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(custom_conv_shape))
                    custom_conv_weights = custom_conv_weights.reshape(custom_conv_shape).transpose([2, 3, 1, 0])

                    if not custom_batch_norm:
                        custom_layer.set_weights([custom_conv_weights, custom_conv_bias])
                    else:
                        custom_layer.set_weights([custom_conv_weights])
                        custom_batch_norm.set_weights(custom_bn_weights)



if not os.path.exists("checkpoints"):
    subprocess.run(["wget", "https://pjreddie.com/media/files/yolov3.weights"])
    subprocess.run(["mkdir", "checkpoints"])
    
yolo_model = YoloV3(classes=num_classes)

load_darknet_weights(yolo_model, weightyolov3)

    
yolo_model.save_weights(checkpoints)
