from BotDeterminant.cv.config import *

from itertools import repeat
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Concatenate, Lambda, Conv2D, Input, LeakyReLU, UpSampling2D, ZeroPadding2D


# The function to calculate IoU


def iOU(box1, box2):
    int_w = max(0, min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin))
    int_h = max(0, min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin))

    return float(int_w * int_h) / (box1.area + box2.area - int_w * int_h)


def DarknetConv(x, filters, size, strides=1, batch_norm=True):
    padding = 'same' if strides == 1 else 'valid'
    x = ZeroPadding2D(((1, 0), (1, 0)))(x) if strides != 1 else x
    x = Conv2D(filters=filters, kernel_size=size, strides=strides, padding=padding, use_bias=not batch_norm,
               kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    x = tf.keras.layers.BatchNormalization()(x) if batch_norm else x
    return LeakyReLU(alpha=0.1)(x)


def DarknetResidual(x, filters):
    return Add()([x, DarknetConv(DarknetConv(x, filters // 2, 1), filters, 3)])


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None):
    inputs = Input([None, None, 3])
    conv = DarknetConv(inputs, 32, 3)
    conv = DarknetBlock(conv, 64, 1)
    conv = DarknetBlock(conv, 128, 2)
    conv_36 = DarknetBlock(conv, 256, 8)
    conv_61 = DarknetBlock(conv_36, 512, 8)
    conv = DarknetBlock(conv_61, 1024, 4)
    return tf.keras.Model(inputs, (conv_36, conv_61, conv), name=name)


def YoloCv(filters, name=None):
    def yolo_conv(inp):
        if type(inp) == tuple:
            inputs = Input(inp[0].shape[1:]), Input(inp[1].shape[1:])
            conv, conv1 = inputs
            conv = Concatenate()([UpSampling2D(2)(DarknetConv(conv, filters, 1)), conv1])
        else:
            conv = inputs = Input(inp.shape[1:])
        for i in range(2):
            conv = DarknetConv(DarknetConv(conv, filters, 1), filters * 2, 3)
        conv = DarknetConv(conv, filters, 1)
        return Model(inputs, conv, name=name)(inp)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_input):
        x = inputs = Input(x_input.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(
            x)
        return tf.keras.Model(inputs, x, name=name)(x_input)

    return yolo_output


def yolo_boxes(pred, anchors, num_classes):
    grid_size = tf.shape(pred)[1]
    num_anchors = len(anchors)

    # Reshape the prediction tensor
    pred = tf.reshape(pred, (-1, grid_size, grid_size, num_anchors, num_classes + 5))

    # Split the prediction tensor into respective components
    pred_xy = tf.sigmoid(pred[..., :2])
    pred_wh = tf.exp(pred[..., 2:4]) * anchors
    pred_confidence = tf.sigmoid(pred[..., 4:5])
    pred_class_probs = tf.sigmoid(pred[..., 5:])

    # Create grid offsets
    grid_range = tf.range(grid_size, dtype=tf.float32)
    grid_x, grid_y = tf.meshgrid(grid_range, grid_range)
    grid_x = tf.reshape(grid_x, (1, grid_size, grid_size, 1, 1))
    grid_y = tf.reshape(grid_y, (1, grid_size, grid_size, 1, 1))

    # Calculate box coordinates relative to the grid
    pred_xy += tf.concat([grid_x, grid_y], axis=-1)
    pred_xy /= tf.cast(grid_size, tf.float32)

    # Calculate bounding box coordinates
    pred_x1y1 = pred_xy - pred_wh / 2
    pred_x2y2 = pred_xy + pred_wh / 2
    box = tf.concat([pred_x1y1, pred_x2y2], axis=-1)

    return box, pred_confidence, pred_class_probs, pred[..., :4]


# The function to suppress non-maximum

def nMS(outputs, anchors, masks, classes):
    bbox = tf.concat(
        [tf.reshape(output[0], (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1])) for output in outputs], axis=1)
    confidence = tf.concat(
        [tf.reshape(output[1], (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1])) for output in outputs], axis=1)
    class_probs = tf.concat(
        [tf.reshape(output[2], (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1])) for output in outputs], axis=1)

    scores = tf.multiply(confidence, class_probs)

    reshaped_bbox = tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4))
    reshaped_scores = tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1]))

    x = tf.image.combined_non_max_suppression(
        boxes=reshaped_bbox,
        scores=reshaped_scores,
        max_output_size_per_class=100,
        max_total_size=100,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold)

    return x[0], x[1], x[2], x[3]


# The main function

def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    inputs = Input(shape=[size, size, channels])

    darknet = Darknet(name='yolo_darknet')
    x_36, x_61, x = darknet(inputs)

    x = YoloCv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloCv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloCv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: nMS(x, anchors, masks, classes),
                     name='nMS')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


# The loss function

from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy

binary_crossentropy = BinaryCrossentropy(from_logits=True)
sparse_categorical_crossentropy = SparseCategoricalCrossentropy(from_logits=True)


def YoloLoss(true_labels, predicted_labels, custom_anchors, num_classes=80, ignore_threshold=0.5):
    binary_crossentropy = BinaryCrossentropy(from_logits=True)
    sparse_categorical_crossentropy = SparseCategoricalCrossentropy(from_logits=True)

    predicted_boxes, predicted_objects, predicted_classes, predicted_xywh = yolo_boxes(predicted_labels, custom_anchors,
                                                                                       num_classes)
    predicted_xy = predicted_xywh[..., 0:2]
    predicted_wh = predicted_xywh[..., 2:4]

    true_boxes, true_objects, true_class_indices = tf.split(true_labels, (4, 1, 1), axis=-1)
    true_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2
    true_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]

    object_mask = tf.squeeze(true_objects, -1)
    true_boxes_flat = tf.boolean_mask(true_boxes, tf.cast(object_mask, tf.bool))
    best_iou = tf.reduce_max(intersection_over_union(predicted_boxes, true_boxes_flat), axis=-1)
    ignore = tf.cast(best_iou < ignore_threshold, tf.float32)

    lossxy = tf.square(true_xy - predicted_xy)
    losswh = tf.square(true_wh - predicted_wh)
    lossobject = binary_crossentropy(true_objects, predicted_objects) + (
                1 - object_mask) * ignore * binary_crossentropy(true_objects, predicted_objects)
    l_class = sparse_categorical_crossentropy(true_class_indices, predicted_classes)

    lossxy = tf.reduce_sum(lossxy, axis=(1, 2, 3))
    losswh = tf.reduce_sum(losswh, axis=(1, 2, 3))
    lossobject = tf.reduce_sum(lossobject, axis=(1, 2, 3))
    l_class = tf.reduce_sum(l_class, axis=(1, 2, 3))

    random_factor = tf.random.uniform(shape=tf.shape(lossxy), minval=0, maxval=1)
    lossxy = random_factor * lossxy
    losswh = random_factor * losswh
    lossobject = random_factor * lossobject
    l_class = random_factor * l_class

    total_loss = lossxy + losswh + lossobject + l_class

    return total_loss


# The function to transform targets outputs tuple of shape

@tf.function
def transtargets_for_output(y_true, grid_size, anchor_idxs, classes):
    N = tf.shape(y_true)[0]
    y_true_out = tf.zeros((N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))
    anchor_idxs = tf.cast(anchor_idxs, tf.int32)
    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            _ = indexes.write(idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]]) if tf.reduce_any(
                tf.equal(anchor_idxs, tf.cast(y_true[i, j, 5], tf.int32))) else None
            _ = updates.write(idx, [box[0], box[1], box[2], box[3], 1, y_true[i, j, 4]]) if tf.reduce_any(
                tf.equal(anchor_idxs, tf.cast(y_true[i, j, 5], tf.int32))) else None
            idx += tf.cast(tf.reduce_any(tf.equal(anchor_idxs, tf.cast(y_true[i, j, 5], tf.int32))), tf.int32)
    return tf.tensor_scatter_nd_update(y_true_out, indexes.stack(), updates.stack())


def transtargets(y_train, anchors, anchor_masks, classes):
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2), (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, -1)
    y_train = tf.concat([y_train, anchor_idx], -1)

    return tuple(
        transtargets_for_output(y_train, 13 * (2 ** idx), anchor_idxs, classes)
        for idx, anchor_idxs in enumerate(anchor_masks)
    )
