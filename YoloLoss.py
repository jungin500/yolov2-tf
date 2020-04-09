import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np
import sys

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

'''
    pred_mins, pred_maxes: shape(? * 13 * 13 * 5 * 2)
    true_mins, true_maxes: shape(? * 13 * 13 * 5 * 2)
    
    returns [? * 13 * 13 * 5]
'''
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)  # ? * 13 * 13 * 5 * 2
    intersect_maxes = K.minimum(pred_maxes, true_maxes)  # ? * 13 * 13 * 5 * 2
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # ? * 13 * 13 * 5 * 2
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # ? * 13 * 13 * 5

    pred_wh = pred_maxes - pred_mins  # ? * 13 * 13 * 5 * 2
    true_wh = true_maxes - true_mins  # ? * 13 * 13 * 5 * 2
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # ? * 13 * 13 * 5
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # ? * 13 * 13 * 5

    union_areas = pred_areas + true_areas - intersect_areas  # ? * 13 * 13 * 5
    iou_scores = intersect_areas / union_areas  # ? * 13 * 13 * 5

    return iou_scores  # ? * 13 * 13 * 5


'''
    box_xy, box_wh -> [B * 13 * 13 * 5 * 1 * 2]
'''
ANCHOR_BOXES = [1.19, 1.99, 2.79, 4.60, 4.54, 8.93, 8.06, 5.29, 10.33, 10.65]
ANCHOR_BOXES_TF = tf.reshape(
    tf.constant(ANCHOR_BOXES),
    shape=(1, 1, 1, 5, 2)
)


def get_bbox_anchor(box_wh):
    anchor_box = ANCHOR_BOXES_TF
    anchor_ratio = anchor_box[..., 0] / anchor_box[..., 1]  # width / height, 1 * 1 * 1 * 5 * 1
    box_ratio = box_wh[..., 0] / box_wh[..., 1]  # width / height, ? * 13 * 13 * 5 * 1
    ratio_diff = tf.abs(anchor_ratio - box_ratio)  # ? * 13 * 13 * 5 * 1
    ratio_diff = tf.reshape(ratio_diff, shape=(-1, 13, 13, 5))  # ? * 13 * 13 * 5
    ratio_diff = tf.cast(ratio_diff, tf.float32)

    bbox_anchor_nms_mask = tf.where(  # Find indices that is max between 5 anchors
        ratio_diff > tf.reduce_min(ratio_diff, axis=3, keepdims=True),
        0.,   # Set to zero if not a min
        1.,  # Set to ratio_diff original value
    )  # ? * 13 * 13 * 5
    return tf.reshape(bbox_anchor_nms_mask, shape=(-1, 13, 13, 5, 1, 1))  # 5 -> (5 * 1 * 1)


def get_anchor_mask():
    anchor_box = ANCHOR_BOXES_TF
    return tf.reshape(anchor_box, shape=(1, 1, 1, 5, 1, 2))


# Returns 1 * 13  * 13 * 1 * 1 * 2
def cell_offset_table():
    # Dynamic implementation of conv dims for fully convolutional model.
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=13)
    conv_width_index = K.arange(0, stop=13)
    conv_height_index = K.tile(conv_height_index, [13]) # 늘어놓는 함수  tile -> 같은걸 N번 반복함
    # 결과 -> 0~12, 0~12, ...., 0~12

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [13, 1]) # tile을 [n, m] 쓰면 dims 2로 만들어줌
    # 결과 -> [0~12], [0~12], [0~12], ...

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    # 결과 -> 0, 0, 0, 0, 0, 0, 0 (13개), 1, 1, 1, 1, 1, 1, 1 (13개), ...

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # 결과 -> [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]

    conv_index = K.reshape(conv_index, [1, 13, 13, 1, 2])
    # 결과 -> 1 * 13 * 13 에 있는 [1 * 2]의 conv index item이 만들어짐
    # 각각 [1 * 2]의 값은 [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]
    # 이런 식으로 이루어져 있음 -> Mask를 만들기 위한 과정
    # 결과 shape -> 1, 13, 13, 1, 2

    conv_index = K.cast(conv_index, tf.float32)

    return conv_index


def Yolov2Loss(y_true, y_pred):
    label_class = y_true[..., :20]                      # ? * 13 * 13 * 5 * 20
    label_box = y_true[..., 20:24]                      # ? * 13 * 13 * 5 * 4
    responsible_mask = y_true[..., 24]                  # ? * 13 * 13 * 5

    predict_class = y_pred[..., :20]                        # ? * 13 * 13 * 5 * 20
    predict_box = y_pred[..., 20:24]                        # ? * 13 * 13 * 5 * 4
    predict_bbox_confidences = y_pred[..., 24]              # ? * 13 * 13 * 5

    label_txty, label_twth = label_box[..., :2], label_box[..., 2:4]          # ? * 13 * 13 * 5 * 2, ? * 13 * 13 * 5 * 2
    label_bxby = tf.sigmoid(label_txty) + cell_offset_table()                 # ? * 13 * 13 * 5 * 2
    label_bwbh = ANCHOR_BOXES_TF * tf.math.exp(label_twth)                    # ? * 13 * 13 * 5 * 2
    label_bxby = tf.expand_dims(label_bxby, 4)                                # ? * 13 * 13 * 5 * 1 * 2
    label_bwbh = tf.expand_dims(label_bwbh, 4)                                # ? * 13 * 13 * 5 * 1 * 2
    label_bxby_min, label_bxby_max = xywh2minmax(label_bxby, label_bwbh)      # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2

    predict_txty, predict_twth = predict_box[..., :2], predict_box[..., 2:4]      # ? * 13 * 13 * 5 * 2, ? * 13 * 13 * 5 * 2
    predict_bxby = tf.sigmoid(predict_txty) + cell_offset_table()                 # ? * 13 * 13 * 5 * 2
    predict_bwbh = ANCHOR_BOXES_TF * tf.math.exp(predict_twth)                    # ? * 13 * 13 * 5 * 2
    predict_bxby = tf.expand_dims(predict_bxby, 4)                                # ? * 13 * 13 * 5 * 1 * 2
    predict_bwbh = tf.expand_dims(predict_bwbh, 4)                                # ? * 13 * 13 * 5 * 1 * 2
    predict_bxby_min, predict_bxby_max = xywh2minmax(predict_bxby, predict_bwbh)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2

    iou_scores = iou(predict_bxby_min, predict_bxby_max, label_bxby_min, label_bxby_max)  # ? * 13 * 13 * 5 * 1
    best_ious = K.max(iou_scores, axis=4)                                                 # ? * 13 * 13 * 5
    best_box = K.max(best_ious, axis=3, keepdims=True)                                    # ? * 13 * 13 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 13 * 13 * 5

    # _label_box = K.reshape(label_box, [-1, 13, 13, 5, 1, 4])           # (원본) ? * 13 * 13 * 5 * 4
    # _predict_box = K.reshape(predict_box, [-1, 13, 13, 5, 1, 4])       # (원본) ? * 13 * 13 * 5 * 4

    # label_xy, label_wh = yolo_head(_label_box)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2
    # predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2
    label_xy, label_wh = label_box[..., 2:], label_box[..., 2:4]   # 각 ? * 13 * 13 * 5 * 2
    predict_xy, predict_wh = predict_box[..., 2:], predict_box[..., 2:4]  # 각 ? * 13 * 13 * 5 * 2

    box_mask = K.expand_dims(box_mask)  # ? * 13 * 13 * 5 * 1
    responsible_mask = K.expand_dims(responsible_mask)  # ? * 13 * 13 * 5 * 1

    # Loss 함수 1번
    box_loss = 5 * box_mask * responsible_mask * K.square(label_xy - predict_xy)

    # Loss 함수 2번
    box_loss += 5 * box_mask * responsible_mask * K.square(label_wh - predict_wh)

    # 1번+2번 총합
    box_loss = K.sum(box_loss)

    predict_bbox_confidences = K.expand_dims(predict_bbox_confidences)

    # Loss 함수 3번 (without lambda_noobj)
    object_loss = box_mask * responsible_mask * K.square(1 - predict_bbox_confidences)
    # Loss 함수 4번 (with lambda_noobj 0.5)
    no_object_loss = 0.5 * (1 - box_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)

    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    # Loss 함수 5번
    class_loss = responsible_mask * K.square(label_class - predict_class)

    # Loss 함수 5번 총합
    class_loss = K.sum(class_loss)

    loss = box_loss + confidence_loss + class_loss

    tf.print("\n- confidence_loss:", confidence_loss, output_stream=sys.stdout)
    tf.print("- class_loss:", class_loss, output_stream=sys.stdout)
    tf.print("- box_loss:", box_loss, output_stream=sys.stdout)

    return loss


def DummyLoss(y_true, y_pred):
    return tf.reduce_sum(y_pred)