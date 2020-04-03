import tensorflow.keras.backend as K
import tensorflow as tf

import numpy as np

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max

'''
    pred_mins, pred_maxes: shape(? * 13 * 13 * 5 * 1 * 2)
    true_mins, true_maxes: shape(? * 13 * 13 * 1 * 1 * 2)
    
    returns [? * 13 * 13 * 5 * 1]
'''
def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)  # ? * 13 * 13 * 5 * 1 * 2
    intersect_maxes = K.minimum(pred_maxes, true_maxes)  # ? * 13 * 13 * 5 * 1 * 2
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # ? * 13 * 13 * 5 * 1 * 2
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # ? * 13 * 13 * 5 * 1

    pred_wh = pred_maxes - pred_mins  # ? * 13 * 13 * 5 * 1 * 2
    true_wh = true_maxes - true_mins  # ? * 13 * 13 * 1 * 1 * 2
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # ? * 13 * 13 * 5 * 1
    true_areas = true_wh[..., 0] * true_wh[..., 1]  # ? * 13 * 13 * 1 * 1

    union_areas = pred_areas + true_areas - intersect_areas  # ? * 13 * 13 * 5 * 1
    iou_scores = intersect_areas / union_areas  # ? * 13 * 13 * 5 * 1

    return iou_scores  # ? * 13 * 13 * 5 * 1


'''
    box_xy, box_wh -> [B * 13 * 13 * 5 * 1 * 2]
'''
ANCHOR_BOXES = [1.19, 1.99, 2.79, 4.60, 4.54, 8.93, 8.06, 5.29, 10.33, 10.65]
ANCHOR_BOXES_TF = tf.reshape(
    tf.convert_to_tensor(np.array(ANCHOR_BOXES, dtype=np.float32)),
    shape=(1, 1, 1, 5, 1, 2)
)


def get_bbox_anchor(box_wh):
    anchor_box = ANCHOR_BOXES_TF
    anchor_ratio = anchor_box[..., 0] / anchor_box[..., 1]  # width / height, 1 * 1 * 1 * 5 * 1
    box_ratio = box_wh[..., 0] / box_wh[..., 1]  # width / height, ? * 13 * 13 * 5 * 1
    ratio_diff = tf.abs(anchor_ratio - box_ratio)  # ? * 13 * 13 * 5 * 1
    ratio_diff = tf.reshape(ratio_diff, shape=(-1, 13, 13, 5))  # ? * 13 * 13 * 5
    ratio_diff = tf.cast(ratio_diff, tf.float32)

    bbox_anchor_nms_mask = tf.where(  # Find indices that is max between 5 anchors
        ratio_diff < tf.reduce_max(ratio_diff, axis=3, keepdims=True),
        0.,   # Set to zero if not a max
        1.,  # Set to ratio_diff original value
    )  # ? * 13 * 13 * 5
    return tf.reshape(bbox_anchor_nms_mask, shape=(-1, 13, 13, 5, 1, 1))  # 5 -> (5 * 1 * 1)


def get_anchor_mask():
    anchor_box = ANCHOR_BOXES_TF
    return tf.reshape(anchor_box, shape=(1, 1, 1, 5, 1, 2))

# feats -> [? * 13 * 13 * 5 * 1 * 4]
# YOLOv2 changed behaviour -> b_x, b_y, b_w, b_h!
def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last, result = 13 * 13

    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]]) # 늘어놓는 함수  tile -> 같은걸 N번 반복함
    # 결과 -> 0~12, 0~12, ...., 0~12

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1]) # tile을 [n, m] 쓰면 dims 2로 만들어줌
    # 결과 -> [0~12], [0~12], [0~12], ...

    conv_width_index = K.flatten(K.transpose(conv_width_index))
    # 결과 -> 0, 0, 0, 0, 0, 0, 0 (13개), 1, 1, 1, 1, 1, 1, 1 (13개), ...

    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    # 결과 -> [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]

    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 1, 2])
    # 결과 -> 1 * 13 * 13 에 있는 [1 * 2]의 conv index item이 만들어짐
    # 각각 [1 * 2]의 값은 [0, 0], [1, 0], [2, 0], ..., [11, 12], [12, 12]
    # 이런 식으로 이루어져 있음 -> Mask를 만들기 위한 과정
    # 결과 shape -> 1, 13, 13, 1, 2

    conv_index = K.cast(conv_index, K.dtype(feats))
    # 타입 맞추기
    # 마지막 box_xy, box_wh에서 덧셈/나눗셈 연산 위해 타입 맞추기

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 1, 2]), K.dtype(feats))
    # [13, 13]을 앞자리 [1, 1, 1]로 맞추기
    # [1, 2]로 맞추어 진행
    # 결과 shape -> 1, 1, 1, 1, 2

    '''
        conv_index는 결국 feats[..., :2] -> [1, 13, 13, 1, 1, 2] -> [x_center, y_center]를
        전체 Image의 Relative Coordinates -> 각 Cell의 Relative Coordinate로 바꾸는 과정
        
        각 연산의 결과는
        -> y_true[? * 13 * 13 * 1 * 1 * 2]
        -> y_pred[? * 13 * 13 * 5 * 1 * 2]
    '''
    box_xy = (feats[..., :2] + conv_index) / conv_dims
    '''
        YOLOv2
        - box_x, box_y = (conv_index / 13) * sigmoid(out_value)
    '''
    box_xy_yolov2 = (conv_index / 13) * tf.sigmoid(box_xy) * 416
    # [..., :2]의 결과 -> [1, 13, 13, 5, 1, 2] (본래 뒷자리 [1, 4]였는데, 앞에 2개를 사용)

    box_wh = feats[..., 2:4]
    '''
        YOLOv2
        - box_w, box_h = (current_anchor_value_wh * exp(out_value)
    '''
    anchor_value_mask = get_anchor_mask()  # [1 * 1 * 1 * 1 * 5 * 1 * 2]
    nms_anchor_mask = get_bbox_anchor(box_wh)  # [B * 13 * 13 * 5 * 1 * 1]
    box_wh = anchor_value_mask * nms_anchor_mask * tf.exp(box_wh) * 416  # [B * 13 * 13 * 5 * 1 * 2]
    # bbox_anchor_nms_mask = get_bbox_anchor(box_wh)
    # box_wh_yolov2 = bbox_anchor_nms_mask * box_wh * 416  # [B * 13 * 13 * 5 * 1 * 1] * [B * 13 * 13 * 5 * 1 * 2]
    # [..., 2:4]의 결과 -> [1, 13, 13, 5, 1, 2] (본래 뒷자리 [1, 4]였는데, 뒤에 2개를 사용)

    return box_xy_yolov2, box_wh, nms_anchor_mask


'''
    y_true.shape = [B, 13, 13, 5, 25]
    - Anchor box 여부 관계 O
    - [5](4번째) axis에서 해당하는 Anchor box에서만 값을 가짐 
    - [실제 box_x, box_y, box_w, box_h, confidence (box existance)] + [classes one-hot]
'''

'''
    y_pred.shape = [B, 13, 13, 5, 25]
    - Anchor box의 갯수(5개)
    - [box_x, box_y, box_w, box_h, confidence] + [classes]
    - 총 shpae 5 + 20 = 25
'''

import tensorflow as tf

def Yolov2Loss(y_true, y_pred):
    # label_class = y_true[..., :20]      # ? * 7 * 7 * 20
    # label_box = y_true[..., 20:24]      # ? * 7 * 7 * 4
    # responsible_mask = y_true[..., 24]  # ? * 7 * 7
    # responsible_mask = K.expand_dims(responsible_mask)  # ? * 7 * 7 * 1

    # V2
    label_class = y_true[..., :20]        # ? * 13 * 13 * 5 * 20
    responsible_mask = y_true[..., 20]    # ? * 13 * 13 * 5 * 1
    label_box = y_true[..., 21:25]        # ? * 13 * 13 * 5 * 4
    extended_responsible_mask = tf.expand_dims(responsible_mask, axis=4)  # ? * 13 * 13 * 5 * 1 * 1
    d_extended_responsible_mask = tf.expand_dims(extended_responsible_mask, axis=5)  # ? * 13 * 13 * 5 * 1 * 1 * 1

    # predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    # predict_bbox_confidences = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    # predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    # V2
    predict_class = y_pred[..., :20]  # ? * 13 * 13 * 5 * 20
    predict_bbox_confidences = y_pred[..., 20]  # ? * 13 * 13 * 5 (1개)
    predict_coord = y_pred[..., 21:25]  # ? * 13 * 13 * 5 * 4

    # _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])  # ? * 7 * 7 * 1 * 4 (4 -> 1 * 4)
    # _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])  # ? * 7 * 7 * 2 * 4 (8 -> 2 * 4)

    # V2
    _label_box = tf.reshape(label_box, [-1, 13, 13, 5, 1, 4])  # ? * 13 * 13 * 5 * 1 * 4 (4 -> 1 * 4)
    _predict_box = tf.reshape(predict_coord, [-1, 13, 13, 5, 1, 4])  # ? * 13 * 13 * 5 * 1 * 4 (4 -> 1 * 4)

    # label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    # label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    # label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    # label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    # V2
    label_xy, label_wh, _ = yolo_head(_label_box)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2

    # predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    # predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    # predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    # predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    # V2
    predict_xy, predict_wh, anchor_mask = yolo_head(_predict_box)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2
    # predict_xy = K.expand_dims(predict_xy, 5)  # ? * 13 * 13 * 5 * 1 * 1 * 2
    # predict_wh = K.expand_dims(predict_wh, 5)  # ? * 13 * 13 * 5 * 1 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 13 * 13 * 5 * 1 * 2, ? * 13 * 13 * 5 * 1 * 2

    # iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    # best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    # best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    # V2
    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 13 * 13 * 5 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 13 * 13 * 5
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 13 * 13 * 1

    # box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2
    # extended_box_mask = tf.expand_dims(box_mask, axis=4)  # ? * 7 * 7 * 2 * 1

    # V2
    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 13 * 13 * 5
    extended_box_mask = tf.expand_dims(box_mask, axis=4)  # ? * 13 * 13 * 5 * 1
    d_extended_box_mask = tf.expand_dims(extended_box_mask, axis=5)  # ? * 13 * 13 * 5 * 1 * 1

    '''
    Box Loss Part
    '''
    # Loss 함수 1번
    box_loss = 5 * (
            # anchor_mask *
            d_extended_box_mask *
            d_extended_responsible_mask
    ) * K.square((label_xy - predict_xy) / 416)

    # Loss 함수 2번
    box_loss += 5 * (
            # anchor_mask *
            d_extended_box_mask *
            d_extended_responsible_mask
    ) * K.square(
        K.sqrt(label_wh) -
        K.sqrt(predict_wh)
    ) / 416

    # 1번+2번 총합
    box_loss = K.sum(box_loss)

    '''
    Confidence Loss Part
    '''
    # Loss 함수 3번 (without lambda_noobj)
    object_loss = box_mask * responsible_mask * K.square(1 - predict_bbox_confidences)
    # Loss 함수 4번 (with lambda_noobj 0.5)
    no_object_loss = 0.5 * (1 - box_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)

    # 3번+4번 총합
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    '''
    Class Loss Part
    '''
    # Loss 함수 5번
    class_loss = K.square(label_class - predict_class) * K.square(label_class - predict_class)

    # Loss 함수 5번 총합
    class_loss = K.sum(class_loss)

    ''' 전체 Loss의 합 '''
    loss = box_loss + confidence_loss + class_loss

    return loss

'''
    y_true.shape = [7, 7, 25]
    0  ~ 19 (20) -> one-hot class
    20 ~ 23 (4)  -> [x, y, w, h]
    24      (1)  -> response???? responsible mask!
'''

'''
    y_pred.shape = [?, 7, 7, 30]
    0  ~ 19 (20) -> predicted class probability
    20 ~ 21 (2)  -> predicted trust values (CONFIDENCE!!!)
    22 ~ 29 (8)  -> predicted bounding boxes [x, y, w, h], [x, y, w, h]
'''
def Yolov1Loss(y_true, y_pred):
    label_class = y_true[..., :20]      # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]      # ? * 7 * 7 * 4
    responsible_mask = y_true[..., 24]  # ? * 7 * 7
    responsible_mask = K.expand_dims(responsible_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_bbox_confidences = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])  # ? * 7 * 7 * 1 * 4 (4 -> 1 * 4)
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])  # ? * 7 * 7 * 2 * 4 (8 -> 2 * 4)

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    # Loss 함수 4번 (with lambda_noobj 0.5)
    no_object_loss = 0.5 * (1 - box_mask * responsible_mask) * K.square(0 - predict_bbox_confidences)
    # Loss 함수 3번 (without lambda_noobj)
    object_loss = box_mask * responsible_mask * K.square(1 - predict_bbox_confidences)

    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    # Loss 함수 5번
    class_loss = responsible_mask * K.square(label_class - predict_class)

    # Loss 함수 5번 총합
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    responsible_mask = K.expand_dims(responsible_mask)

    # Loss 함수 1번
    box_loss = 5 * box_mask * responsible_mask * K.square((label_xy - predict_xy) / 416)

    # Loss 함수 2번
    box_loss += 5 * box_mask * responsible_mask * K.square(K.sqrt(label_wh) - K.sqrt(predict_wh)) / 416

    # 1번+2번 총합
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss