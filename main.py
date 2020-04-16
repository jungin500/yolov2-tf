# INFO까지의 로그 Suppress하기
import datetime
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from YoloLoss import Yolov2Loss, ANCHOR_BOXES
from YoloModel import Yolov2Model
from DataGenerator import Yolov2Dataloader

import tensorflow as tf

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def parse_gtlabel(gt_label, data_generator):
    '''
    gt_label -> [13 * 13 * 5 * 25]
    - 13*13의 cell 중에 Confidence가 0이 아닌 값이 있는 곳(confidence mask가 1인 곳)만 보면 됨
    - 마찬가지로 5개의 Anchor 중 Confidence가 0이 아닌 값이 있는 곳만 보면 됨
    - Class ID는 25개 나열 시 앞 20개, 맨 뒤 5개(x, y, w, h, c=1)는 제외한다.
      이 값들은 x_c, y_c, w, h로 parse한다.
    '''

    confidence_mask = gt_label[:, :, :, 24]  # 13 * 13 * 5
    anchor_mask = np.argmax(confidence_mask, axis=2)  # 13 * 13. 5개의 anchor 중엔 무조건 1개만 있기 때문이다.

    parsed_objects = []

    for y in range(13):
        for x in range(13):

            # 값이 0이면 skip이다...
            anchor_idx = anchor_mask[y, x]

            if confidence_mask[y, x, anchor_idx] != 1.0:
                continue

            # 값이 있다면 해당 anchor는 cell의 5개 anchor중 유일하게 값을 가지며,
            # 해당 cell은 해당 object에 responsible하다.

            x_center = gt_label[y][x][anchor_idx][20]
            y_center = gt_label[y][x][anchor_idx][21]
            width = gt_label[y][x][anchor_idx][22]
            height = gt_label[y][x][anchor_idx][23]

            cell_x1 = x_center - (width / 2)
            cell_y1 = y_center - (height / 2)
            cell_x2 = x_center + (width / 2)
            cell_y2 = y_center + (height / 2)

            object_bbox = {
                'cell_x1': cell_x1,
                'cell_y1': cell_y1,
                'cell_x2': cell_x2,
                'cell_y2': cell_y2,
                'cell_id_x': x,
                'cell_id_y': y,
                'class_id': np.argmax(gt_label[y][x][anchor_idx][:20], axis=0),
                'class_name': data_generator.GetLabelName(np.argmax(gt_label[y][x][anchor_idx][:20], axis=0)),
                'anchor_idx': anchor_idx
            }

            parsed_objects.append(object_bbox)

    return parsed_objects


def check_if_responsible_cell_anchor(x, y, anchor_id, parsed_gt):
    for parsed_object in parsed_gt:
        label_anchor_id = parsed_object['anchor_idx']
        cell_x = parsed_object['cell_id_x']
        cell_y = parsed_object['cell_id_y']

        if x == cell_x and y == cell_y and anchor_id == label_anchor_id:
            return True
    return False


# result: 1 * 13 * 13 * 5 * 25
def display_result_image_v2(input_image, network_output, label, data_generator, no_suppress=False, display_all=True, display_by_anchors=False):
    classes = network_output[..., :20]  # ? * 13 * 13 * 5 * 20
    bbox = network_output[..., 20:24]  # ? * 13 * 13 * 5 * 4
    confidence = tf.sigmoid(network_output[..., 24])  # ? * 13 * 13 * 5

    class_score_bbox = np.expand_dims(confidence, axis=4) * classes  # ? * 13 * 13 * 5 * 20

    # Set zero if core < thresh1 (0.2)
    class_score_bbox[np.where(class_score_bbox < thresh1)] = 0.

    # class_score 중에서 가장 높은 class id
    class_score_bbox_max_class = np.argmax(class_score_bbox, axis=4)
    class_score_bbox_max_score = np.amax(class_score_bbox, axis=4)

    batch_size = np.shape(input_image)[0]

    if not display_all:
        display_range = list(range(batch_size))
        random.shuffle(display_range)
        display_range = display_range[:4]
    else:
        display_range = range(batch_size)

    for batch in display_range:
        input_image_single = input_image[batch]

        if display_by_anchors:
            input_images = [
                Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
                for _ in range(5)
            ]
        else:
            input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')

        max_anchor_id_per_cell = np.argmax(confidence, axis=3)  # ? * 13 * 13

        # GT를 그린다.
        # 현재는 display_by_anchors 상태에서만 가능하다.
        parsed_gt = parse_gtlabel(label[batch], data_generator)
        for parsed_object in parsed_gt:
            anchor_id = parsed_object['anchor_idx']
            x_1 = parsed_object['cell_x1']
            y_1 = parsed_object['cell_y1']
            x_2 = parsed_object['cell_x2']
            y_2 = parsed_object['cell_y2']
            class_name = parsed_object['class_name']

            outline_mask = Image.new('RGBA', (416, 416))
            outline_mask_draw = ImageDraw.Draw(outline_mask)
            outline_mask_draw.rectangle([x_1, y_1, x_2, y_2], outline=(0, 0, 255, 255), width=3)  # Blue
            outline_mask_draw.text([x_1 + 5, y_1 + 5], text='GT-' + class_name, fill='blue')

            if display_by_anchors:
                input_images[anchor_id].paste(outline_mask, mask=outline_mask)
            else:
                input_image_pil.paste(outline_mask, mask=outline_mask)


        # 모델의 Inference 결과를 그린다.
        for y in range(13):
            for x in range(13):
                if no_suppress:
                    anchor_range = range(5)
                else:
                    anchor_range = [ max_anchor_id_per_cell[batch][y][x] ]  # 하나만 넣기...!

                for anchor_id in anchor_range:
                    class_id = class_score_bbox_max_class[batch][y][x][anchor_id]
                    class_score_bbox = class_score_bbox_max_score[batch][y][x][anchor_id]

                    if not no_suppress and class_score_bbox_max_score[batch][y][x][anchor_id] == 0:
                        continue

                    if not no_suppress and class_score_bbox < thresh2:
                        continue

                    # Confidence를 그린다.
                    confidence_value = int(confidence[batch][y][x][anchor_id] * 100) / 100

                    (t_x, t_y, t_w, t_h) = bbox[batch][y][x][anchor_id]
                    diff = (1 / 13 * 416)

                    x_c = (sigmoid(t_x) * 32) + (x * diff)
                    y_c = (sigmoid(t_y) * 32) + (y * diff)
                    w = ANCHOR_BOXES[2 * anchor_id] * np.exp(t_w)
                    h = ANCHOR_BOXES[2 * anchor_id + 1] * np.exp(t_h)

                    x_1 = (x_c - (w / 2))
                    y_1 = (y_c - (h / 2))
                    x_2 = (x_c + (w / 2))
                    y_2 = (y_c + (h / 2))

                    # class_score_bbox 값에 따라 투명도를 달리 한다.
                    outline_mask = Image.new('RGBA', (416, 416))
                    outline_mask_draw = ImageDraw.Draw(outline_mask)

                    # supress_text = '' if no_suppress else '[' + str(class_score_bbox) + ']'
                    supress_text = str(int(class_score_bbox * 100) / 100) + '\n'

                    # Red
                    if check_if_responsible_cell_anchor(x, y, anchor_id, parsed_gt):
                        outline_mask_draw.rectangle([x_1, y_1, x_2, y_2],
                                                    outline=(255, 0, 0, 255), width=3)
                        outline_mask_draw.text([x_1 + 5, y_1 + 5], text=supress_text + train_data.GetLabelName(class_id),
                                               fill='red')
                        # print("y={}, x={}, anchor={}, confidence={}, class_score_bbox={}".format(y, x, anchor_id, confidence_value, class_score_bbox))
                    else:
                        outline_mask_draw.rectangle([x_1, y_1, x_2, y_2],
                                                    outline=(255, 0, 0, int(class_score_bbox * 255)), width=1)
                        outline_mask_draw.text([x_1 + 5, y_1 + 5], text=supress_text + train_data.GetLabelName(class_id), fill='yellow')

                    # outline_mask_draw.text([x * 32, y * 32], text=str(confidence_value), fill='white')


                    if display_by_anchors:
                        input_images[anchor_id].paste(outline_mask, mask=outline_mask)
                    else:
                        input_image_pil.paste(outline_mask, mask=outline_mask)

        if display_by_anchors:
            for image in input_images:
                image.show()
        else:
            input_image_pil.show()


MODEL_SAVE = True
MODE_TRAIN = True
INTERACTIVE_TRAIN = False
LOAD_WEIGHT = False

train_data = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=32, augmentation=True)
train_data_no_augmentation = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=32,
                                              augmentation=False)
valid_train_data = Yolov2Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
test_data = Yolov2Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

dev_1 = Yolov2Dataloader(file_name='manifest-1.txt', numClass=20, batch_size=1, augmentation=False)
dev_2 = Yolov2Dataloader(file_name='manifest-2.txt', numClass=20, batch_size=2, augmentation=False)
dev_16 = Yolov2Dataloader(file_name='manifest-16.txt', numClass=20, batch_size=16, augmentation=False)

TARGET_TRAIN_DATA = train_data_no_augmentation

LOG_NAME = "all-items-augm-500epoches-lr0.005-decay0.00001"

CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov2-impl\\"
LOAD_CHECKPOINT_FILENAME = CHECKPOINT_SAVE_DIR + "20200414-143027-weights.epoch1000-loss1.94.hdf5"
CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

GLOBAL_EPOCHS = 500
SAVE_PERIOD_SAMPLES = len(TARGET_TRAIN_DATA.image_list) * 5  # 5 epoch

'''
    Learning Rate에 대한 고찰
    - 다양한 Augmentation이 활성화되어 있을 시, 2e-5  (loss: 100 언저리까지 가능)
    - Augmentation 비활성화 시, 1e-4: loss 20 언저리까지 가능
    - 1e-5: 20 언저리까지 떨어진 이후
    - Augmentation 비활성화 시, 시작부터 5e-6: 23까지는 잘 떨어짐
'''
LEARNING_RATE = 5e-4  # ref: 3e-4 on BN
DECAY_RATE = 1e-5
thresh1 = 0.2
thresh2 = 0.2

model = Yolov2Model()
optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
model.compile(optimizer=optimizer, loss=Yolov2Loss)

# model.summary()

save_frequency_raw = SAVE_PERIOD_SAMPLES
print("Save frequency is {} sample, batch_size={}.".format(save_frequency_raw, TARGET_TRAIN_DATA.batch_size))

if LOAD_WEIGHT and (LOAD_CHECKPOINT_FILENAME is not None):
    model.load_weights(LOAD_CHECKPOINT_FILENAME)

if LOG_NAME is not None:
    log_dir = "logs\\" + LOG_NAME + datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")
else:
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

model_checkpoint = ModelCheckpoint(
    CHECKPOINT_SAVE_DIR + CHECKPOINT_TIMESTAMP + 'weights.epoch{epoch:02d}-loss{loss:.2f}.hdf5',
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    # save_freq=save_frequency
    save_freq=save_frequency_raw
)

tensor_board = TensorBoard(
    log_dir=log_dir,
    write_graph=True,
    update_freq=5,
    profile_batch=0
)

early_stopping = EarlyStopping(
    monitor='loss',
    patience=10,
    baseline=5e-1
)

if MODE_TRAIN:
    if INTERACTIVE_TRAIN:
        import random

        epoch_divide_by = 5
        epoch_iteration = 0
        while epoch_iteration * (GLOBAL_EPOCHS / epoch_divide_by) < GLOBAL_EPOCHS:
            # Train <GLOBAL_EPOCHS / epoch_divide_by> epoches

            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                validation_data=valid_train_data,
                shuffle=True,
                callbacks=[model_checkpoint, tensor_board],
                verbose=1
            )

            image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
            result = model.predict(image)
            display_result_image_v2(image, result, label, TARGET_TRAIN_DATA, no_suppress=True, display_all=True, display_by_anchors=True)

            epoch_iteration += 1
    else:
        model.fit(
            TARGET_TRAIN_DATA,
            epochs=GLOBAL_EPOCHS,
            validation_data=valid_train_data,
            shuffle=True,
            callbacks=[model_checkpoint, tensor_board],
            verbose=1
        )
else:
    import random

    data_iterations = 8
    for _ in range(data_iterations):
        image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
        result = model.predict(image)
        display_result_image_v2(image, result, label, TARGET_TRAIN_DATA, no_suppress=False, display_all=False, display_by_anchors=False)
