# INFO까지의 로그 Suppress하기
import datetime
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from YoloLoss import Yolov2Loss, Yolov2Loss_v2, DummyLoss, ANCHOR_BOXES, cell_offset_table
from YoloModel import Yolov2Model
from DataGenerator import Yolov2Dataloader


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# result: 1 * 13 * 13 * 5 * 25
def display_result_image_v2(input_image, network_output, no_suppress=False, display_all=True):
    classes = network_output[..., :20]  # ? * 13 * 13 * 5 * 20
    bbox = network_output[..., 20:24]  # ? * 13 * 13 * 5 * 4
    confidence = network_output[..., 24]  # ? * 13 * 13 * 5

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

        input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
        input_image_draw = ImageDraw.Draw(input_image_pil)

        max_anchor_id_per_cell = np.argmax(confidence, axis=3)  # ? * 13 * 13
        max_anchor_mask = np.where(
            confidence
        )
        drawn = False

        for y in range(13):
            for x in range(13):
                for anchor_id in range(5):
                    class_id = class_score_bbox_max_class[batch][y][x][anchor_id]
                    class_score_bbox = class_score_bbox_max_score[batch][y][x][anchor_id]

                    if not no_suppress and anchor_id is not max_anchor_id_per_cell[batch][y][x]:
                        continue

                    if not no_suppress and class_score_bbox_max_score[batch][y][x][anchor_id] == 0:
                        continue

                    if not no_suppress and class_score_bbox < thresh2:
                        continue

                    drawn = True

                    print("batch, y, x, anchor_id: ", batch, y, x, anchor_id)
                    (t_x, t_y, t_w, t_h) = bbox[batch][y][x][anchor_id]

                    diff = (1 / 13 * 416)

                    x_c = sigmoid(t_x) + (x * diff)
                    y_c = sigmoid(t_y) + (y * diff)
                    w = ANCHOR_BOXES[2 * anchor_id] * np.exp(t_w)
                    h = ANCHOR_BOXES[2 * anchor_id + 1] * np.exp(t_h)

                    # x_c, y_c = [x_c + x, y_c + y]
                    # x_c, y_c = [x_c / 7, y_c / 7]

                    x_1 = (x_c - (w / 2))
                    y_1 = (y_c - (h / 2))
                    x_2 = (x_c + (w / 2))
                    y_2 = (y_c + (h / 2))

                    input_image_draw.rectangle([x_1, y_1, x_2, y_2], outline="red", width=3)
                    input_image_draw.text([x_1 + 5, y_1 + 5], text=str(class_score_bbox), fill='yellow')
                    input_image_draw.text([x_1 + 5, y_1 + 13], text=train_data.GetLabelName(class_id), fill='yellow')

        if drawn:
            input_image_pil.show()


MODEL_SAVE = True
MODE_TRAIN = False
INTERACTIVE_TRAIN = True
LOAD_WEIGHT = True

train_data = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=8, augmentation=True)
train_data_no_augmentation = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=32,
                                              augmentation=False)
valid_train_data = Yolov2Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
test_data = Yolov2Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

dev_2 = Yolov2Dataloader(file_name='manifest-2.txt', numClass=20, batch_size=2, augmentation=False)
dev_16 = Yolov2Dataloader(file_name='manifest-16.txt', numClass=20, batch_size=16, augmentation=False)

TARGET_TRAIN_DATA = dev_2

LOG_NAME = "v3-2items-500epochs-lr0.0001-decay0.00001"

CHECKPOINT_SAVE_DIR = "D:\\ModelCheckpoints\\2020-yolov2-impl\\"
LOAD_CHECKPOINT_FILENAME = CHECKPOINT_SAVE_DIR + "20200410-110304-weights.epoch200-loss0.97.hdf5"
CHECKPOINT_TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-")

GLOBAL_EPOCHS = 1000
SAVE_PERIOD_SAMPLES = len(TARGET_TRAIN_DATA.image_list) * 1000  # 20 epoch

'''
    Learning Rate에 대한 고찰
    - 다양한 Augmentation이 활성화되어 있을 시, 2e-5  (loss: 100 언저리까지 가능)
    - Augmentation 비활성화 시, 1e-4: loss 20 언저리까지 가능
    - 1e-5: 20 언저리까지 떨어진 이후
    - Augmentation 비활성화 시, 시작부터 5e-6: 23까지는 잘 떨어짐
'''
LEARNING_RATE = 3e-3  # ref: 1e-4
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
    update_freq=94,
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

            image, _ = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
            result = model.predict(image)
            display_result_image_v2(image, result, no_suppress=False, display_all=False)
            # display_result_image_v2(image, result, no_suppress=False, display_all=True)

            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                # validation_data=valid_train_data,
                shuffle=False,
                callbacks=[model_checkpoint, tensor_board],
                verbose=1
            )

            epoch_iteration += 1
    else:
        model.fit(
            TARGET_TRAIN_DATA,
            epochs=GLOBAL_EPOCHS,
            # validation_data=valid_train_data,
            shuffle=False,
            callbacks=[model_checkpoint, tensor_board],
            verbose=1
        )
else:
    import random

    data_iterations = 8
    for _ in range(data_iterations):
        image, label = TARGET_TRAIN_DATA.__getitem__(random.randrange(0, TARGET_TRAIN_DATA.__len__()))
        result = model.predict(image)
        display_result_image_v2(image, result, no_suppress=True)
