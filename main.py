# INFO까지의 로그 Suppress하기
import datetime
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from YoloLoss import Yolov2Loss, DummyLoss
from YoloModel import Yolov2Model
from DataGenerator import Yolov2Dataloader

# result: 1 * 7 * 7 * 30
def display_result_image_v2(input_image, network_output, no_suppress=False, display_all=True):
    classes = network_output[:, :, :, :20]
    confidence_1, confidence_2 = network_output[:, :, :, 20], network_output[:, :, :, 21]
    bbox_1, bbox_2 = network_output[:, :, :, 22:26], network_output[:, :, :, 26:30]

    class_score_bbox_1 = np.expand_dims(confidence_1, axis=3) * classes
    class_score_bbox_2 = np.expand_dims(confidence_2, axis=3) * classes

    # Set zero if score < thresh1 (0.2)
    class_score_bbox_1[np.where(class_score_bbox_1 < thresh1)] = 0.
    class_score_bbox_2[np.where(class_score_bbox_2 < thresh1)] = 0.

    # class_score 중에서 가장 높은 class id
    class_score_bbox_1_max_class = np.argmax(class_score_bbox_1, axis=3)
    class_score_bbox_2_max_class = np.argmax(class_score_bbox_2, axis=3)
    class_score_bbox_1_max_score = np.amax(class_score_bbox_1, axis=3)
    class_score_bbox_2_max_score = np.amax(class_score_bbox_2, axis=3)

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

        for y in range(7):
            for x in range(7):
                first_bigger = class_score_bbox_1_max_score[batch][y][x] > class_score_bbox_2_max_score[batch][y][x]
                if not no_suppress and (first_bigger and class_score_bbox_1_max_score[batch][y][x] == 0) and (
                        not first_bigger and class_score_bbox_2_max_score[batch][y][x] == 0):
                    continue

                class_id = None
                class_score_bbox = None
                bbox = None
                if first_bigger:
                    class_id = class_score_bbox_1_max_class[batch][y][x]
                    class_score_bbox = class_score_bbox_1_max_score[batch][y][x]
                    bbox = bbox_1[batch][y][x]
                else:
                    class_id = class_score_bbox_2_max_class[batch][y][x]
                    class_score_bbox = class_score_bbox_2_max_score[batch][y][x]
                    bbox = bbox_2[batch][y][x]

                if not no_suppress and class_score_bbox < thresh2:
                    continue

                (x_c, y_c, w, h) = bbox

                x_c, y_c = [x_c + x, y_c + y]
                x_c, y_c = [x_c / 7, y_c / 7]

                x_1 = (x_c - (w / 2)) * 448
                y_1 = (y_c - (h / 2)) * 448
                x_2 = (x_c + (w / 2)) * 448
                y_2 = (y_c + (h / 2)) * 448

                input_image_draw.rectangle([x_1, y_1, x_2, y_2], outline="red", width=3)
                input_image_draw.text([x_1 + 5, y_1 + 5], text=str(class_score_bbox), fill='yellow')
                input_image_draw.text([x_1 + 5, y_1 + 13], text=train_data.GetLabelName(class_id), fill='yellow')

        input_image_pil.show(title="Sample Image")

# result: 1 * 7 * 7 * 30
SAVE_AS_CHECKPOINT_FILENAME = None
CHECKPOINT_FILENAME = "yolov2.hdf5"

MODEL_SAVE = True
MODE_TRAIN = True
INTERACTIVE_TRAIN = False
LOAD_WEIGHT = False

train_data = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=8, augmentation=True)
train_data_no_augmentation = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=32, augmentation=False)
valid_train_data = Yolov2Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
test_data = Yolov2Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=2)

dev_2 = Yolov2Dataloader(file_name='manifest-2.txt', numClass=20, batch_size=8, augmentation=False)

TARGET_TRAIN_DATA = dev_2
# valid_train_data = TARGET_TRAIN_DATA

LOG_NAME = "2items"

GLOBAL_EPOCHS = 500
# SAVE_PERIOD_EPOCHS = 100
SAVE_PERIOD_SAMPLES = len(TARGET_TRAIN_DATA.image_list) * 1 # 1 epoch

'''
    Learning Rate에 대한 고찰
    - 다양한 Augmentation이 활성화되어 있을 시, 2e-5  (loss: 100 언저리까지 가능)
    - Augmentation 비활성화 시, 1e-4: loss 20 언저리까지 가능
    - 1e-5: 20 언저리까지 떨어진 이후
    - Augmentation 비활성화 시, 시작부터 5e-6: 23까지는 잘 떨어짐
'''
LEARNING_RATE = 1e-4
DECAY_RATE = 1e-5
thresh1 = 0.2
thresh2 = 0.2

model = Yolov2Model()
optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
model.compile(optimizer=optimizer, loss=Yolov2Loss)

# model.summary()

save_frequency_raw = SAVE_PERIOD_SAMPLES * 5
print("Save frequency is {} sample, batch_size={}.".format(save_frequency_raw, TARGET_TRAIN_DATA.batch_size))

save_best_model = ModelCheckpoint(
    SAVE_AS_CHECKPOINT_FILENAME if SAVE_AS_CHECKPOINT_FILENAME is not None else CHECKPOINT_FILENAME,
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    # save_freq=save_frequency
    save_freq=save_frequency_raw
)

if LOAD_WEIGHT:
    model.load_weights(CHECKPOINT_FILENAME)

if LOG_NAME is not None:
    log_dir = "logs\\" + LOG_NAME
else:
    log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

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
            display_result_image_v2(image, result, no_suppress=True, display_all=False)

            model.fit(
                TARGET_TRAIN_DATA,
                epochs=int(GLOBAL_EPOCHS / epoch_divide_by),
                # validation_data=valid_train_data,
                shuffle=False,
                callbacks=[save_best_model, tensor_board],
                verbose=1
            )


            epoch_iteration += 1
    else:
        model.fit(
            TARGET_TRAIN_DATA,
            epochs=GLOBAL_EPOCHS,
            # validation_data=valid_train_data,
            shuffle=False,
            callbacks=[save_best_model, tensor_board],
            verbose=1
        )
else:
    import random

    data_iterations = 8
    for _ in range(data_iterations):
        image, label = test_data.__getitem__(random.randrange(0, test_data.__len__()))
        result = model.predict(image)
        # postprocess_calculate_precision(result, label)
        display_result_image_v2(image, result, no_suppress=False)