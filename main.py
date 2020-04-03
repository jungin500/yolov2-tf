# INFO까지의 로그 Suppress하기
import os
import os.path
import numpy as np
from PIL import Image, ImageDraw

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from YoloLoss import Yolov2Loss
from YoloModel import Yolov2Model
from DataGenerator import Yolov2Dataloader

# result: 1 * 7 * 7 * 30
def postprocess_non_nms_result_v2(input_image, network_output):
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
    for batch in range(batch_size):
        input_image_single = input_image[batch]
        input_image_pil = Image.fromarray((input_image_single * 255).astype(np.uint8), 'RGB')
        input_image_draw = ImageDraw.Draw(input_image_pil)

        for y in range(7):
            for x in range(7):
                first_bigger = class_score_bbox_1_max_score[batch][y][x] > class_score_bbox_2_max_score[batch][y][x]
                if (first_bigger and class_score_bbox_1_max_score[batch][y][x] == 0) and (
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

                if class_score_bbox < thresh2:
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


GLOBAL_EPOCHS = 500
SAVE_PERIOD_EPOCHS = 200 #  (sample num = 2)
CHECKPOINT_FILENAME = "yolov2-training.hdf5"
MODE_TRAIN = False
LOAD_WEIGHT = True

# LEARNING_RATE = 1e-2
# DECAY_RATE = 5e-5
thresh1 = 0.2
thresh2 = 0.2

'''
    ANCHOR_BOXES에는 값으로 들어간다.
    -> 계산시 각 값을 13으로 나누어 작업하면 됨.
'''

train_data = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=8, augmentation=True)
train_data_no_augmentation = Yolov2Dataloader(file_name='manifest-train.txt', numClass=20, batch_size=4,
                                              augmentation=False)
valid_train_data = Yolov2Dataloader(file_name='manifest-valid.txt', numClass=20, batch_size=2)
test_data = Yolov2Dataloader(file_name='manifest-test.txt', numClass=20, batch_size=4)

train_twoimg = Yolov2Dataloader(file_name='manifest-twoimg.txt', numClass=20, batch_size=2,
                                              augmentation=False)

TARGET_TRAIN_DATA = train_twoimg
model = Yolov2Model()
# optimizer = Adam(learning_rate=LEARNING_RATE, decay=DECAY_RATE)
# model.compile(optimizer=optimizer, loss=Yolov2Loss)
model.compile(optimizer='adam', loss=Yolov2Loss)

model.summary()

print("Save frequency is {} sample, batch_size={}.".format(SAVE_PERIOD_EPOCHS, TARGET_TRAIN_DATA.batch_size))

save_best_model = ModelCheckpoint(
    CHECKPOINT_FILENAME,
    save_best_only=True,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_freq=SAVE_PERIOD_EPOCHS
)

if LOAD_WEIGHT and os.path.isfile(CHECKPOINT_FILENAME):
    model.load_weights(CHECKPOINT_FILENAME)

if MODE_TRAIN:
    model.fit(
        TARGET_TRAIN_DATA,
        epochs=GLOBAL_EPOCHS,
        # validation_data=valid_train_data,
        shuffle=True,
        callbacks=[save_best_model],
        verbose=1
    )
else:
    import random

    data_iterations = 1
    result_set = []
    for _ in range(data_iterations):
        image, _ = test_data.__getitem__(random.randrange(0, test_data.__len__()))
        result = model.predict(image)
        # postprocess_non_nms_result(image, result)
        postprocess_non_nms_result_v2(image, result)

    print(result_set)
