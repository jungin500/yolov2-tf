from tensorflow.keras import utils
import math
import numpy as np

import imgaug as ia
import imgaug.augmenters as iaa
from PIL import Image


def test_augmented_items(image_aug, bbs_aug):
    bbs_aug = bbs_aug.remove_out_of_image()
    bbs_aug = bbs_aug.clip_out_of_image()

    Image.fromarray(bbs_aug.draw_on_image(np.array(image_aug)), 'RGB').show()
    pass


class Labeler():
    def __init__(self, names_filename):
        self.names_list = {}

        with open(names_filename) as f:
            idx = 0
            for line in f:
                self.names_list[idx] = line
                idx += 1

    def get_name(self, index):
        return self.names_list[index].replace("\n", "")


# Necessary directives
ANCHOR_BOXES = [1.19, 1.99, 2.79, 4.60, 4.54, 8.93, 8.06, 5.29, 10.33, 10.65]
ANCHOR_BOXES = np.array(ANCHOR_BOXES).reshape((-1, 2))


def get_best_iou_anchor_idx(width, height):
    height_ratios = (ANCHOR_BOXES / np.reshape(ANCHOR_BOXES[:, 0], (-1, 1)))[:, 1]
    current_heights = np.tile(height / width, np.shape(ANCHOR_BOXES)[0])

    height_differences = np.abs(height_ratios - current_heights)
    return np.argmin(height_differences)


class Yolov2Dataloader(utils.Sequence):

    DEFAULT_AUGMENTER = iaa.SomeOf(2, [
        iaa.Multiply((1.2, 1.5)),  # change brightness, doesn't affect BBs
        iaa.Affine(
                translate_px={"x": 3, "y": 10},
                scale=(1.2, 1.2)
        ),  # translate by 40/60px on x/y axis, and scale to 50-70%, affects BBs
        iaa.AdditiveGaussianNoise(scale=0.1 * 255),
        iaa.CoarseDropout(0.02, size_percent=0.15, per_channel=0.5),
        # iaa.Affine(rotate=45),
        iaa.Sharpen(alpha=0.5)
    ])

    def __init__(self, file_name, dim=(416, 416, 3), batch_size=1, numClass=1, augmentation=False, shuffle=True):
        self.image_list, self.label_list = self.GetDataList(file_name)
        self.dim = dim
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenter = self.DEFAULT_AUGMENTER if augmentation else False
        self.outSize = 5 + numClass
        self.labeler = Labeler('voc.names')
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_list) / self.batch_size)

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [self.image_list[k] for k in indexes]
        batch_y = [self.label_list[k] for k in indexes]

        X, Y = self.__data_generation(batch_x, batch_y)

        # 마지막에 [None]을 넣는 것은... 다른 버전에서 동작하지 않는다
        return np.asarray(X), np.asarray(Y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def GetDataList(self, folder_path: str):
        train_list = []
        lable_list = []

        f = open(folder_path, 'r')
        while True:
            line = f.readline()
            if not line: break
            train_list.append(line.replace("\n", ""))
            label_text = line.replace(".jpg", ".txt")
            # label_text = label_text.replace(
            #     'C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCdevkit\\VOC2007\\JPEGImages\\',
            #     "C:\\Users\\jungin500\\Desktop\\Study\\2020-yolov3-impl\\VOCyolo\\")
            label_text = label_text.replace("\n", "")
            lable_list.append(label_text)

        return train_list, lable_list

    def __convert_yololabel_to_iaabbs(self, yolo_raw_label):
        # raw_label = [bboxes, 5], np.array([center_x, center_y, w, h, c])
        return ia.BoundingBoxesOnImage([
            ia.BoundingBox(
                x1=yolo_raw_bbox[0],
                y1=yolo_raw_bbox[1],
                x2=yolo_raw_bbox[2],
                y2=yolo_raw_bbox[3],
                # label=class_list[int(yolo_bbox[0])] # Label을 id로 활용하자
                label=yolo_raw_bbox[4]
            ) for yolo_raw_bbox in yolo_raw_label
        ], shape=(self.dim[0], self.dim[1]))

    def __convert_iaabbs_to_yololabel(self, iaa_bbs_out):
        label = np.zeros((13, 13, 5, 25), dtype=np.float32)
        raw_label = []

        for bbox in iaa_bbs_out.bounding_boxes:
            center_x = bbox.center_x
            center_y = bbox.center_y
            width = bbox.width
            height = bbox.height
            class_id = int(float(bbox.label))  # Explicit

            anchor_idx = get_best_iou_anchor_idx(width, height)

            scale_factor = (1 / 13)

            grid_x_index = int((center_x / 416) // scale_factor)
            grid_y_index = int((center_y / 416) // scale_factor)
            grid_x_index, grid_y_index = \
                np.clip([grid_x_index, grid_y_index], a_min=0, a_max=6)

            if label[grid_y_index][grid_x_index][anchor_idx][class_id] == 0.:
                label[grid_y_index][grid_x_index][anchor_idx][class_id] = 1.
                label[grid_y_index][grid_x_index][anchor_idx][20:] = np.array([center_x, center_y, width, height, 1])

                raw_label.append(np.array([center_x, center_y, width, height, class_id]))

        return label, np.array(raw_label)

    def __data_generation(self, list_img_path, list_label_path):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *(13, 13, 5, 25)))

        # Generate data
        for i, path in enumerate(list_img_path):
            original_image = (np.array(Image.open(path).resize((self.dim[0], self.dim[1]))) / 255).astype(np.float32)

            # raw_label은 x_1, y_1, x_2, y_2, c를 가지고 있다.
            label, raw_label = self.GetLabel(list_label_path[i], original_image.shape[0], original_image.shape[1])
            if self.augmenter:
                iaa_bbs = self.__convert_yololabel_to_iaabbs(raw_label)
                augmented_image, augmented_label = self.augmenter(
                    image=(original_image * 255).astype(np.uint8),
                    bounding_boxes=iaa_bbs
                )
                # test_augmented_items(augmented_image, augmented_label)
                X[i,] = augmented_image / 255
                Y[i,], _ = \
                    self.__convert_iaabbs_to_yololabel(augmented_label.remove_out_of_image().clip_out_of_image())
            else:
                X[i,] = original_image
                Y[i,] = label

        return X, Y

    def GetLabel(self, label_path, img_h, img_w):
        f = open(label_path, 'r')
        label = np.zeros((13, 13, 5, 25), dtype=np.float32)
        raw_label = []
        size = 416
        while True:
            line = f.readline()
            if not line: break

            split_line = line.split()
            c, x, y, w, h = split_line

            x = float(x) * size
            y = float(y) * size
            w = float(w) * size
            h = float(h) * size
            c = int(c)

            anchor_idx = get_best_iou_anchor_idx(w, h)
            scale_factor = (1 / 13)

            # // : 몫
            grid_x_index = int((x / size) // scale_factor)
            grid_y_index = int((y / size) // scale_factor)

            # 레이블은 하나만 지정한다.
            # 같은 Cell에 두 개 이상의 레이블이 들어가게 되면,
            # 하나의 객체만 사용한다.
            if label[grid_y_index][grid_x_index][anchor_idx][c] == 0.:
                label[grid_y_index][grid_x_index][anchor_idx][c] = 1.
                label[grid_y_index][grid_x_index][anchor_idx][20:] = np.array([x, y, w, h, 1])

                raw_label.append(np.array([
                    x - w / 2,
                    y - h / 2,
                    x + w / 2,
                    y + h / 2,
                    c
                ]))
            else:
                # print("Skipping labeling ... two or more bbox in same cell")
                pass

        return label, np.array(raw_label)

    def GetLabelName(self, label_id):
        return self.labeler.get_name(label_id)
