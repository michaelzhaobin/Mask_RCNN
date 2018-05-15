import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import PIL

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#%matplotlib inline
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "words"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels
    #RPN_ANCHOR_SCALES = (16, 52, 120, 252, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation
    # steps since the epoch is small
    VALIDATION_STEPS = 20

    LEARNING_RATE = 0.001

    MAX_GT_INSTANCES = 250
    DETECTION_MAX_INSTANCES = 250




config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class ShapesDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def __init__(self):
        utils.Dataset.__init__(self)
        self._data_path = "/home/file/Mask_RCNN/TianChi_data"
        self.real_image_path = os.path.join(self._data_path, 'image_9000')
        self.image_path = os.path.join(self._data_path, 'txt_9000')
        self._classes = ('__background__', 'words')
        self._class_to_ind = dict(zip(self._classes, range(2)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        self._real_image_index = self._load_real_image_set_index()
        self.num_images = len(self._image_index)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path, 'txt_9000') #
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        image_index = os.listdir(image_set_file)
        return image_index

    def _load_real_image_set_index(self):
        tem = []
        for txt_name in self._image_index:
            base_txt_name = os.path.splitext(txt_name)[0]
            real_image = base_txt_name + '.jpg'
            tem.append(real_image)
        return tem

    def _find_outer_ranctangle(self, coor_list):
        min_x = min(coor_list[0], coor_list[2], coor_list[4], coor_list[6])
        min_y = min(coor_list[1], coor_list[3], coor_list[5], coor_list[7])
        max_x = max(coor_list[0], coor_list[2], coor_list[4], coor_list[6])
        max_y = max(coor_list[1], coor_list[3], coor_list[5], coor_list[7])
        outer_ranctangle = [min_x, min_y, max_x, max_y]
        return outer_ranctangle

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("words", 1, "s")
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(self.num_images):

            txt_path = os.path.join(self.image_path, self._image_index[i])
            open_file = open(txt_path, "r")
            lines = open_file.readlines()
            num_objs = len(lines)

            tmp = np.zeros((num_objs, 8), dtype=np.uint16)

            real_image_path = os.path.join(self.real_image_path, self._real_image_index[i])
            width_height = PIL.Image.open(real_image_path).size
            width = width_height[0]
            height = width_height[1]

            shapes = []
            boxes = []
            for j in range(num_objs):
                line_split = lines[j].split(',')
                for k in range(8):
                    line_split[k] = float(line_split[k])

                if line_split[0] < 0:
                    line_split[0] = 0
                if line_split[1] < 0:
                    line_split[1] = 0
                if line_split[2] < 0:
                    line_split[2] = 0
                if line_split[3] < 0:
                    line_split[3] = 0
                if line_split[4] < 0:
                    line_split[4] = 0
                if line_split[5] < 0:
                    line_split[5] = 0
                if line_split[6] < 0:
                    line_split[6] = 0
                if line_split[7] < 0:
                    line_split[7] = 0

                if line_split[0] > width:
                    line_split[0] = width
                if line_split[1] > height:
                    line_split[1] = height
                if line_split[2] > width:
                    line_split[2] = width
                if line_split[3] > height:
                    line_split[3] = height
                if line_split[4] > width:
                    line_split[4] = width
                if line_split[5] > height:
                    line_split[5] = height
                if line_split[6] > width:
                    line_split[6] = width
                if line_split[7] > height:
                    line_split[7] = height

                """
                tmp[j, :] = [float(line_split[0]), float(line_split[1]), float(line_split[2]), float(line_split[3]), float(line_split[4]), float(line_split[5]), float(line_split[6]), float(line_split[7])]
                gt_widths = math.sqrt((tmp[j, 4] - tmp[j, 2]) ** 2 + (tmp[j, 5] - tmp[j, 3]) ** 2)
                gt_heights = math.sqrt((tmp[j, 3] - tmp[j, 1]) ** 2 + (tmp[j, 2] - tmp[j, 0]) ** 2)
                # if gt_widths == 0:
                #    print(tmp[i, :])
                #    print(filename)
                # if gt_heights == 0:
                #    print(tmp[i, :])
                #    print(filename)
                # assert gt_widths > 0
                # assert gt_heights > 0
                if gt_widths == 0 or gt_heights == 0:
                    continue

                # print(line_split)
                # print("===============")]
                
                for k in [0, 1, 2, 3, 4, 5, 6, 7]:
                    assert line_split[k] >= 0
                for l in [0, 2, 4, 6]:
                    assert line_split[l] <= width
                for n in [1, 3, 5, 7]:
                    assert line_split[n] <= height
                """

                outer_ranctangle = self._find_outer_ranctangle(line_split[:8])
                x_min = float(outer_ranctangle[0])
                y_min = float(outer_ranctangle[1])
                x_max = float(outer_ranctangle[2])
                y_max = float(outer_ranctangle[3])
                real_width = x_max - x_min
                real_height = y_max - y_min

                shape = "s"
                # Color
                color = 'None'
                # Center x, y
                y = x_min + 0.5*real_width
                x = y_min + 0.5*real_height

                # Size
                dims = (x, y, real_height, real_width)
                real_points = (line_split[0], line_split[1], line_split[2], line_split[3], line_split[4],
                               line_split[5], line_split[6], line_split[7])


                # ["square", "circle", "triangle"]随机选择一个
                # [_, _, _] 三维颜色
                # （x, y, s）
                shapes.append((shape, color, dims, real_points))

            self.add_image("words", image_id=i, path=real_image_path,
                           width=width, height=height,
                           bg_color=None, shapes=shapes)
        """
        self.image_info: len: num_images
        self.image_info[i] = {
            "id": i,
            "source": "words",
            "path": real_image_path,
            "width": width, 
            "height": height,
            "bg_color": None, 
            "shapes": [(shape('kind_square'), color('None'), dims((x, y, real_height, real_width)), real_points(2*4)), ....]
            len(shapes): num_objs
        }
        """


    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        image = cv2.imread(self.image_info[image_id]["path"])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "words":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims, real_points) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, real_points, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, real_points, color):
        """Draws a shape from the given specs."""
        # Get the center x, y and the size s
        x1, y1, x2, y2, x3, y3, x4, y4 = real_points
        if shape == "s":

            #pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            #pts = pts.reshape((-1, 1, 2))
            #cv2.fillPoly(image, pts, 1)

            pts = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
            cv2.fillPoly(image, pts, color)

            #pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            #pts = pts.reshape((-1, 1, 2))
            #cv2.polylines(image, [pts], True, color, 5)

            #cv2.rectangle(image, (int(x1), int(y1)), (int(x3), int(y3)), 1, -1)
        return image

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()









# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 2)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    print(dataset_train._real_image_index[image_id])
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


model_path = "/home/file/Mask_RCNN/mask_rcnn_words_0010.h5"
model.load_weights(model_path, by_name=True)



"""
# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
"""

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE/10/10,
            epochs=20,
            layers='all')

"""






class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]
model_path = "/home/file/Mask_RCNN/mask_rcnn_words_0012.h5"

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
for _ in range(10):
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train.class_names, figsize=(8, 8))
"""
"""
    print(original_image)
    print(original_image.shape)
    original_image = cv2.imread("/home/file/Mask_RCNN/2.jpg")
    original_image, window, scale, padding, crop = utils.resize_image(
        original_image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    print(original_image)
    print(original_image.shape)
"""
"""
    print("==========")
    results = model.detect([original_image], verbose=1)
    print("==========")
    r = results[0]
    #visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                            dataset_val.class_names, r['scores'], ax=get_ax())
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], figsize=(8, 8))
    print("==========")



"""
