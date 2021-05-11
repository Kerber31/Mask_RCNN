import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import matplotlib as mpl
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to save results
RESULTS_DIR = os.path.join(ROOT_DIR, "logs/borehole_results/")

############################################################
#  Configurations
############################################################


class BoreholeConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "borehole"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 3  # Background, fracture, vug

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100


class BoreholeInferenceConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "borehole"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 3  # Background, fracture, vug

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    USE_MINI_MASK = False


############################################################
#  Dataset
############################################################

class BoreholeDataset(utils.Dataset):

    def load_borehole(self, dataset_dir, subset=1.0):
        
        # Add classes.
        self.add_class("borehole", 1, "fracture")
        self.add_class("borehole", 2, "vug")

        data = None

        with open(dataset_dir, "r") as read_file:
            data = json.load(read_file)

        data = data[0:int(len(data)*subset)]

        # data format:

        # [{
        #     "image_id": int,
        #     "raw_image": image in the form of a 2D list [width, height],
        #     "masks": list [width, height, num_instances],
        #     "class_ids": list [num_instances]
        # }, ...]

        # Add images
        for a in data:
            image = np.array(a['raw_image'])*255
            height, width = image.shape[:2]

            if len(a['masks']) > 0 and len(a['class_ids']) > 0:
                masks = np.array(a['masks'])
                transposed_masks = []

                for mask in masks:
                    transposed_masks.append(mask.T)

                transposed_masks = np.array(transposed_masks).T

                self.add_image(
                    "borehole",
                    image_id=a['image_id'],
                    path=None,
                    width=width, height=height,
                    image=image,
                    masks=transposed_masks.astype(np.bool),
                    class_ids=np.array(a['class_ids']).astype(np.int32))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]

        masks = image_info['masks']
        class_ids = image_info['class_ids']

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return masks, class_ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        #get image
        image = self.image_info[image_id]["image"]

        #convert dims
        image = skimage.color.gray2rgb(image)

        return image

############################################################
# Training and Testing 
############################################################

def get_model(mode, model_dir, startpoint="last", inference_config=False):
    model = None
    config = BoreholeConfig()

    if inference_config:
        config = BoreholeInferenceConfig()

    if mode == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=model_dir)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                    model_dir=model_dir)

    if startpoint == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        weights_path = model.find_last()
        model.load_weights(weights_path, by_name=True)

    return model

def train(model, train_path, validation_path, subset=1.0):
    """Train the model."""
    config = BoreholeConfig()

    # Training dataset.
    dataset_train = BoreholeDataset()
    dataset_train.load_borehole(train_path, subset=subset)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BoreholeDataset()
    dataset_val.load_borehole(validation_path, subset=subset)
    dataset_val.prepare()
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                layers='heads')

def predict(model, dataset_dir, subset=1.0):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    prediction_dir = "prediction_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())
    prediction_dir = os.path.join(RESULTS_DIR, prediction_dir)
    os.makedirs(prediction_dir)

    # Read dataset
    dataset = BoreholeDataset()
    dataset.load_borehole(dataset_dir, subset=subset)
    dataset.prepare()

    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig("{}/{}.png".format(prediction_dir, dataset.image_info[image_id]["id"]))

def evaluate_model(model, dataset_dir, subset=1.0):
    APs = []
    precisions_dict = {}
    recall_dict     = {}

    # Read dataset
    dataset = BoreholeDataset()
    dataset.load_borehole(dataset_dir, subset=subset)
    dataset.prepare()

    for index, image_id in enumerate(dataset.image_ids):
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, model.config, image_id)
        # make prediction
        yhat = model.detect([image], verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, precisions, recalls, _ = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        precisions_dict[image_id] = np.mean(precisions)
        recall_dict[image_id] = np.mean(recalls)
        # store
        APs.append(AP)

    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP, precisions_dict, recall_dict