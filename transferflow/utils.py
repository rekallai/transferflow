
import os
import json
import numpy as np
import cv2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import tensorflow as tf
import shutil
import time
from distutils.version import LooseVersion

import logging
logger = logging.getLogger("transferflow.utils")

TENSORFLOW_VERSION = LooseVersion(tf.__version__)

def transfer_model_meta(source_path, destination_path):
    shutil.copyfile(source_path + '/nnscaffold.json', destination_path + '/nnpackage.json')
    shutil.copyfile(source_path + '/labels.json', destination_path + '/labels.json')

def tf_concat(axis, values, **kwargs):
    if TENSORFLOW_VERSION >= LooseVersion('1.0'):
        return tf.concat(values, axis, **kwargs)
    else:
        return tf.concat(axis, values, **kwargs)

def draw_rectangles(orig_image, rects, min_confidence=0.1, color=(0, 0, 255)):
    image = np.copy(orig_image)
    for rect in rects:
        if rect.confidence > min_confidence:
            cv2.rectangle(image,
                (rect.cx-int(rect.width/2), rect.cy-int(rect.height/2)),
                (rect.cx+int(rect.width/2), rect.cy+int(rect.height/2)),
                color,
                2)
    return image

def get_tensors(sess):
    layers = []
    for op in sess.graph.get_operations():
        layers.append(op.name)
    return layers

def get_tensor_namespaces(sess):
    namespaces = []
    for op in sess.graph.get_operations():
        path = op.name.split('/')
        if len(path) > 1 and path[0] not in namespaces:
            namespaces.append(path[0])
    return namespaces


def prune_models(model_list, model_accuracy_list, accepted_accuracy_delta, absolute=True):
    validation_accuracy = [entry['validation'] for entry in model_accuracy_list]
    max_acc = max(validation_accuracy)

    # Use absolute when final max_acc is not known yet, e.g. during training
    #  Else it would not be fair; models would be pruned that would not have
    #  been pruned when this method would only have beeen called after all
    #  training steps were completed
    if absolute:
        accepted_acc = max_acc - (accepted_accuracy_delta / 100)
    else:
        accepted_acc = (1 - (accepted_accuracy_delta / 100)) * max_acc

    accepted_model_indices = filter(
        lambda i: validation_accuracy[i] >= accepted_acc,
        range(len(validation_accuracy)))

    logger.debug('Models pruned ... %d of %d models remaining ... ' % (len(accepted_model_indices), len(model_list)))

    return [model_list[i] for i in accepted_model_indices], \
           [model_accuracy_list[i] for i in accepted_model_indices]


def log_model_accuracy(logger, model_type, set_name, accuracy_data):
        logger.info('%s model (step=%d): %s accuracy = %.1f%%' % (model_type,
                                                                  set_name,
                                                                  accuracy_data['step'],
                                                                  accuracy_data[set_name] * 100))