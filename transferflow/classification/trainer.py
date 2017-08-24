
import os
import json
from datetime import datetime

import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from inception import *
from transferflow.utils import transfer_model_meta
from transferflow.utils import create_image_lists
from transferflow.utils import prune_models
from transferflow.utils import log_model_accuracy
from nnpack.models import create_empty_model, save_model_benchmark_info
from nnpack import load_labels
from . import DEFAULT_SETTINGS

import logging
logger = logging.getLogger("transferflow.classification")


class Trainer(object):
    def __init__(self, base_model_path, scaffold_path, **kwargs):
        self.base_model_path = base_model_path
        self.scaffold_path = scaffold_path
        self.settings = DEFAULT_SETTINGS
        for key in kwargs:
            self.settings[key] = kwargs[key]
        if not self.settings.has_key('base_graph_path'):
            self.settings['base_graph_path'] = base_model_path + '/state/model.pb'
        self.labels = load_labels(scaffold_path)

    def prepare(self):
        settings = self.settings
        sess = tf.Session()
        self.bottleneck_tensor, self.jpeg_data_tensor, self.resized_image_tensor = load_base_graph(sess, settings['base_graph_path'])

        image_dir = self.scaffold_path + '/images'
        if not os.path.isdir(self.scaffold_path + '/cache'):
            os.mkdir(self.scaffold_path + '/cache')
        bottleneck_dir = self.scaffold_path + '/cache/bottlenecks'

        self.image_lists = create_image_lists(image_dir, settings['testing_percentage'], settings['validation_percentage'])
        for label in self.image_lists:
            category_lists = self.image_lists[label]
        class_count = len(self.image_lists.keys())

        if class_count == 0:
            raise Exception('No valid folders of images found at ' + image_dir)
        if class_count == 1:
            raise Exception('Only one valid folder of images found at ' + image_dir + ', multiple classes are needed for classification')

        # Link labels to new softmax layer
        self._add_softmax_ids_to_labels()

        self.do_distort_images = should_distort_images(settings['flip_left_right'], settings['random_crop'], settings['random_scale'], settings['random_brightness'])

        if self.do_distort_images:
            logger.debug('Distorting images')
            self.distorted_jpeg_data_tensor, self.distorted_image_tensor = add_input_distortions(settings['flip_left_right'], settings['random_crop'], settings['random_scale'], settings['random_brightness'])
        else:
            cache_bottlenecks(sess, self.image_lists, image_dir, bottleneck_dir, self.jpeg_data_tensor, self.bottleneck_tensor)

    def train(self, output_model_path):
        if self.settings['max_num_steps'] < 1:
            logger.error('max_num_steps should at least be 1')
            return

        settings = self.settings
        image_dir = self.scaffold_path + '/images'
        bottleneck_dir = self.scaffold_path + '/bottlenecks'

        sess = tf.Session()

        final_tensor_name = 'retrained_layer'
        (train_step, cross_entropy, bottleneck_input, ground_truth_input, final_tensor) = add_final_training_ops(len(self.image_lists.keys()), final_tensor_name, self.bottleneck_tensor, settings['learning_rate'])

        # Set up all our weights to their initial default values.
        init = tf.variables_initializer(tf.global_variables())
        sess.run(init)

        # Create the operations we need to evaluate the accuracy of our new layer.
        evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)
        # Run the training for as many cycles as requested on the command line.

        # Prepare test data, they will be calculated at every step when a new
        #  best validation model is found and at the final step
        test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
            sess, self.image_lists, settings['test_batch_size'], 'testing',
            bottleneck_dir, image_dir, self.jpeg_data_tensor,
            self.bottleneck_tensor)

        # Models that are not more than accepted_accuracy_delta% from best model validation accuracy
        #  earliest model will be first in the list
        good_models = list()
        # Contains validation accuracy of selected models
        good_models_accuracy = list()

        # Model with final best validation accuracy
        best_validation_model = None
        # Accuracy on all sets for this model
        # TODO : change name from _accuracy to metrics and add cross_entropy as prop
        best_validation_model_accuracy = {
            'step': -1,
            'training': -1,
            'validation': -1,
            'test': -1
        }
        # Training step at which the current best validation model occurred
        # This is used for the early stopping criterion
        training_step_best_validation_model = 0

        # Final earliest 'good' model that is not more than accepted_accuracy_delta% from
        #  best model validation accuracy
        earliest_good_model = None
        # accuracy on all sets for this model
        earliest_good_model_accuracy = None

        relative_wait_time = settings['accepted_time_without_improvement']
        absolute_wait_time = (relative_wait_time/100.0)*settings['max_num_steps']

        # model at last training step
        final_model = None
        final_model_accuracy = None
        for step in range(settings['max_num_steps']):
            # Get a catch of input bottleneck values, either calculated fresh every time
            # with distortions applied, or from the cache stored on disk.
            if self.do_distort_images:
                train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                    sess, self.image_lists, settings['train_batch_size'], 'training',
                    image_dir, self.distorted_jpeg_data_tensor,
                    self.distorted_image_tensor, self.resized_image_tensor, self.bottleneck_tensor)
            else:
                train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                    sess, self.image_lists, settings['train_batch_size'], 'training',
                    bottleneck_dir, image_dir, self.jpeg_data_tensor,
                    self.bottleneck_tensor)

            # Feed the bottlenecks and ground truth into the graph, and run a training
            # step.
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks,
                                ground_truth_input: train_ground_truth})
            # Every so often, print out how well the graph is training.
            is_last_step = (step + 1 == settings['max_num_steps'])
            has_waited_too_long = False
            if (step % settings['eval_step_interval']) == 0 or is_last_step:

                # Evaluate current model on training set
                train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy], feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
                logger.debug('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), step, train_accuracy * 100))
                logger.debug('%s: Step %d: Cross entropy = %f' % (datetime.now(), step, cross_entropy_value))

                # Evaluate current model on validation set
                validation_bottlenecks, validation_ground_truth = (
                    get_random_cached_bottlenecks(
                        sess, self.image_lists, settings['validation_batch_size'], 'validation',
                        bottleneck_dir, image_dir, self.jpeg_data_tensor,
                        self.bottleneck_tensor))
                validation_accuracy = sess.run(
                    evaluation_step,
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})
                logger.debug('%s: Step %d: Validation accuracy = %.1f%%' %
                      (datetime.now(), step, validation_accuracy * 100))

                test_accuracy = None

                if validation_accuracy > best_validation_model_accuracy['validation']:
                    logger.debug('New best validation model found ...')
                    training_step_best_validation_model = step

                    best_validation_model_accuracy['step'] = step
                    best_validation_model_accuracy['training'] = train_accuracy
                    best_validation_model_accuracy['validation'] = validation_accuracy

                    test_accuracy = sess.run(
                        evaluation_step,
                        feed_dict={bottleneck_input: test_bottlenecks,
                                   ground_truth_input: test_ground_truth})

                    best_validation_model_accuracy['test'] = test_accuracy

                    best_validation_model = graph_util.convert_variables_to_constants(
                        sess, sess.graph.as_graph_def(), [final_tensor_name])

                    good_models.append(best_validation_model)
                    good_models_accuracy.append(best_validation_model_accuracy.copy())

                    # Intermediate pruning to while training to constrain memory use
                    good_models, good_models_accuracy = prune_models(
                        good_models,
                        good_models_accuracy,
                        settings['accepted_accuracy_delta'],
                        True)

                has_waited_too_long = (step - training_step_best_validation_model + 1) > absolute_wait_time

                if is_last_step:
                    logger.debug('Last step ...')

                if has_waited_too_long:
                    logger.debug('No improvements in validation set accuracy found in %d steps, stopping early ...' % absolute_wait_time)

                if is_last_step or has_waited_too_long:
                    final_model = graph_util.convert_variables_to_constants(
                        sess,
                        sess.graph.as_graph_def(),
                        [final_tensor_name])

                    if test_accuracy is None:
                        test_accuracy = sess.run(
                            evaluation_step,
                            feed_dict={bottleneck_input: test_bottlenecks,
                                       ground_truth_input: test_ground_truth})

                    final_model_accuracy = {
                        'step': step,
                        'training': train_accuracy,
                        'validation': validation_accuracy,
                        'test': test_accuracy
                    }

            if has_waited_too_long:
                break

        # Final pruning
        good_models, good_models_accuracy = prune_models(
            good_models,
            good_models_accuracy,
            settings['accepted_accuracy_delta'],
            False)

        earliest_good_model = good_models[0]
        earliest_good_model_accuracy = good_models_accuracy[0]

        # TODO : MAKE MODEL TYPE NAMES STATICALLY AVAILABLE
        models = {
            'earliest-good' : {
                'model': earliest_good_model,
                'accuracy': earliest_good_model_accuracy
            },
            'best-validation': {
                'model': best_validation_model,
                'accuracy': best_validation_model_accuracy
            },
            'final': {
                'model': final_model,
                'accuracy': final_model_accuracy
            }
        }

        for model_type in models.keys():
            log_model_accuracy(logger, model_type, 'test', models[model_type]['accuracy'])

            accuracy = models[model_type]['accuracy']

            benchmark_info = {
                'validation_accuracy': float(accuracy['validation']),
                'train_accuracy': float(accuracy['training']),
                'test_accuracy': float(accuracy['test']),
                'training-step': int(accuracy['step'])
            }

            model_path_for_type = '%s-%s' % (output_model_path, model_type)
            create_empty_model(model_path_for_type)
            transfer_model_meta(self.scaffold_path, model_path_for_type)
            output_graph_path = model_path_for_type + '/state/model.pb'

            with gfile.FastGFile(output_graph_path, 'wb') as f:
              f.write(models[model_type]['model'].SerializeToString())
              f.close()

            # Persist labels with softmax IDs
            with open(model_path_for_type + '/labels.json', 'w') as f:
                json.dump({'labels': self.labels.values()}, f)

            # Store benchmark_info
            with open(model_path_for_type + '/benchmark.json', 'w+') as f:
                json.dump(benchmark_info, f)
                f.close()

        # Cleanup
        tf.reset_default_graph()
        sess.close()

        return models

    def _add_softmax_ids_to_labels(self):
        i = 0
        for label_id in self.image_lists:
            if label_id not in self.labels:
                raise Exception('Label with ID {} does not appear in labels.json, bad scaffold'.format(label_id))
            label = self.labels[label_id]
            label['node_id'] = i
            i += 1
