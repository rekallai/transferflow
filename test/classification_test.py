
import os
import sys
import unittest
from PIL import Image
import imghdr

test_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(test_dir + '/../')

from transferflow.classification.trainer import Trainer
from transferflow.classification.runner import Runner
from transferflow.classification.tfslim_runner import SlimRunner
from transferflow.classification.tfslim_trainer import SlimTrainer
from transferflow.utils import *
from nnpack.models import validate_model
from nnpack.scaffolds import clear_scaffold_cache

import logging
logger = logging.getLogger("transferflow")
logger.setLevel(logging.DEBUG)

if not os.path.isdir(test_dir + '/fixtures/tmp'):
    os.mkdir(test_dir + '/fixtures/tmp')

base_models_dir = test_dir + '/../models'

def _ensure_jpeg_path(image_path):
    image_type = imghdr.what(image_path)
    if image_type != 'jpeg':
        jpg_image_path = image_path + '.jpg'
        try:
            im = Image.open(image_path)
            im.save(jpg_image_path)
            return jpg_image_path
        except:
            logging.warn('Warning, could not convert to JPEG ({}), removing {}'.format(sys.exc_info()[0], image_path))
            os.remove(jpg_image_path)
    return image_path

class ClassificationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_1_run_base_model(self):
        runner = Runner(base_models_dir + '/inception_v3', softmax_layer='softmax:0')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['name'], 'boathouse')

    def test_2_train_scene_type(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test'
        base_model_path = base_models_dir + '/inception_v3'
        trainer = Trainer(base_model_path, scaffold_dir, max_num_steps=100)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.80, True)
        validate_model(output_model_path)

    def test_3_run_scene_type(self):
        runner = Runner(test_dir + '/fixtures/tmp/scene_type_test')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)
        self.assertEqual(labels[0]['name'], 'Outdoor')

    def test_4_run_inception_resnet_v2_base_model(self):
        runner = SlimRunner(base_models_dir + '/inception_resnet_v2', 'inception_resnet_v2')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['name'], 'boathouse')

    def test_5_train_scene_type_resnet_v2(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/scene_type'
        output_model_path = test_dir + '/fixtures/tmp/scene_type_test_resnetv2'
        base_model_path = base_models_dir + '/inception_resnet_v2'
        trainer = SlimTrainer(base_model_path, scaffold_dir, num_steps=20, batch_size=32)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['final_loss'] <= 0.80, True)
        validate_model(output_model_path)

    def test_6_run_scene_type_resnet_v2(self):
        runner = SlimRunner(test_dir + '/fixtures/tmp/scene_type_test_resnetv2',
                            'inception_resnet_v2')
        labels = runner.run(test_dir + '/fixtures/images/lake.jpg')
        self.assertEqual(labels[0]['node_id'], 1)
        self.assertEqual(labels[0]['name'], 'Outdoor')

    def test_7_train_coffee_roasts(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/coffee_roasts'
        output_model_path = test_dir + '/fixtures/tmp/coffee_roasts_test'
        base_model_path = base_models_dir + '/inception_v3'
        trainer = Trainer(base_model_path, scaffold_dir, num_steps=100)
        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        benchmark_info = trainer.train(output_model_path)
        self.assertEqual(benchmark_info['test_accuracy'] >= 0.75, True)
        validate_model(output_model_path)

    def test_8_train_and_validate_handguns(self):
        scaffold_dir = test_dir + '/fixtures/scaffolds/handguns'
        output_model_path = test_dir + '/fixtures/tmp/handguns'
        base_model_path = base_models_dir + '/inception_v3'
        # trainer = Trainer(base_model_path,
        #                   scaffold_dir,
        #                   eval_step_interval=10,
        #                   max_num_steps=2000,
        #                   learning_rate=0.01,
        #                   validation_percentage=20,
        #                   testing_percentage=20)

        trainer = Trainer(base_model_path,
                          scaffold_dir,
                          eval_step_interval=1,
                          max_num_steps=1200,
                          flip_left_right=True,
                          random_crop=20,
                          random_scale=20,
                          random_brightness=50,
                          learning_rate=0.01,
                          validation_percentage=30,
                          testing_percentage=20,
                          accepted_time_without_improvement=33)

        clear_scaffold_cache(scaffold_dir)
        trainer.prepare()
        models = trainer.train(output_model_path)

        expected_results = {
            '18.png': 'None',
            '350.png': 'Handgun',
            '357.png': 'Handgun',
            '384.png': 'Handgun',
            '410.png': 'Handgun',
            '424.png': 'Handgun',
            '456.png': 'None',
            '538.png': 'None',
            '554.png': 'None'
        }
        validation_images_dir = test_dir + '/fixtures/images/handguns'
        runner = Runner(output_model_path + '-earliest-good')
        print('')

        external_test_acc = 0.0
        num_images = 0.0
        for path in expected_results:
            num_images += 1

            labels = runner.run(_ensure_jpeg_path(validation_images_dir + '/' + path))
            passed = False
            if labels[0]['name'] == expected_results[path]:
                passed = True
            summary = ', '.join(map(lambda x: '{}={}'.format(x['name'], x['score']), labels))
            if passed:
                print('{}: GOOD (expected {}, was {})'.format(path, expected_results[path], summary))
                external_test_acc += 1
            else:
                print('{}: BAD (expected {}, was {})'.format(path, expected_results[path], summary))

        external_test_acc = external_test_acc/num_images

        print('EXTERNAL Test Accuracy: {}'.format(external_test_acc))
        #self.assertEqual(external_test_acc >= 0.8, True)


if __name__ == "__main__":
    unittest.main()
