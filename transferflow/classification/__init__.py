
DEFAULT_SETTINGS = {
    'flip_left_right': False,
    'random_crop': 0,
    'random_scale': 0,
    'random_brightness': 0,
    'validation_batch_size': 100,
    'test_batch_size': 500,
    'train_batch_size': 100,
    'eval_step_interval': 10,
    'testing_percentage': 10,
    'validation_percentage': 10,
    'learning_rate': 0.01,
    'max_num_steps': 4000,
    # accepted_accuracy_delta is a percentage
    # 'good' model is not lower than accepted_accuracy_delta% from the
    #  best validation set accuracy
    'accepted_accuracy_delta' : 2,
    # accepted_time_without_improvement is a percentage
    # Percentage of max_num_steps that the training continues without encountering a better validation model
    # After that the training stops
    'accepted_time_without_improvement' : 15,
}
