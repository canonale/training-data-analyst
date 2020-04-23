import tensorflow as tf
import numpy as np


CSV_COLUMNS = [
    'fare_amount',
    'pickuplon',
    'pickuplat',
    'dropofflon',
    'dropofflat',
    'passengers',
    'key'
]
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label

        file_list = tf.gfile.Glob(filename)

        dataset = tf.data.TextLineDataset(file_list).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(batch_size=(10 * batch_size))
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn

def add_more_features(feats):
    return feats

feature_cols = add_more_features(INPUT_COLUMNS)

def serving_input_fn():
    feature_placeholders = {
        column.name: tf.placeholder(tf.float32, [None]) for column in INPUT_COLUMNS
    }
    features = feature_placeholders
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def train_and_evaluate(args):
    # TODO: Clear cache
    tf.summary.FileWriterCache.clear()
    # TODO: Estimator
    estimator = tf.estimator.DNNRegressor(
        hidden_units=args['hidden_units'],
        feature_columns=feature_cols,
        model_dir=args['model_dir']
    )
    # TODO: Trainner
    train_spec = tf.estimator.TrainSpec(
        input_fn=read_dataset(
                args['data_set'],
                mode=tf.estimator.ModeKeys.TRAIN,
                batch_size=args['batch_size']
            ),
        max_steps=args['max_steps']
    )
    # TODO: Export to use it
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    # TODO: Evaluate
    eval_spec = tf.estimator.EvalSpec(
        input_fn=read_dataset(
            args['eval_data_paths'],
            mode=tf.estimator.ModeKeys.EVAL,
            batch_size=10000
        ),
        steps=None,
        start_delay_secs=args['eval_delay_secs'],
        throttle_secs=args['throttle_secs'],
        exporters=exporter
    )
    # TODO: Train
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

