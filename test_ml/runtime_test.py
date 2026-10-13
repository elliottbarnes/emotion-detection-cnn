"""Modern-runtime compatibility using synthetic data and the trusted bundled model."""
import contextlib
import csv
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import DataManager
from model import EmotionDetectionModel


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        tf.keras.utils.set_random_seed(7)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'data').mkdir()
        (self.root / 'src').mkdir()
        self.previous_cwd = Path.cwd()
        os.chdir(self.root / 'src')
        self.addCleanup(os.chdir, self.previous_cwd)
        self.csv = self.root / 'data' / 'synthetic.csv'
        with self.csv.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['emotion', 'pixels', 'Usage'])
            for usage in ('Training', 'PrivateTest', 'PublicTest'):
                for emotion, pixel in ((3, 0), (6, 255)):
                    for _ in range(2):
                        writer.writerow([emotion, ' '.join([str(pixel)] * 2304), usage])

    def manager(self, from_scratch=True):
        return DataManager(str(self.csv), {3: 'Happy', 6: 'Neutral'},
                           batch_size=2, import_from_scratch=from_scratch)

    def test_preprocessing_augmentation_and_cache(self):
        manager = self.manager()
        data = manager.get_formatted_data()
        self.assertEqual(data['mapped_emotions'], {0: 'Happy', 1: 'Neutral'})
        for prefix in ('train', 'val', 'test'):
            self.assertEqual(data[prefix + '_data'].shape, (4, 48, 48, 1))
            np.testing.assert_array_equal(data[prefix + '_data'][0], 0)
            np.testing.assert_array_equal(data[prefix + '_data'][-1], 1)
            np.testing.assert_array_equal(data[prefix + '_labels'],
                                          [[1, 0], [1, 0], [0, 1], [0, 1]])
        augmented, labels = next(data['train_data_augmented'])
        self.assertEqual(augmented.shape, (2, 48, 48, 1))
        self.assertTrue(np.isfinite(augmented).all())
        self.assertTrue(((augmented >= 0) & (augmented <= 1)).all())
        self.assertEqual(labels.shape, (2, 2))
        cached = self.manager(False).get_formatted_data()
        np.testing.assert_array_equal(cached['train_data'], data['train_data'])

    def test_training_callbacks_and_checkpoint_reload(self):
        data = self.manager().get_formatted_data()
        checkpoint = self.root / 'synthetic.h5'
        training = EmotionDetectionModel(data, 2, model_path=str(checkpoint),
                                        batch_size=2, epochs=1)
        with contextlib.redirect_stdout(io.StringIO()):
            history = training.train_model()
        for metric in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
            self.assertEqual(len(history.history[metric]), 1)
            self.assertTrue(np.isfinite(history.history[metric]).all())
        restored = tf.keras.models.load_model(checkpoint, compile=False)
        self.assertEqual(restored.count_params(), 156898)
        predictions = restored(data['test_data'], training=False).numpy()
        self.assertEqual(predictions.shape, (4, 2))
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_allclose(predictions.sum(axis=1), 1, atol=1e-6)

    def test_historical_checkpoint_preserved_and_weights_match(self):
        checkpoint = ROOT / 'out' / 'emotion_model.h5'
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                         'd18b1e7259d2dc7dc012ec1fd19e743ce3f17268d4ac5c321df492ead26f5e0c')
        restored = tf.keras.models.load_model(checkpoint, compile=False)
        with contextlib.redirect_stdout(io.StringIO()):
            rebuilt = EmotionDetectionModel({}, 2).create_sequential_model()
        rebuilt.load_weights(checkpoint)
        images = np.random.default_rng(7).random((3, 48, 48, 1), dtype=np.float32)
        expected = restored(images, training=False).numpy()
        np.testing.assert_allclose(rebuilt(images, training=False).numpy(), expected,
                                   rtol=1e-5, atol=1e-6)
        self.assertEqual(restored.count_params(), 156898)
        self.assertTrue(np.isfinite(expected).all())
        np.testing.assert_allclose(expected.sum(axis=1), 1, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
