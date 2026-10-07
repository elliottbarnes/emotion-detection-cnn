import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from training_log import parse_training_log, read_training_log
ROW = '1/1 [==============================] - 2s 4ms/step - loss: 0.3 - accuracy: 0.5 - val_loss: 0.4 - val_accuracy: 0.6'
class LogTests(unittest.TestCase):
    def test_epoch_order_and_unrelated_lines(self):
        rows = parse_training_log('diagnostic\nEpoch 1/2\n'+ROW+'\nEpoch 2/2\n'+ROW+'\nProcess finished')
        self.assertEqual([x['epoch'] for x in rows], [1, 2])
        self.assertEqual(rows[0]['seconds'], 2)
    def test_malformed_or_duplicate_records_rejected(self):
        for text in ['', ROW, 'Epoch 1/2\n'+ROW+'\n'+ROW, 'Epoch 1/2\n'+ROW.replace('accuracy: 0.5','accuracy: 1.5')]:
            with self.assertRaises(ValueError): parse_training_log(text)
    def test_actual_log_is_chronological(self):
        rows = read_training_log(Path(__file__).resolve().parents[1] / 'out/out.txt')
        self.assertEqual([x['epoch'] for x in rows], list(range(1,50)))
if __name__ == '__main__': unittest.main()
