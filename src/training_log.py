"""Parse the committed historical training log without ML dependencies."""
from pathlib import Path
import re

METRICS = re.compile(r"^\d+/\d+ \[=+\] - (\d+)s .*? - loss: ([0-9.]+) - accuracy: ([0-9.]+) - val_loss: ([0-9.]+) - val_accuracy: ([0-9.]+)\s*$")
EPOCH = re.compile(r"^Epoch (\d+)/(\d+)$")

def parse_training_log(text):
    rows = []
    epoch = None
    for line in text.splitlines():
        header = EPOCH.match(line)
        if header:
            epoch = int(header.group(1))
        match = METRICS.match(line)
        if match:
            if epoch is None or (rows and epoch <= rows[-1]['epoch']):
                raise ValueError('Missing or non-increasing epoch header')
            seconds, loss, accuracy, val_loss, val_accuracy = map(float, match.groups())
            if not 0 <= accuracy <= 1 or not 0 <= val_accuracy <= 1:
                raise ValueError('Accuracy outside [0, 1]')
            rows.append(dict(epoch=epoch, seconds=int(seconds), loss=loss, accuracy=accuracy,
                             valLoss=val_loss, valAccuracy=val_accuracy))
            epoch = None
    if not rows:
        raise ValueError('No complete epoch records found')
    return rows

def read_training_log(path):
    return parse_training_log(Path(path).read_text(encoding='utf-8'))
