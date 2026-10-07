"""Export only numeric epoch metrics from the public log to the demo artifact."""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from training_log import read_training_log
source = ROOT / 'out/out.txt'
record = {'source': 'out/out.txt', 'sourceSha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'recordedRun': '2021-03-11', 'epochs': read_training_log(source)}
expected = json.dumps(record, indent=2) + '\n'
target = ROOT / 'demo/history.json'
if '--check' in sys.argv:
    if target.read_text() != expected:
        raise SystemExit('demo/history.json differs from the source log; run python3 scripts/export-history.py')
    print('Historical metrics match the committed source log')
else:
    target.write_text(expected)
