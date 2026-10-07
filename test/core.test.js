import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { architecture, convolve, maxPool, synthetic, KERNELS } from '../demo/core.js';
test('source architecture matches every shape and parameter count in the historical model summary', () => {
  const log = readFileSync('out/out.txt', 'utf8');
  const historical = [...log.matchAll(/^\w+ \([^\n]+?\s+\(None, ([\d, ]+)\)\s+(\d+)$/gm)].map(match => ({ shape: match[1].split(',').map(x => Number(x.trim())), parameters: Number(match[2]) }));
  const actual = architecture(48).slice(1).map(({ shape, parameters }) => ({ shape, parameters }));
  assert.equal(historical.length, 26);
  assert.deepEqual(actual, historical);
  assert.equal(actual.reduce((sum, row) => sum + row.parameters, 0), 156898);
  assert.throws(() => architecture(32), /does not fit/);
  assert.ok(architecture(96).at(-1).shape[0] === 2);
});
test('normalization, identity convolution and max pooling preserve known pixel values', () => {
  const pixels = [0,0,0,0,255,0,0,0,0];
  assert.deepEqual(convolve(pixels, 3, KERNELS.identity), { pixels: [1], size: 1 });
  assert.ok(Math.abs(convolve(Array(9).fill(255), 3, KERNELS.blur).pixels[0] - 1) < 1e-12);
  assert.deepEqual(maxPool([1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16], 4), { pixels: [6,8,14,16], size: 2 });
  for (const kind of ['stripes', 'checker', 'circle']) { const image=synthetic(kind); assert.equal(image.length, 2304); assert.ok(image.every(x => x === 32 || x === 224)); }
});
test('published historical metrics reproduce the original log with no ML dependency', () => {
  const exportCheck = spawnSync('python3', ['scripts/export-history.py', '--check'], { encoding:'utf8' });
  assert.equal(exportCheck.status, 0, exportCheck.stderr);
  const result = spawnSync('python3', ['-m', 'unittest', 'discover', '-s', 'test', '-p', '*_test.py'], { encoding:'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const history = JSON.parse(readFileSync('demo/history.json','utf8'));
  assert.equal(history.epochs.length,49);
  assert.equal(history.epochs[0].epoch,1);
  assert.equal(history.epochs.at(-1).epoch,49);
  assert.equal(history.epochs[0].accuracy,0.5826);
});
