// Mirrors src/model.py shape/parameter arithmetic; no trained weights are loaded.
export function architecture(size = 48) {
  if (!Number.isInteger(size) || size < 32 || size > 96) throw new Error('Input side must be an integer from 32 to 96.');
  let h = size, channels = 1, flat = false, width = 0;
  const layers = [{ name: 'Input', shape: [h, h, channels], parameters: 0, detail: 'One grayscale channel; pixels divided by 255.' }];
  function add(name, parameters, detail) { layers.push({ name, shape: flat ? [width] : [h, h, channels], parameters, detail }); }
  function conv(filters) { const parameters = (3 * 3 * channels + 1) * filters; h -= 2; if (h < 1) throw new Error('The fourth convolution does not fit this input.'); channels = filters; add('Conv2D · ' + filters, parameters, '3 × 3 kernel, valid padding, stride 1; one bias per filter.'); }
  function pool() { h = Math.floor(h / 2); if (h < 1) throw new Error('A 2 × 2 pool does not fit. Try an input side of 46 or larger.'); add('MaxPool2D', 0, '2 × 2 window, stride 2, valid padding.'); }
  function activation(name = 'ReLU') { add(name, 0, name === 'ReLU' ? 'Keep positive activations; negative values become zero.' : 'Normalize class scores; this explorer does not compute predictions.'); }
  function batch() { add('Batch normalization', 4 * channels, 'Scale, offset, moving mean and variance per channel; two values per channel are non-trainable.'); }
  function flatten() { if (!flat) width = h * h * channels; flat = true; add('Flatten', 0, 'Reshape values into a vector; later flatten layers are redundant but retained from source.'); }
  function dense(units) { const parameters = (width + 1) * units; width = units; add('Dense · ' + units, parameters, 'One weight per input/output pair plus one bias per output.'); }
  conv(32); activation(); pool(); batch();
  conv(64); activation(); pool();
  conv(64); add('Dropout · 0.1', 0, 'Training-only dropout; shape unchanged.'); activation(); pool(); batch();
  conv(128); activation(); pool();
  for (const n of [128, 64, 32]) { flatten(); dense(n); activation(); }
  dense(2); activation('Softmax');
  return layers;
}
export const KERNELS = Object.freeze({ edge: [-1,0,1,-2,0,2,-1,0,1], blur: Array(9).fill(1/9), identity: [0,0,0,0,1,0,0,0,0] });
export function synthetic(kind, size = 48) {
  if (!['stripes', 'checker', 'circle'].includes(kind) || size !== 48) throw new Error('Unsupported synthetic pattern.');
  return Array.from({ length: size * size }, (_, i) => { const x = i % size, y = Math.floor(i / size); return kind === 'stripes' ? (Math.floor(x / 6) % 2 ? 224 : 32) : kind === 'checker' ? ((Math.floor(x/6)+Math.floor(y/6)) % 2 ? 224 : 32) : Math.hypot(x-23.5,y-23.5) < 14 ? 224 : 32; });
}
export function convolve(pixels, size, kernel) {
  if (pixels.length !== size * size || kernel.length !== 9 || size < 3) throw new Error('Invalid convolution shape.');
  const output = [], side = size - 2;
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) {
    let sum = 0;
    for (let ky=0; ky<3; ky++) for (let kx=0; kx<3; kx++) sum += pixels[(y+ky)*size+x+kx] / 255 * kernel[ky*3+kx];
    output.push(Math.max(0, sum));
  }
  return { pixels: output, size: side };
}
export function maxPool(pixels, size) {
  const side = Math.floor(size / 2), output = [];
  for (let y=0; y<side; y++) for (let x=0; x<side; x++) output.push(Math.max(pixels[2*y*size+2*x],pixels[2*y*size+2*x+1],pixels[(2*y+1)*size+2*x],pixels[(2*y+1)*size+2*x+1]));
  return { pixels: output, size: side };
}
