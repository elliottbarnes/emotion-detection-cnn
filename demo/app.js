import { architecture, synthetic, convolve, maxPool, KERNELS } from './core.js';
const $ = id => document.getElementById(id);
let layers = [], history;
function describe() { const row = layers[Number($('layer').value)]; if (!row) return; $('layer-info').textContent = `${row.name}: [${row.shape.join(' × ')}] · ${row.parameters.toLocaleString()} parameters. ${row.detail}`; }
function shape() {
  try {
    layers = architecture(Number($('size').value)); $('shape-error').textContent = '';
    $('parameters').textContent = layers.reduce((sum, x) => sum + x.parameters, 0).toLocaleString();
    $('layer').replaceChildren(...layers.map((row, i) => { const option=document.createElement('option'); option.value=i; option.textContent = `${String(i).padStart(2,'0')} · ${row.name} → ${row.shape.join(' × ')}`; return option; }));
    $('shapes').replaceChildren(...layers.map(row => { const tr=document.createElement('tr'); for (const value of [row.name,row.shape.join(' × '),row.parameters.toLocaleString()]) { const td=document.createElement('td'); td.textContent=value; tr.append(td); } return tr; }));
    $('architecture-result').hidden=false; describe();
  } catch (e) { $('shape-error').textContent=e.message; $('architecture-result').hidden=true; }
}
function draw(canvas, pixels, side, maxValue) { const ctx=canvas.getContext('2d'); canvas.width=side;canvas.height=side;const image=ctx.createImageData(side,side);pixels.forEach((v,i)=>{const value=Math.round(Math.min(1,v/maxValue)*255);image.data.set([value,value,value,255],i*4);});ctx.putImageData(image,0,0); }
function pattern() {
  const pixels=synthetic($('pattern').value), result=convolve(pixels,48,KERNELS[$('kernel').value]), pooled=maxPool(result.pixels,result.size);
  draw($('original'),pixels,48,255);draw($('filtered'),result.pixels,46,Math.max(1,...result.pixels));draw($('pooled'),pooled.pixels,23,Math.max(1,...pooled.pixels));
  const x=Number($('pixel-x').value),y=Number($('pixel-y').value), value=pixels[y*48+x];
  $('pixel-value').textContent=`Pixel (${x}, ${y}): ${value} ÷ 255 = ${(value/255).toFixed(6)}. Tensor position [0, ${y}, ${x}, 0].`;
  $('kernel-values').textContent=KERNELS[$('kernel').value].map((v,i)=>(i&&i%3===0?'\n':'')+v.toFixed(2).padStart(6)).join(' ');
}
function selectedEpoch() { if (!history) return; const row=history.epochs[Number($('epoch').value)-1];$('epoch-label').textContent=`Epoch ${row.epoch} / ${history.epochs.length}`;$('train-accuracy').textContent=(row.accuracy*100).toFixed(2)+'%';$('val-accuracy').textContent=(row.valAccuracy*100).toFixed(2)+'%';$('loss').textContent=row.loss.toFixed(4);$('val-loss').textContent=row.valLoss.toFixed(4);$('epoch-record').textContent=`Recorded epoch duration: ${row.seconds}s. Values are copied from the historical log, not a new run.`; const x=40+(row.epoch-1)/(history.epochs.length-1)*600;$('marker').setAttribute('x1',x);$('marker').setAttribute('x2',x); }
function chart() {
  const rows=history.epochs, metric=$('metric').value, train=metric==='accuracy'?'accuracy':'loss', validation=metric==='accuracy'?'valAccuracy':'valLoss';
  const max=metric==='accuracy'?1:Math.max(...rows.flatMap(x=>[x.loss,x.valLoss]));
  const points=key=>rows.map((row,i)=>`${40+i/(rows.length-1)*600},${220-row[key]/max*180}`).join(' ');
  $('train-line').setAttribute('points',points(train));$('validation-line').setAttribute('points',points(validation));$('chart-title').textContent=`Recorded ${metric} by epoch`;$('top-label').textContent=max.toFixed(2);selectedEpoch();
}
$('size').addEventListener('input',shape);$('layer').addEventListener('change',describe);
for(const id of ['pattern','kernel','pixel-x','pixel-y'])$(id).addEventListener('input',pattern);
$('epoch').addEventListener('input',selectedEpoch);$('metric').addEventListener('change',chart);
shape();pattern();
try { const response=await fetch('./history.json');if(!response.ok)throw new Error('History file did not load.');history=await response.json();$('epoch').max=history.epochs.length;$('epoch').value=history.epochs.length;$('history-status').textContent=`${history.epochs.length} epochs from the committed 11 March 2021 log.`;chart(); }
catch { $('history-status').textContent='The historical metrics could not be loaded. Serve the demo over HTTP, then reload.';$('epoch').disabled=true; }
