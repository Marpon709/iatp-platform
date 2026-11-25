const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '../editable-app/web');
const demo = JSON.parse(fs.readFileSync(path.join(root, 'data/activity-space-demo.json'), 'utf8'));
const config = JSON.parse(fs.readFileSync(path.join(root, 'data/live-config.json'), 'utf8'));

function element() {
  return {
    listeners: {}, classList: {toggle() {}}, style: {}, hidden: false, innerHTML: '',
    setAttribute() {}, addEventListener(type, handler) { this.listeners[type] = handler; },
    querySelector() { return {textContent: ''}; },
  };
}
const nodes = new Map();
const inputs = config.activityTypes.map(type => ({...element(), dataset: {spaceType: type.id}, checked: demo.parameters.selectedTypes.includes(type.id)}));
const panel = {
  querySelector(selector) { return nodes.get(selector) || null; },
  querySelectorAll() { return inputs; },
};
const sources = new Map(), layers = new Map();
const map = {
  addSource(id, source) { sources.set(id, {...source, setData(data) { this.data = data; }}); },
  getSource(id) { return sources.get(id); },
  addLayer(layer) { layers.set(layer.id, layer); },
  getLayer(id) { return layers.get(id); },
  setLayoutProperty(id, key, value) { layers.get(id).visibility = value; },
  on() {}, getCanvas() { return {style: {}}; },
};
class Marker {
  constructor({element}) { this.element = element; }
  remove() {} setLngLat() { return this; } addTo() { return this; } getElement() { return this.element; }
}
const context = {
  window: {}, structuredClone, Map, Set, console, AbortController,
  setTimeout: callback => { Promise.resolve().then(callback); return 0; },
  fetch: async () => ({ok: true, json: async () => demo}),
  document: {createElement: element, querySelector(selector) {
    if (!nodes.has(selector)) nodes.set(selector, element());
    return nodes.get(selector);
  }, querySelectorAll() { return []; }},
  maplibregl: {Marker},
};
vm.runInNewContext(fs.readFileSync(path.join(root, 'activity-space.js'), 'utf8'), context);

(async () => {
  let paused = 0;
  const view = context.window.createActivitySpaceView({map, panel, config, pause: () => paused++, onChange() {}});
  await view.initialize();
  view.setVisible(true);
  assert.match(view.html(), /地图和得分随选择更新/);
  const initial = view.summary();
  assert.equal(initial.count, 61);
  assert.equal(initial.categories.find(category => category.id === 'supermarket').count, 50);
  assert.equal(initial.score.toFixed(2), '30.07');
  view.bind();
  const supermarket = inputs.find(input => input.dataset.spaceType === 'supermarket');
  supermarket.checked = false;
  supermarket.listeners.change();
  assert.equal(view.summary().count, 11);
  assert.equal(view.summary().score.toFixed(2), '4.79');
  assert.equal(sources.get('space-pois').data.features.length, 11);
  const mall = inputs.find(input => input.dataset.spaceType === 'shopping_mall');
  mall.checked = true;
  mall.listeners.change();
  const mallCount = view.summary().count, mallScore = view.summary().score;
  supermarket.checked = true;
  supermarket.listeners.change();
  const uniqueNames = new Set(view.summary().pois.map(poi => JSON.stringify([poi.name, poi.coordinate])));
  assert.equal(uniqueNames.size, view.summary().count);
  assert.ok(view.summary().count <= mallCount + 50);
  assert.ok(view.summary().score >= mallScore);
  view.present(0);
  assert.equal(layers.get('space-poi-dots').visibility, 'none');
  view.present(4);
  assert.equal(layers.get('space-poi-dots').visibility, 'visible');
  view.present(15);
  assert.match(nodes.get('#space-result').innerHTML, /沿途可达性得分/);
  view.setVisible(false);
  assert.equal(layers.get('space-route-line').visibility, 'none');
  assert.ok(paused >= 3);
  console.log('PASS: preference filtering changes map/count/score together, duplicate matches count once, and playback reveals the three steps.');

  // Playback restores the immutable sample rather than replaying edited preferences/results.
  view.result.parameters.origin.label = 'changed after initialization';
  view.setVisible(true);
  view.prepareDemo();
  assert.equal(view.summary().count, initial.count);
  assert.equal(view.summary().score, initial.score);
  assert.equal(view.result.parameters.origin.label, demo.parameters.origin.label);
  assert.match(view.html(), /预计算样例/);
  assert.equal(layers.get('space-poi-dots').visibility, 'none');

  // A submission may finish after replay starts, even if abort is ignored by the transport.
  for (const [selector, value] of Object.entries({
    '#space-departure': '17:00', '#space-arrival': '18:30', '#space-mode': 'walk', '#space-stay': '20',
  })) nodes.set(selector, {...element(), value});
  const form = element();
  nodes.set('#space-form', form);
  let resolveSubmission;
  const calls = [];
  context.fetch = (url, options) => {
    calls.push({url, options});
    if (url === '/api/showcase/activity-space') {
      return new Promise(resolve => { resolveSubmission = value => resolve({ok: true, json: async () => value}); });
    }
    return Promise.resolve({ok: true, json: async () => ({success: true})});
  };
  view.bind();
  form.listeners.submit({preventDefault() {}});
  assert.match(view.html(), /正在提交计算条件/);
  const submittedSignal = calls[0].options.signal;
  view.prepareDemo();
  assert.equal(submittedSignal.aborted, true);
  resolveSubmission({success: true, jobId: 'late-job', pollUrl: '/api/showcase/jobs/late-job'});
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(calls.some(call => call.url === '/api/showcase/jobs/late-job/cancel'));
  assert.equal(view.summary().count, initial.count);
  assert.ok(!view.html().includes('正在提交计算条件'));
  console.log('PASS: replay restores the original sample and cancels a late submission without overwriting it.');

  let resolveSnapshot;
  context.fetch = (url, options) => {
    calls.push({url, options});
    if (url === '/api/showcase/activity-space') return Promise.resolve({ok: true, json: async () => ({jobId: 'running-job', pollUrl: '/api/showcase/jobs/running-job'})});
    if (url === '/api/showcase/jobs/running-job') return new Promise(resolve => { resolveSnapshot = value => resolve({ok: true, json: async () => value}); });
    return Promise.resolve({ok: true, json: async () => ({success: true})});
  };
  form.listeners.submit({preventDefault() {}});
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(typeof resolveSnapshot, 'function');
  view.prepareDemo();
  const staleResult = structuredClone(demo);
  staleResult.pois = [];
  resolveSnapshot({status: 'completed', result: staleResult});
  await new Promise(resolve => setImmediate(resolve));
  assert.ok(calls.some(call => call.url === '/api/showcase/jobs/running-job/cancel'));
  assert.equal(view.summary().count, initial.count);
  assert.match(view.html(), /预计算样例/);
  console.log('PASS: a late polling result cannot replace the replay sample.');
})().catch(error => { console.error(error); process.exitCode = 1; });
