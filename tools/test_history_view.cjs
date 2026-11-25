// Browser-view regression checks with controlled API responses; no map or network required.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const nodes = new Map();
function element(id) {
  if (!nodes.has(id)) nodes.set(id, {
    value: '', innerHTML: '', textContent: '', disabled: false, attributes: {}, listeners: {},
    setAttribute(key, value) { this.attributes[key] = value; },
    addEventListener(event, listener) { this.listeners[event] = listener; },
    querySelectorAll() { return []; },
  });
  return nodes.get(id);
}
const run = {
  jobId: 'job-1', kind: '<img src=x onerror=alert(1)>', status: 'completed',
  modelVersion: 'model-2026', created_at: '2026-11-19T08:00:00+00:00',
  finished_at: '2026-11-19T08:01:00+00:00',
  parameters: {query: '</pre><script>alert(1)</script>'},
  summary: {totalCarbonG: 123.4}, result: {success: true, source: {kind: '<script>bad</script>'}},
};
let rows = [run], failList = false, delayDetail = false, finishDetail;
const requests = [];
const context = {
  document: {getElementById: element}, AbortController, URLSearchParams, setTimeout, clearTimeout, console,
  fetch: async url => {
    requests.push(url);
    if (url.startsWith('/api/showcase/history?')) {
      if (failList) throw new Error('test service unavailable');
      const query = new URLSearchParams(url.split('?')[1]);
      return {ok: true, json: async () => ({success: true, items: rows, total: rows.length ? 12 : 0, page: Number(query.get('page')), pageSize: 10})};
    }
    if (delayDetail) return new Promise(resolve => { finishDetail = () => resolve({ok: true, json: async () => ({success: true, run})}); });
    return {ok: true, json: async () => ({success: true, run})};
  },
};
const flush = () => new Promise(resolve => setImmediate(resolve));
vm.runInNewContext(fs.readFileSync(path.resolve(__dirname, '../editable-app/web/history.js'), 'utf8'), context);

(async () => {
  await flush();
  assert.match(element('history-list').innerHTML, /&lt;img/);
  assert.ok(!element('history-list').innerHTML.includes('<img'));
  assert.match(element('history-detail').innerHTML, /&lt;script&gt;/);
  assert.ok(!element('history-detail').innerHTML.includes('<script>'));
  assert.match(element('history-detail').innerHTML, /123\.4/);
  assert.equal(element('history-next').disabled, false);
  element('history-next').listeners.click();
  await flush();
  assert.ok(requests.some(url => url.includes('page=2')));
  assert.equal(element('history-page').textContent, '2 / 2');
  assert.equal(element('history-next').disabled, true);

  // Changing filters resets pagination and invalidates the previous detail response.
  delayDetail = true;
  element('history-status').value = 'completed';
  element('history-status').listeners.change();
  await flush();
  assert.ok(requests.some(url => url.includes('page=1') && url.includes('status=completed')));
  assert.equal(typeof finishDetail, 'function');
  rows = [];
  element('history-kind').value = 'activity-space';
  element('history-kind').listeners.change();
  await flush();
  finishDetail();
  await flush();
  assert.match(element('history-list').innerHTML, /暂无计算记录/);
  assert.ok(!element('history-detail').innerHTML.includes('123.4'));
  assert.equal(element('history-prev').disabled, true);
  assert.equal(element('history-next').disabled, true);

  failList = true;
  element('history-filters').listeners.submit({preventDefault() {}});
  await flush();
  assert.match(element('history-list').innerHTML, /刷新记录/);
  assert.equal(element('history-list').attributes['aria-busy'], 'false');
  console.log('PASS: history escapes API data, paginates and filters, shows empty/error states, and rejects stale details.');
})().catch(error => { console.error(error); process.exitCode = 1; });
