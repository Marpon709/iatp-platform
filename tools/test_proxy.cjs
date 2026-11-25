const assert = require('node:assert/strict');
const http = require('node:http');
const path = require('node:path');
const {startServer} = require('../editable-app/server.cjs');

(async () => {
  const upstream = http.createServer((request, response) => {
    let body = '';
    request.on('data', chunk => body += chunk);
    request.on('end', () => {
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify({length: Number(request.headers['content-length']), body: JSON.parse(body)}));
    });
  });
  await new Promise(resolve => upstream.listen(0, '127.0.0.1', resolve));
  const proxy = await startServer({
    webRoot: path.resolve(__dirname, '../editable-app/web'),
    getBridge: () => `http://127.0.0.1:${upstream.address().port}`,
  });
  try {
    const payload = {selectedTypes: ['bookstore', 'park'], stayMin: 20};
    const body = JSON.stringify(payload);
    const response = await fetch(proxy.url + '/api/showcase/activity-space', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body,
    });
    const received = await response.json();
    assert.deepEqual(received.body, payload);
    assert.equal(received.length, Buffer.byteLength(body));
    console.log('PASS: desktop proxy forwards the JSON body with its Content-Length.');
  } finally {
    await proxy.close();
    await new Promise(resolve => upstream.close(resolve));
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
