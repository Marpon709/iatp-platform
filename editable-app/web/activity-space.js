'use strict';

window.createActivitySpaceView = function createActivitySpaceView(options) {
  const {map, panel, config, pause, onChange, replay} = options;
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
  const number = (value, digits = 1) => Number(value).toLocaleString('zh-CN', {maximumFractionDigits: digits});
  const modes = {walk: '步行', bike: '骑行', bus: '公交', metro: '地铁', drive: '驾车'};
  const colors = {shopping_dining: '#ba782e', life_services: '#8461b0', recreation: '#238b70'};
  const typeById = new Map(config.activityTypes.map(type => [type.id, type]));
  const emptyCollection = () => ({type: 'FeatureCollection', features: []});
  let result, parameters, initialDemo, selected = new Set();
  let usingSample = true;
  let visible = false, editing = false, busy = false, status = '', error = '';
  let version = 0, jobId = null, controller = null, mapSelection = null;
  let step = 2, popup = null, lastPresentation = '';
  const markers = [];

  function colorFor(type) {
    return colors[typeById.get(type)?.group] || colors.life_services;
  }

  function summary() {
    const facilities = new Map();
    const facilitiesByType = new Map();
    for (const poi of result?.pois || []) {
      if (!selected.has(poi.type)) continue;
      const key = JSON.stringify([poi.name, poi.coordinate]);
      if (!facilities.has(key)) facilities.set(key, poi);
      if (!facilitiesByType.has(poi.type)) facilitiesByType.set(poi.type, new Set());
      facilitiesByType.get(poi.type).add(key);
    }
    const categories = (result?.categories || []).filter(category => selected.has(category.id))
      .map(category => ({...category, count: facilitiesByType.get(category.id)?.size || 0}));
    const pois = [...facilities.values()];
    return {
      categories,
      score: pois.reduce((total, poi) => total + poi.contribution, 0),
      count: pois.length,
      pois,
    };
  }

  async function request(path, payload, signal) {
    const response = await fetch(path, {
      method: payload ? 'POST' : 'GET',
      headers: {'Content-Type': 'application/json'},
      body: payload ? JSON.stringify(payload) : undefined,
      signal,
    });
    const value = await response.json();
    if (!response.ok || value.success === false) {
      throw new Error(value.message || value.error || '计算服务暂未响应，请重试。');
    }
    return value;
  }

  function refresh() {
    lastPresentation = '';
    if (visible) {
      onChange();
      updateMap();
      presentStep(step);
    }
  }

  function formHTML() {
    return `<form id="space-form" class="space-form">
      <div class="space-location-fields">${['origin', 'destination'].map(key => `
        <label class="field-label" for="space-${key}">${key === 'origin' ? '出发地' : '目的地'}</label>
        <div class="space-location-row"><input id="space-${key}" class="field-control" value="${escape(parameters[key].label)}" readonly>
          <button type="button" data-space-pick="${key}">地图选点</button></div>`).join('')}</div>
      <p class="space-form-hint" id="space-pick-hint">可在地图上指定出发地和目的地。</p>
      <div class="form-pair"><label><span class="field-label">出发时间</span><input id="space-departure" class="field-control" type="time" required value="${parameters.departure}"></label>
        <label><span class="field-label">最晚到达</span><input id="space-arrival" class="field-control" type="time" required value="${parameters.arrival}"></label></div>
      <div class="form-pair"><label><span class="field-label">出行方式</span><select id="space-mode" class="field-control">${Object.entries(modes).map(([id, label]) => `<option value="${id}" ${parameters.mode === id ? 'selected' : ''}>${label}</option>`).join('')}</select></label>
        <label><span class="field-label">每处活动停留</span><span class="duration-control"><input id="space-stay" type="number" min="1" max="240" required value="${parameters.stayMin}"><small>分钟</small></span></label></div>
      <div class="space-form-actions"><button type="button" id="space-edit-cancel">取消</button><button type="submit" class="primary-action">计算可活动范围</button></div>
    </form>`;
  }

  function html() {
    if (!result) return '<p class="panel-intro">正在读取活动空间示例…</p>';
    return `<div class="chapter-kicker">时间约束下的活动空间</div>
      <div class="panel-title-row"><h2>生活便利水平</h2><button type="button" id="space-edit" class="space-edit">${editing ? '收起条件' : '修改条件'}</button></div>
      <p class="space-intro">这段行程中，还能去哪里？ <small class="space-result-source">${usingSample ? '预计算样例' : '本次计算结果'}</small></p>
      ${editing ? formHTML() : `<div class="space-trip">
        <div><i class="space-start-dot"></i><span><small>${parameters.departure} 出发</small><strong>${escape(parameters.origin.label)}</strong></span></div>
        <div><i class="space-end-dot"></i><span><small>${parameters.arrival} 前到达</small><strong>${escape(parameters.destination.label)}</strong></span></div>
        <p>${result.gap.walk_fallback ? '步行接驳（本次路线回退）' : modes[parameters.mode]}<span>·</span>途中停留 ${number(parameters.stayMin, 0)} 分钟</p>
      </div>`}
      <div class="space-preferences"><h3>途中想去哪些地方？</h3><p>可多选，地图和得分随选择更新。</p>
        ${config.groups.map(group => `<fieldset><legend>${escape(group.label)}</legend><div class="space-type-grid">${config.activityTypes.filter(type => type.group === group.id).map(type => `<label class="space-type" style="--type-color:${colorFor(type.id)}"><input type="checkbox" data-space-type="${type.id}" ${selected.has(type.id) ? 'checked' : ''} ${busy ? 'disabled' : ''}><span>${escape(type.label)}</span></label>`).join('')}</div></fieldset>`).join('')}
      </div>
      ${busy ? `<div class="space-job" role="status"><i class="loading-orbit"></i><span id="space-job-status">${escape(status)}</span><button type="button" id="space-cancel">取消计算</button></div>` : ''}
      ${error ? `<p class="space-error" role="alert">${escape(error)}</p>` : ''}
      ${!selected.size ? '<p class="space-error" role="status">请选择至少一种感兴趣的设施。</p>' : ''}
      <details class="space-method"><summary>得分如何得到</summary>
        <p>先根据两个地点、可用时间和交通方式生成可活动范围，再统计范围内您选择的设施。设施离通勤路线越近，对得分的贡献越大。</p>
        <p>每处设施贡献 = exp(−0.003 × 偏离路线的距离〔米〕)，所选设施按地点去重后汇总贡献，得到沿途可达性得分。各类别数量允许重叠。</p>
        <p>总分为模型指数。在相同设施偏好下，可以比较不同时间或交通方式的结果。活动范围由路网采样估算，具体活动行程可在“我的一天”继续校验。</p>
      </details>
      <button type="button" id="space-replay" class="space-replay">▶ 演示路线、设施与得分</button>`;
  }

  function bind() {
    panel.querySelector('#space-edit')?.addEventListener('click', () => {
      pause();
      if (busy) return;
      editing = !editing;
      if (!editing) parameters = structuredClone(result.parameters);
      mapSelection = null;
      map.getCanvas().style.cursor = '';
      refresh();
    });
    panel.querySelector('#space-edit-cancel')?.addEventListener('click', () => {
      editing = false;
      parameters = structuredClone(result.parameters);
      mapSelection = null;
      map.getCanvas().style.cursor = '';
      refresh();
    });
    panel.querySelectorAll('[data-space-pick]').forEach(button => button.addEventListener('click', () => {
      pause();
      mapSelection = button.dataset.spacePick;
      map.getCanvas().style.cursor = 'crosshair';
      panel.querySelector('#space-pick-hint').textContent = `请点击地图设置${mapSelection === 'origin' ? '出发地' : '目的地'}。`;
    }));
    panel.querySelectorAll('[data-space-type]').forEach(input => input.addEventListener('change', () => {
      pause();
      if (input.checked) selected.add(input.dataset.spaceType);
      else selected.delete(input.dataset.spaceType);
      step = 2;
      updateMap();
      presentStep(step);
      const existing = panel.querySelector('.space-error[role="status"]');
      existing?.remove();
      if (!selected.size) {
        const message = document.createElement('p');
        message.className = 'space-error';
        message.setAttribute('role', 'status');
        message.textContent = '请选择至少一种感兴趣的设施。';
        panel.querySelector('.space-preferences').after(message);
      }
    }));
    panel.querySelector('#space-form')?.addEventListener('submit', event => {
      event.preventDefault();
      pause();
      const payload = structuredClone(parameters);
      payload.departure = panel.querySelector('#space-departure').value;
      payload.arrival = panel.querySelector('#space-arrival').value;
      payload.mode = panel.querySelector('#space-mode').value;
      payload.stayMin = Number(panel.querySelector('#space-stay').value);
      payload.selectedTypes = [...selected];
      calculate(payload);
    });
    panel.querySelector('#space-replay')?.addEventListener('click', replay);
    panel.querySelector('#space-cancel')?.addEventListener('click', cancel);
  }

  // 回放与取消都先使旧请求失效，迟到的响应不能覆盖当前画面。
  function stopCurrentRequest() {
    version++;
    controller?.abort();
    if (jobId) request(`/api/showcase/jobs/${jobId}/cancel`, {}).catch(() => {});
    controller = null;
    jobId = null;
    busy = false;
    status = '';
  }

  async function cancel() {
    stopCurrentRequest();
    error = '已取消本次计算，可调整条件后重试。';
    refresh();
  }

  function prepareDemo() {
    if (!initialDemo) return;
    stopCurrentRequest();
    popup?.remove();
    mapSelection = null;
    map.getCanvas().style.cursor = '';
    result = structuredClone(initialDemo);
    parameters = structuredClone(result.parameters);
    selected = new Set(parameters.selectedTypes);
    usingSample = true;
    editing = false;
    error = '';
    step = 0;
    refresh();
  }

  async function calculate(payload) {
    if (!payload.selectedTypes.length) {
      error = '请至少选择一种感兴趣的设施。';
      refresh();
      return;
    }
    const stamp = ++version;
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    busy = true;
    error = '';
    status = '正在提交计算条件…';
    // Keep the current result's conditions until a matching new result arrives.
    editing = false;
    parameters = structuredClone(result.parameters);
    refresh();
    try {
      const job = await request('/api/showcase/activity-space', {...payload, timeoutSec: 600}, signal);
      if (stamp !== version) {
        request(`/api/showcase/jobs/${job.jobId}/cancel`, {}).catch(() => {});
        return;
      }
      jobId = job.jobId;
      let snapshot;
      do {
        await new Promise(resolve => setTimeout(resolve, 800));
        if (stamp !== version) return;
        snapshot = await request(job.pollUrl, null, signal);
        if (stamp !== version) return;
        status = snapshot.stage || '计算中…';
        const statusElement = panel.querySelector('#space-job-status');
        if (statusElement) statusElement.textContent = status;
      } while (snapshot.status === 'running');
      if (stamp !== version) return;
      if (snapshot.status !== 'completed' || !snapshot.result?.success) {
        throw new Error(snapshot.error || snapshot.result?.message || snapshot.stage || '计算未完成。');
      }
      result = snapshot.result;
      usingSample = false;
      parameters = structuredClone(result.parameters);
      selected = new Set(parameters.selectedTypes);
      step = 2;
      popup?.remove();
      busy = false;
      jobId = null;
      refresh();
      frame();
    } catch (failure) {
      if (stamp !== version) return;
      busy = false;
      jobId = null;
      error = failure.name === 'AbortError' ? '本次请求已取消。' : failure.message;
      refresh();
    }
  }

  function initializeMap() {
    map.addSource('space-area', {type: 'geojson', data: emptyCollection()});
    map.addSource('space-route', {type: 'geojson', data: emptyCollection()});
    map.addSource('space-pois', {type: 'geojson', data: emptyCollection()});
    const before = map.getLayer('road-casings') ? 'road-casings' : undefined;
    map.addLayer({id: 'space-area-fill', type: 'fill', source: 'space-area', paint: {'fill-color': '#398b7c', 'fill-opacity': 0.19}}, before);
    map.addLayer({id: 'space-area-outline', type: 'line', source: 'space-area', paint: {'line-color': '#23816e', 'line-width': 2.5, 'line-dasharray': [3, 2]}});
    map.addLayer({id: 'space-route-casing', type: 'line', source: 'space-route', layout: {'line-cap': 'round', 'line-join': 'round'}, paint: {'line-color': '#ffffff', 'line-width': 9}});
    map.addLayer({id: 'space-route-line', type: 'line', source: 'space-route', layout: {'line-cap': 'round', 'line-join': 'round'}, paint: {'line-color': '#176ea0', 'line-width': 4.5}});
    map.addLayer({id: 'space-poi-dots', type: 'circle', source: 'space-pois', paint: {'circle-radius': ['interpolate', ['linear'], ['zoom'], 12, 4, 16, 7], 'circle-color': ['get', 'color'], 'circle-stroke-color': '#ffffff', 'circle-stroke-width': 1.6}});
    map.on('click', 'space-poi-dots', event => {
      if (!visible || mapSelection || !event.features.length) return;
      pause();
      const feature = event.features[0], properties = feature.properties;
      popup?.remove();
      popup = new maplibregl.Popup({maxWidth: '250px'}).setLngLat(feature.geometry.coordinates)
        .setHTML(`<div class="space-popup"><small>${escape(properties.label)}</small><strong>${escape(properties.name)}</strong></div>`).addTo(map);
    });
    map.on('click', event => {
      if (!visible || !mapSelection || busy) return;
      const key = mapSelection;
      parameters[key] = {label: `${key === 'origin' ? '出发地' : '目的地'} · 地图选点`, coordinate: [event.lngLat.lng, event.lngLat.lat]};
      mapSelection = null;
      map.getCanvas().style.cursor = '';
      panel.querySelector(`#space-${key}`).value = parameters[key].label;
      const marker = markers[key === 'origin' ? 0 : 1];
      marker?.setLngLat(parameters[key].coordinate);
      const markerLabel = marker?.getElement().querySelector('strong');
      if (markerLabel) markerLabel.textContent = parameters[key].label;
      panel.querySelector('#space-pick-hint').textContent = '地点已更新，点击“计算可活动范围”应用新条件。';
    });
    document.querySelector('#space-steps').addEventListener('click', event => {
      const button = event.target.closest('[data-space-step]');
      if (!button) return;
      pause();
      step = Number(button.dataset.spaceStep);
      presentStep(step);
    });
  }

  function updateMap() {
    if (!map.getSource('space-area') || !result) return;
    const values = summary();
    map.getSource('space-area').setData({type: 'FeatureCollection', features: result.areas.map(geometry => ({type: 'Feature', properties: {}, geometry}))});
    map.getSource('space-route').setData({type: 'FeatureCollection', features: result.routes.map(coordinates => ({type: 'Feature', properties: {}, geometry: {type: 'LineString', coordinates}}))});
    map.getSource('space-pois').setData({type: 'FeatureCollection', features: values.pois.map(poi => ({type: 'Feature', properties: {name: poi.name, label: typeById.get(poi.type)?.label || '', color: colorFor(poi.type)}, geometry: {type: 'Point', coordinates: poi.coordinate}}))});
    markers.forEach(marker => marker.remove());
    markers.length = 0;
    for (const [key, label, symbol] of [['origin', '出发', '起'], ['destination', '到达', '终']]) {
      const point = result.parameters[key];
      const node = document.createElement('div');
      node.className = `space-anchor ${key}`;
      node.innerHTML = `<b>${symbol}</b><span><small>${label} · ${result.parameters[key === 'origin' ? 'departure' : 'arrival']}</small><strong>${escape(point.label)}</strong></span>`;
      markers.push(new maplibregl.Marker({element: node, anchor: 'bottom', offset: [0, -8]}).setLngLat(point.coordinate).addTo(map));
    }
  }

  function presentStep(nextStep) {
    step = nextStep;
    const value = summary(), gap = result?.gap || {};
    const signature = JSON.stringify([visible, step, [...selected], version, result?.computedAt, result?.parameters]);
    if (signature === lastPresentation) return;
    lastPresentation = signature;
    const stageIds = ['space-area-fill', 'space-area-outline', 'space-route-casing', 'space-route-line', 'space-poi-dots'];
    stageIds.forEach(id => {
      if (map.getLayer(id)) map.setLayoutProperty(id, 'visibility', visible && (id !== 'space-poi-dots' || step >= 1) ? 'visible' : 'none');
    });
    markers.forEach(marker => marker.getElement().hidden = !visible);
    document.querySelector('#space-steps').hidden = !visible;
    document.querySelectorAll('[data-space-step]').forEach(button => {
      const current = Number(button.dataset.spaceStep) === step;
      button.classList.toggle('active', current);
      button.setAttribute('aria-pressed', String(current));
    });
    const card = document.querySelector('#space-result');
    card.hidden = !visible;
    const hasSpace = result?.hasActivitySpace;
    const totalMinutes = result ? minutes(result.parameters.arrival) - minutes(result.parameters.departure) : 0;
    card.innerHTML = `<div class="space-result-kicker">${['路线与可活动范围', '您的沿途设施选择', '沿途可达性评估'][step]}</div>
      <h3>${!hasSpace ? '当前时间下无额外活动范围' : step === 0 ? '留出停留时间后，还能走多远' : step === 1 ? '这些设施位于可活动范围内' : '这段行程的生活便利水平'}</h3>
      ${step === 2 && hasSpace ? `<div class="space-score"><strong>${selected.size ? number(value.score, 2) : '—'}</strong><span>沿途可达性得分</span></div>` : ''}
      ${step >= 1 && hasSpace ? `<div class="space-count"><strong>${value.count}</strong><span>处所选设施</span><b>${selected.size} 类偏好</b></div><div class="space-breakdown">${value.categories.map(category => `<div><span><i style="background:${colorFor(category.id)}"></i>${escape(category.label)}</span><b>${category.count}<small> 处</small></b></div>`).join('')}</div>${step === 2 ? '<p class="space-score-caption">汇总所选设施的贡献，同一地点计入一次。越贴近路线，贡献越大。</p>' : ''}` : ''}
      <div class="space-time-equation"><span>可用 <b>${totalMinutes}</b> 分钟</span><span>直达约 <b>${number(gap.commute_time_min || 0)}</b></span><span>停留 <b>${number(result?.parameters.stayMin || 0)}</b></span><span>机动约 <b>${number(Math.max(0, gap.elastic_time_min || 0))}</b></span></div>
      ${!hasSpace ? `<p class="space-score-caption">${escape(result?.message || '请延长时间窗口或缩短停留时间后重新计算。')}</p>` : ''}
      <p class="space-route-source">${result?.source.routeSource === 'local_network' ? '南京本地路网计算' : '路线与活动空间由计算引擎生成'}</p>`;
    const legend = document.querySelector('#space-legend');
    legend.hidden = !visible;
    legend.innerHTML = '<span><i class="space-legend-route"></i>直达路线</span><span><i class="space-legend-area"></i>可活动范围</span>' + (step >= 1 ? '<span><i class="space-legend-poi"></i>所选设施</span>' : '');
  }

  function minutes(time) {
    const [hour, minute] = time.split(':').map(Number);
    return hour * 60 + minute;
  }

  function frame() {
    if (!result) return;
    const coordinates = [result.parameters.origin.coordinate, result.parameters.destination.coordinate, ...result.polygons.flat()];
    const bounds = new maplibregl.LngLatBounds();
    coordinates.forEach(coordinate => bounds.extend(coordinate));
    const narrow = innerWidth < 1100;
    map.fitBounds(bounds, {padding: {left: innerWidth > 680 ? 420 : 28, right: narrow ? 70 : 380, top: 180, bottom: 170}, pitch: 0, bearing: 0, duration: 800, maxZoom: 15.5});
  }

  async function initialize() {
    const demo = await (await fetch('./data/activity-space-demo.json')).json();
    initialDemo = structuredClone(demo);
    result = structuredClone(demo);
    parameters = structuredClone(result.parameters);
    selected = new Set(parameters.selectedTypes);
    initializeMap();
    updateMap();
    return view;
  }

  const view = {
    initialize, html, bind, summary, frame, prepareDemo,
    setVisible(value) {
      visible = value;
      if (!value) {
        popup?.remove();
        mapSelection = null;
        map.getCanvas().style.cursor = '';
      }
      presentStep(step);
    },
    present(time) { presentStep(time == null ? 2 : time < 2.8 ? 0 : time < 6 ? 1 : 2); },
    get result() { return result; },
  };
  return view;
};
