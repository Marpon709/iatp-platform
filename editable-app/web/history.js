'use strict';

(() => {
  const byId = id => document.getElementById(id);
  const list = byId('history-list'), detail = byId('history-detail');
  const statusLabels = {running: '计算中', completed: '已完成', failed: '失败', cancelled: '已取消', timed_out: '超时', interrupted: '已中断'};
  const kindLabels = {'activity-space': '活动空间', recommendation: '场所推荐', 'route-chain': '行程路线', 'result-review': '结果复核'};
  const embedded = typeof location !== 'undefined' && new URLSearchParams(location.search).get('showcase') === '1';
  if (embedded) document.body.classList.add('embedded-history');
  const escape = value => String(value ?? '').replace(/[&<>"']/g, character =>
    ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[character]));
  let page = 1, pageCount = 1, listVersion = 0, detailVersion = 0, selectedId = null;

  function formatTime(value) {
    if (!value) return '—';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', {hour12: false});
  }

  function statusHTML(status) {
    const known = Object.hasOwn(statusLabels, status);
    return `<span class="status${known ? ' status-' + status : ''}">${escape(statusLabels[status] || status || '未知')}</span>`;
  }

  async function request(path, body) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(path, {signal: controller.signal, method: body ? 'POST' : 'GET', headers: body ? {'Content-Type':'application/json'} : {}, body: body ? JSON.stringify(body) : undefined});
      let value;
      try { value = await response.json(); }
      catch { throw new Error('计算记录暂未返回，请确认计算服务已启动。'); }
      if (!response.ok || value.success === false) throw new Error(value.message || value.error || '无法读取计算记录。');
      return value;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('读取记录超时，请稍后刷新。');
      if (error instanceof TypeError) throw new Error('计算服务未连接，请返回分析页面确认服务已启动。');
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function selectRow(jobId) {
    list.querySelectorAll('[data-record]').forEach(row => row.classList.toggle('selected', row.dataset.record === jobId));
  }

  function renderSummary(summary) {
    const fields = {
      totalCarbonG: ['总碳排放', 'g CO₂'], totalTravelMin: ['总出行时间', '分钟'], totalDistanceKm: ['出行距离', 'km'],
      feasibleCandidateCount: ['可行场所', '处'], validatedCandidateCount: ['已校验候选', '处'], candidateCount: ['候选场所', '处'],
      poiCount: ['范围内设施', '处'], facilityCount: ['范围内设施', '处'], score: ['可达性得分', ''],
      totalScore: ['可达性得分', ''], routeCount: ['路线段数', '段'], legCount: ['路线段数', '段'],
    };
    const entries = Object.entries(summary || {}).filter(([key, value]) => fields[key] && typeof value === 'number' && Number.isFinite(value));
    if (!entries.length) return '';
    return `<h3 class="summary-heading">结果摘要</h3><div class="result-grid">${entries.map(([key, value]) =>
      `<div class="result-value">${escape(value.toLocaleString('zh-CN', {maximumFractionDigits: 2}))}<span class="result-label">${fields[key][0]}${fields[key][1] ? ' · ' + fields[key][1] : ''}</span></div>`).join('')}</div>`;
  }

  async function openRun(jobId) {
    selectedId = jobId;
    selectRow(jobId);
    const stamp = ++detailVersion;
    detail.setAttribute('aria-busy', 'true');
    detail.innerHTML = '<p class="empty-state">正在读取计算详情…</p>';
    try {
      const data = await request('/api/showcase/history/' + encodeURIComponent(jobId));
      if (stamp !== detailVersion) return;
      const run = data.run;
      if (!run || run.jobId !== jobId) throw new Error('这条记录暂不可用，请刷新后重试。');
      detail.innerHTML = `<div class="detail-heading"><div><h2>${escape(kindLabels[run.kind] || run.kind || '分析计算')}</h2><p class="detail-id">${escape(run.jobId)}</p></div>${statusHTML(run.status)}</div>
        <dl class="detail-meta"><dt>提交时间</dt><dd>${escape(formatTime(run.created_at))}</dd><dt>结束时间</dt><dd>${escape(formatTime(run.finished_at))}</dd><dt>模型版本</dt><dd class="mono">${escape(run.modelVersion || '—')}</dd></dl>
        ${run.stage ? `<p class="detail-stage">${escape(run.stage)}</p>` : ''}
        ${run.error ? `<p class="record-error">${escape(run.error)}</p>` : ''}
        ${run.isPartial || run.result?.partial ? '<p class="partial-note">仅保留已完成的部分结果，请结合计算状态使用。</p>' : ''}
        ${renderSummary(run.summary)}
        ${renderValidation(run.validation)}
        <details class="json-detail"><summary>查看计算条件</summary><pre>${escape(JSON.stringify(run.parameters || {}, null, 2))}</pre></details>
        ${run.result ? `<details class="json-detail"><summary>查看完整返回结果</summary><pre>${escape(JSON.stringify(run.result, null, 2))}</pre></details>` : '<p class="history-note">这条记录尚无计算结果。</p>'}`;
    } catch (error) {
      if (stamp !== detailVersion) return;
      detail.innerHTML = `<p class="empty-state error-state">${escape(error.message)}<button type="button" id="detail-retry">重新读取</button></p>`;
      byId('detail-retry')?.addEventListener('click', () => openRun(jobId));
    } finally {
      if (stamp === detailVersion) detail.setAttribute('aria-busy', 'false');
    }
  }

  function renderValidation(validation) {
    if (!validation) return '';
    const labels = {passed: '基础校验通过', failed: '存在需检查的结果', skipped: '本次未执行自动校验'};
    return `<section class="validation-result"><h3>${escape(labels[validation.status] || '结果校验')}</h3>
      ${(validation.checks || []).map(check => `<p><strong>${check.passed ? '✓' : '!'} ${escape(check.label)}</strong><br><small>${escape(check.message)}</small></p>`).join('')}
      <p class="history-note">${escape(validation.message || '')}</p></section>`;
  }

  async function loadRuns() {
    const stamp = ++listVersion;
    ++detailVersion;
    selectedId = null;
    detail.innerHTML = '<p class="empty-state">选择一条记录，查看计算条件和结果。</p>';
    detail.setAttribute('aria-busy', 'false');
    list.setAttribute('aria-busy', 'true');
    list.innerHTML = '<p class="empty-state">正在读取计算记录…</p>';
    byId('history-prev').disabled = true;
    byId('history-next').disabled = true;
    byId('history-total').textContent = '';
    const query = new URLSearchParams({page: String(page), pageSize: '10'});
    for (const key of ['kind', 'status']) {
      const value = byId('history-' + key).value;
      if (value) query.set(key, value);
    }
    try {
      const data = await request('/api/showcase/history?' + query.toString());
      if (stamp !== listVersion) return;
      const items = data.items || [];
      page = data.page || page;
      pageCount = Math.max(1, Math.ceil((data.total || 0) / (data.pageSize || 10)));
      byId('history-page').textContent = `${page} / ${pageCount}`;
      byId('history-total').textContent = `共 ${data.total || 0} 条计算记录`;
      byId('history-prev').disabled = page <= 1;
      byId('history-next').disabled = page >= pageCount;
      if (!items.length) {
        list.innerHTML = '<p class="empty-state">当前条件下暂无计算记录。<br><a href="./">返回分析页面，修改条件并提交计算</a></p>';
        return;
      }
      list.innerHTML = `<div class="table-scroll"><table><thead><tr><th scope="col">计算</th><th scope="col">状态</th><th scope="col">提交时间</th></tr></thead><tbody>${items.map(run => `<tr data-record="${escape(run.jobId)}"><td><button type="button" class="record-open" data-open="${escape(run.jobId)}">${escape(kindLabels[run.kind] || run.kind || '分析计算')}<small>${escape(run.jobId)}</small></button></td><td>${statusHTML(run.status)}</td><td class="time">${escape(formatTime(run.created_at))}</td></tr>`).join('')}</tbody></table></div>`;
      list.querySelectorAll('[data-open]').forEach(button => button.addEventListener('click', () => openRun(button.dataset.open)));
      openRun(items[0].jobId);
    } catch (error) {
      if (stamp !== listVersion) return;
      byId('history-page').textContent = '—';
      list.innerHTML = `<p class="empty-state error-state">${escape(error.message)}<br>请确认计算服务已启动，再点击“刷新记录”。</p>`;
    } finally {
      if (stamp === listVersion) list.setAttribute('aria-busy', 'false');
    }
  }

  byId('history-filters').addEventListener('submit', event => { event.preventDefault(); page = 1; loadRuns(); });
  for (const key of ['kind', 'status']) byId('history-' + key).addEventListener('change', () => { page = 1; loadRuns(); });
  byId('history-prev').addEventListener('click', () => { if (page > 1) { page--; loadRuns(); } });
  byId('history-next').addEventListener('click', () => { if (page < pageCount) { page++; loadRuns(); } });
  async function reviewSample() {
    const button=byId('review-sample');
    if(button)button.disabled=true;
    try {
      const data=await request('/api/showcase/review-sample',{});
      byId('history-kind').value=embedded?'':'result-review';page=1;
      await loadRuns();
      await openRun(data.historyId);
    } catch(error) { detail.innerHTML=`<p class="empty-state error-state">${escape(error.message)}</p>`; }
    finally { if(button)button.disabled=false; }
  }
  byId('review-sample')?.addEventListener('click', reviewSample);
  if (embedded) reviewSample(); else loadRuns();
  let conditionsShown=false,filteredForStory=false;
  if (typeof window !== 'undefined') window.addEventListener('message', event => {
    if(event.origin!==location.origin || event.data?.type!=='iatp-showcase')return;
    if(embedded && event.data.time>=41 && !filteredForStory){filteredForStory=true;byId('history-kind').value='result-review';loadRuns();}
    if(embedded && event.data.time<41)filteredForStory=false;
    const showConditions=event.data.time>=44;
    const conditions=detail.querySelector('.json-detail');
    if(conditions && conditionsShown!==showConditions){conditionsShown=showConditions;conditions.open=showConditions;if(showConditions)conditions.scrollIntoView({block:'end',behavior:'smooth'});else window.scrollTo({top:0});}
  });
})();
