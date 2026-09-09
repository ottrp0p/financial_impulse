/* Impulse Screener — single-page frontend, no build step. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const api = async (path, opt = {}) => {
  const r = await fetch('/api' + path, { headers: { 'Content-Type': 'application/json' }, ...opt, body: opt.body ? JSON.stringify(opt.body) : undefined });
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
};
const fmt = (v, d = 2) => (v == null || !isFinite(v)) ? '–' : (+v).toFixed(d);
const pct = (v, d = 1) => (v == null || !isFinite(v)) ? '–' : ((+v) * 100).toFixed(d) + '%';
const sgn = v => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
const usd = v => '$' + Math.round(v).toLocaleString();
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a = {}, p) => { const n = document.createElementNS(NS, t); for (const k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; };
const tip = $('#tip');
const showTip = (e, h) => { tip.innerHTML = h; tip.style.display = 'block'; tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px'; };
const hideTip = () => tip.style.display = 'none';

const S = { sid: null, screen: null, presets: null, book: null, nameSide: {} };

/* ---------- boot ---------- */
async function boot() {
  S.presets = await api('/presets');
  await loadScreens(new URLSearchParams(location.search).get('screen'));
  $$('#tabs button').forEach(b => b.addEventListener('click', () => showTab(b.dataset.tab)));
  $('#newScreen').addEventListener('click', async () => { const name = prompt('Screen name'); if (!name) return; const s = await api('/screens', { method: 'POST', body: { name } }); await loadScreens(s.id); showTab('screen'); });
  $('#screenSel').addEventListener('change', e => selectScreen(e.target.value));
  $('#delScreen').addEventListener('click', async () => { if (!S.sid || !confirm(`Delete screen "${S.screen.name}"? Price data is kept.`)) return; await api(`/screens/${S.sid}`, { method: 'DELETE' }); S.sid = null; S.book = null; await loadScreens(); if (!S.sid) { renderHeader(); $('#todayLong').innerHTML = $('#todayShort').innerHTML = '<div class="empty">Create a screen to begin.</div>'; } });
  $('#refresh').addEventListener('click', async () => { await api(`/screens/${S.sid}/refresh`, { method: 'POST' }); pollJob(); });
  $$('button.add').forEach(b => b.addEventListener('click', () => addNames(+b.dataset.side)));
  $$('select.preset').forEach(sel => sel.addEventListener('change', () => setStrategy(+sel.dataset.side, sel.value)));
  $('#generate').addEventListener('click', async () => { try { await api(`/screens/${S.sid}/generate`, { method: 'POST' }); pollJob(); } catch (e) { alert(e.message); } });
  $('#unitDollars').addEventListener('change', async e => { await api(`/screens/${S.sid}/units`, { method: 'PUT', body: { unit_dollars: +e.target.value } }); S.screen.unit_dollars = +e.target.value; renderToday(); });
  $('#nameSel').addEventListener('change', e => loadName(e.target.value));
  const t = new URLSearchParams(location.search).get('tab'); if (t) showTab(t);
}
function showTab(t) { history.replaceState(null, '', '?tab=' + t + (S.sid ? '&screen=' + S.sid : '')); $$('#tabs button').forEach(b => b.classList.toggle('on', b.dataset.tab === t)); $$('.tab').forEach(s => s.classList.toggle('on', s.id === 'tab-' + t)); }

async function loadScreens(pick) {
  const list = await api('/screens');
  const sel = $('#screenSel'); sel.innerHTML = list.map(s => `<option value="${s.id}">${s.name}</option>`).join('') || '<option value="">(no screens)</option>';
  const id = pick || (list[0] && list[0].id);
  if (id) { sel.value = id; await selectScreen(id); }
}
async function selectScreen(id) {
  S.sid = id; S.screen = await api(`/screens/${id}`);
  renderScreenTab();
  if (S.screen.job && S.screen.job.running) pollJob(); else { renderJob(null); await loadBook(); }
}
async function loadBook() {
  if (!S.sid) return;
  try { S.book = await api(`/screens/${S.sid}/book`); } catch (e) { S.book = null; }
  renderHeader(); renderToday(); renderBook(); renderNameSelect();
}
function setBusy(on) { $$('button.add, #generate, #refresh').forEach(b => b.disabled = on); }
function renderJob(j) {
  const bar = $('#jobbar');
  if (!j || (!j.running && !j.error && !j.result)) { bar.hidden = true; return; }
  bar.hidden = false;
  $('#jobtext').textContent = j.running ? (j.kind === 'add' ? 'adding names — ' : 'refreshing — ') + (j.progress || '') : (j.error ? 'failed: ' + j.error.split('\n')[0] : 'done');
  $('#jobticks').innerHTML = Object.entries(j.tickers || {}).map(([s, st]) => `<span class="tick ${st}">${s} · ${st}</span>`).join('');
  $('.spin', bar).style.visibility = j.running ? 'visible' : 'hidden';
  if (!j.running) setTimeout(() => { if (!$('#jobbar').dataset.running) bar.hidden = true; }, 6000);
}
async function pollJob() {
  const j = await api(`/screens/${S.sid}/job`);
  setBusy(!!j.running); renderJob(j); $('#jobbar').dataset.running = j.running ? '1' : '';
  $('#jobStatus').textContent = j.running ? `working… ${j.progress || ''}` : (j.error ? 'failed: ' + j.error.split('\n')[0] : '');
  if (j.result) $('#addResult').innerHTML = (j.result.added.length ? `<span class="pos">added ${j.result.added.join(', ')}</span>` : '') + (j.result.rejected.length ? `<div class="rej">rejected: ${j.result.rejected.join(' · ')}</div>` : '');
  if (j.running) setTimeout(pollJob, 1200); else { S.screen = await api(`/screens/${S.sid}`); renderScreenTab(); await loadBook(); }
}

/* ---------- header ---------- */
function renderHeader() {
  const b = S.book; if (!b) { $('#asof').textContent = ''; $('#strip').innerHTML = ''; return; }
  $('#asof').textContent = `as of ${b.as_of || '–'}` + (b.inception ? ` · live since ${b.inception}` : ' · not generated') + (b.session && b.session.forming ? ' · session open, next close pending' : '');
  const lv = b.book.live, last = lv.dates.length - 1, s = lv.summary;
  $('#strip').innerHTML = last < 0 ? '<span>live book is flat</span>' :
    `<span>long <b>${fmt(lv.long[last], 1)}</b></span><span>short <b>${fmt(lv.short[last], 1)}</b></span><span>net <b class="${sgn(lv.net[last])}">${fmt(lv.net[last], 1)}</b></span>` +
    `<span>today <b class="${sgn(lv.pnl[last])}">${fmt(lv.pnl[last], 3)}</b></span><span>since inception <b class="${sgn(s.total)}">${fmt(s.total, 2)}</b> u${s.sharpe != null ? ` · Sharpe <b>${fmt(s.sharpe)}</b>` : ''}</span>`;
}

/* ---------- today ---------- */
function renderToday() {
  const b = S.book, U = S.screen.unit_dollars;
  const side = (host, sideVal) => {
    const acts = b ? b.today.actions.filter(a => a.side === sideVal) : [];
    if (!b || !b.inception) { host.innerHTML = '<div class="empty">Generate the screen to start the live book.</div>'; return; }
    if (!acts.length) {
      const names = b.sides[String(sideVal)].names.length;
      host.innerHTML = `<div class="empty">No actions today across ${names} name${names === 1 ? '' : 's'}. ${sideVal > 0 ? 'The rule buys on the next close more than ' + fmt(b.sides['1'].strategy.z_in, 1) + 'σ below the band.' : 'The rule shorts on the next close more than ' + fmt(b.sides['-1'].strategy.z_in, 1) + 'σ above the band.'}</div>`;
      return;
    }
    host.innerHTML = acts.map(a => `<div class="action ${a.action.toLowerCase()}"><span class="verb">${a.action}</span><span class="sym">${a.symbol}</span><span class="why">${fmt(a.size, 2)} u · ${a.reason}</span><span class="usd">${usd(a.size * U)}</span></div>`).join('');
  };
  side($('#todayLong'), 1); side($('#todayShort'), -1);
  const t = $('#posTable');
  if (!b || !b.today.positions.length) { t.innerHTML = '<tbody><tr><td class="muted">no open positions in the live book</td></tr></tbody>'; return; }
  t.innerHTML = '<thead><tr><th>symbol</th><th>side</th><th>units</th><th>notional</th><th>avg entry</th><th>last</th><th>unreal (u)</th><th>oldest</th><th>armed</th><th>z</th></tr></thead><tbody>' +
    b.today.positions.map(p => `<tr class="click" data-sym="${p.symbol}" data-side="${p.side}"><td>${p.symbol}</td><td>${p.side > 0 ? 'long' : 'short'}</td><td>${fmt(p.units, 2)}</td><td>${usd(p.units * U)}</td><td>${fmt(p.avg_px)}</td><td>${fmt(p.px)}</td><td class="${sgn(p.unreal)}">${fmt(p.unreal, 3)}</td><td>${p.oldest}</td><td>${p.armed ? '<span class="badge">armed</span>' : ''}</td><td>${fmt(p.z)}</td></tr>`).join('') + '</tbody>';
  $$('tr.click', t).forEach(r => r.addEventListener('click', () => { S.nameSide[r.dataset.sym] = +r.dataset.side; $('#nameSearch').value = ''; renderNameSelect(r.dataset.sym + '|' + r.dataset.side); showTab('name'); }));
}

/* ---------- screen tab ---------- */
function renderScreenTab() {
  const s = S.screen, st = s.state;
  const chips = (host, sideKey) => {
    const names = st.names[sideKey];
    host.innerHTML = Object.entries(names).map(([sym, m]) => `<span class="chip ${m.dropped ? 'dropped' : ''}" title="added ${m.added}${m.dropped ? ', dropped ' + m.dropped : ''}">${sym}${s.symbols && s.symbols[sym] && s.symbols[sym].progress ? ' <span class="badge">' + s.symbols[sym].progress + '</span>' : ''}${m.dropped ? '' : `<button title="drop ${sym}" data-sym="${sym}" data-side="${sideKey}">×</button>`}</span>`).join('') || '<span class="muted small">none yet</span>';
    $$('button', host).forEach(b => b.addEventListener('click', async () => { if (!confirm(`Drop ${b.dataset.sym}? Its tracked position is liquidated at the next close.`)) return; await api(`/screens/${S.sid}/names/${b.dataset.sym}?side=${b.dataset.side}`, { method: 'DELETE' }); await selectScreen(S.sid); }));
  };
  chips($('#listLong'), '1'); chips($('#listShort'), '-1');
  const presets = S.presets.presets;
  for (const [sel, sideKey] of [[$('#presetLong'), '1'], [$('#presetShort'), '-1']]) {
    const side = +sideKey, cfg = s.strategy[sideKey];
    sel.innerHTML = Object.entries(presets).filter(([k, p]) => p.side === side).map(([k]) => `<option value="${k}">${k}</option>`).join('') + '<option value="custom">custom</option>';
    sel.value = cfg.preset || 'custom';
    renderParams(sideKey, cfg);
  }
  $('#unitDollars').value = s.unit_dollars;
  $('#generate').textContent = s.generated ? 'Regenerate (refresh data)' : 'Generate';
  $('#eventLog').innerHTML = '<h3>Event log</h3>' + s.events.slice().reverse().map(e => `<div class="muted">${e.date} · ${e.kind}${e.symbols ? ' ' + e.symbols.join(', ') : ''}${e.symbol ? ' ' + e.symbol : ''}${e.side != null ? ' (' + (e.side > 0 ? 'long' : 'short') + ')' : ''}${e.cfg ? ' → ' + (e.cfg.preset && e.cfg.preset !== 'custom' ? e.cfg.preset : 'custom: ' + ['band','h','z_in','z_out','z_norm','max_hold','sizing','ratio','max_stack','max_units','capitulate'].map(k => k + '=' + (e.cfg[k] == null ? 'none' : e.cfg[k])).join(' ')) : ''}</div>`).join('');
}
const PARAM_OPTS = { band: ['field', 'brownian'], sizing: ['fixed', 'decel', 'accel'] };
const PARAM_SKIP = new Set(['side', 'preset', 'kind']);
function renderParams(sideKey, cfg) {
  const host = sideKey === '1' ? $('#paramsLong') : $('#paramsShort');
  const kindNote = cfg.kind === 'confirm' ? '<div class="muted small" style="grid-column:1/-1">confirmation short: break above the band (z_break) → first close back below z_enter → short <b>start</b> units; each close <b>step</b>σ lower adds <b>accel</b>× the last; cover when z &gt; z_exit. accel &lt; 1 = decelerating stack.</div>' : '';
  host.innerHTML = kindNote + Object.entries(cfg).filter(([k]) => !PARAM_SKIP.has(k)).map(([k, val]) => {
    if (PARAM_OPTS[k]) return `<label>${k}<select data-k="${k}">${PARAM_OPTS[k].map(o => `<option ${val === o ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`;
    if (typeof val === 'boolean') return `<label>${k}<select data-k="${k}" data-bool="1"><option value="false" ${!val ? 'selected' : ''}>false</option><option value="true" ${val ? 'selected' : ''}>true</option></select></label>`;
    return `<label>${k}<input data-k="${k}" type="number" step="any" value="${val == null ? '' : val}" placeholder="none"></label>`;
  }).join('');
  $$('[data-k]', host).forEach(inp => inp.addEventListener('change', async () => {
    const params = {}; $$('[data-k]', host).forEach(i => { params[i.dataset.k] = i.value === '' ? null : (i.dataset.bool ? i.value === 'true' : (i.tagName === 'SELECT' ? i.value : +i.value)); });
    S.screen = await api(`/screens/${S.sid}/strategy`, { method: 'PUT', body: { side: +sideKey, params } }); S.screen.state = (await api(`/screens/${S.sid}`)).state;
    renderScreenTab(); await loadBook();
  }));
}
async function setStrategy(side, preset) {
  const body = preset === 'custom' ? { side, params: {} } : { side, preset };
  await api(`/screens/${S.sid}/strategy`, { method: 'PUT', body }); await selectScreen(S.sid);
}
async function addNames(side) {
  const inp = side > 0 ? $('#addLong') : $('#addShort'); const syms = inp.value.split(/[,\s]+/).filter(Boolean);
  if (!syms.length) return;
  try { $('#addResult').innerHTML = ''; await api(`/screens/${S.sid}/names`, { method: 'POST', body: { side, symbols: syms } }); inp.value = ''; setBusy(true); pollJob(); }
  catch (e) { $('#addResult').innerHTML = `<span class="rej">${e.message}</span>`; }
}

/* ---------- book tab ---------- */
function tiles(host, bt, lv) {
  const B = bt.summary, L = lv.summary, has = lv.dates.length > 0;
  host.innerHTML = [
    ['book Sharpe', fmt(B.sharpe), `live ${has ? fmt(L.sharpe) : '–'}`],
    ['total (u)', fmt(B.total, 1), `live ${has ? fmt(L.total, 2) : '–'}`],
    ['long / short (u)', `${fmt(B.total_long, 1)} / ${fmt(B.total_short, 1)}`, `live ${has ? fmt(L.total_long, 2) + ' / ' + fmt(L.total_short, 2) : '–'}`],
    ['on peak gross', pct(B.ret_peak), `peak ${fmt(B.peak_gross, 1)} u`],
    ['on day‑wt gross', pct(B.ret_daywt), `avg ${fmt(B.avg_gross, 1)} u`],
    ['avg net', fmt(B.avg_net, 1), `live ${has ? fmt(L.avg_net, 1) : '–'}`],
    ['max DD (u)', fmt(B.max_dd, 1), `live ${has ? fmt(L.max_dd, 2) : '–'}`],
    ['days', B.days, `live ${L.days}`],
  ].map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
}
function frame(W, H, m, dates) {
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}` }); const n = dates.length; const x = i => m.l + (W - m.l - m.r) * (i + .5) / n;
  dates.forEach((d, i) => { if (i === 0 || d.slice(5, 7) !== dates[i - 1].slice(5, 7)) { const t = el('text', { x: x(i), y: H - m.b + 15 }, svg); t.textContent = d.slice(0, 7); el('line', { x1: x(i), x2: x(i), y1: m.t, y2: H - m.b, stroke: 'var(--grid)' }, svg); } });
  return { svg, x, n };
}
function hover(svg, W, H, m, x, n, html) {
  const hit = el('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent' }, svg); const cross = el('line', { y1: m.t, y2: H - m.b, stroke: 'var(--ink2)', opacity: 0 }, svg);
  hit.addEventListener('mousemove', e => { const r = svg.getBoundingClientRect(); const px = (e.clientX - r.left) * W / r.width; let i = Math.round((px - m.l) / (W - m.l - m.r) * n - .5); i = Math.max(0, Math.min(n - 1, i)); cross.setAttribute('x1', x(i)); cross.setAttribute('x2', x(i)); cross.setAttribute('opacity', .6); showTip(e, html(i)); });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('opacity', 0); hideTip(); });
}
function yTicks(svg, W, m, y, mn, mx, d = 1) { for (let k = 0; k <= 4; k++) { const v = mn + (mx - mn) * k / 4; el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--grid)' }, svg); const t = el('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg); t.textContent = v.toFixed(d); } }
function renderBook() {
  const b = S.book; if (!b) { $('#bookTiles').innerHTML = '<div class="empty">Generate the screen first.</div>'; $('#expoChart').innerHTML = ''; $('#eqChart').innerHTML = ''; $('#nameTable').innerHTML = ''; return; }
  const bt = b.book.backtest, lv = b.book.live; tiles($('#bookTiles'), bt, lv);
  const dates = bt.dates, n = dates.length, li = lv.dates.length ? dates.indexOf(lv.dates[0]) : -1;
  const lvAt = (arr, i) => { const j = i - li; return (li >= 0 && j >= 0 && j < arr.length) ? arr[j] : null; };
  // exposure
  { const W = 1200, H = 320, m = { l: 50, r: 14, t: 12, b: 26 }; const { svg, x } = frame(W, H, m, dates);
    const mx = Math.max(1, ...bt.long, ...lv.long), mn = -Math.max(1, ...bt.short, ...lv.short); const y = v => m.t + (H - m.t - m.b) * (1 - (v - mn) / (mx - mn));
    yTicks(svg, W, m, y, mn, mx, 0);
    el('polygon', { points: `${x(0)},${y(0)} ` + bt.long.map((v, i) => `${x(i)},${y(v)}`).join(' ') + ` ${x(n - 1)},${y(0)}`, fill: 'var(--long)', 'fill-opacity': .18 }, svg);
    el('polygon', { points: `${x(0)},${y(0)} ` + bt.short.map((v, i) => `${x(i)},${y(-v)}`).join(' ') + ` ${x(n - 1)},${y(0)}`, fill: 'var(--short)', 'fill-opacity': .18 }, svg);
    el('polyline', { points: bt.net.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--bt)', 'stroke-width': 1.4 }, svg);
    if (li >= 0) { el('line', { x1: x(li), x2: x(li), y1: m.t, y2: H - m.b, stroke: 'var(--lv)', 'stroke-dasharray': '4 3' }, svg);
      el('polyline', { points: lv.long.map((v, j) => `${x(li + j)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--long)', 'stroke-width': 2 }, svg);
      el('polyline', { points: lv.short.map((v, j) => `${x(li + j)},${y(-v)}`).join(' '), fill: 'none', stroke: 'var(--short)', 'stroke-width': 2 }, svg);
      el('polyline', { points: lv.net.map((v, j) => `${x(li + j)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--lv)', 'stroke-width': 2 }, svg); }
    hover(svg, W, H, m, x, n, i => `<b>${dates[i]}</b><br>backtest long ${fmt(bt.long[i], 1)} · short ${fmt(bt.short[i], 1)} · net ${fmt(bt.net[i], 1)}` + (lvAt(lv.net, i) != null ? `<br>live long ${fmt(lvAt(lv.long, i), 2)} · short ${fmt(lvAt(lv.short, i), 2)} · net ${fmt(lvAt(lv.net, i), 2)}` : ''));
    $('#expoChart').innerHTML = ''; $('#expoChart').appendChild(svg); }
  // equity
  { const W = 1200, H = 300, m = { l: 50, r: 14, t: 12, b: 26 }; const { svg, x } = frame(W, H, m, dates);
    const all = [...bt.equity, ...bt.eq_long, ...bt.eq_short, ...lv.equity]; const mn = Math.min(0, ...all), mx = Math.max(.01, ...all); const y = v => m.t + (H - m.t - m.b) * (1 - (v - mn) / (mx - mn));
    yTicks(svg, W, m, y, mn, mx, 1); el('line', { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), stroke: 'var(--line)' }, svg);
    el('polyline', { points: bt.eq_long.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--long)', 'stroke-width': 1, 'stroke-opacity': .6 }, svg);
    el('polyline', { points: bt.eq_short.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--short)', 'stroke-width': 1, 'stroke-opacity': .6 }, svg);
    el('polyline', { points: bt.equity.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--bt)', 'stroke-width': 1.6 }, svg);
    if (li >= 0) { el('line', { x1: x(li), x2: x(li), y1: m.t, y2: H - m.b, stroke: 'var(--lv)', 'stroke-dasharray': '4 3' }, svg);
      el('polyline', { points: lv.equity.map((v, j) => `${x(li + j)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--lv)', 'stroke-width': 2.2 }, svg); }
    hover(svg, W, H, m, x, n, i => `<b>${dates[i]}</b><br>backtest ${fmt(bt.equity[i], 2)} (L ${fmt(bt.eq_long[i], 2)} · S ${fmt(bt.eq_short[i], 2)})` + (lvAt(lv.equity, i) != null ? `<br>live ${fmt(lvAt(lv.equity, i), 3)}` : ''));
    $('#eqChart').innerHTML = ''; $('#eqChart').appendChild(svg); }
  // table
  const rows = [];
  for (const [sym, sides] of Object.entries(b.names)) for (const [sideKey, e] of Object.entries(sides)) { const B = e.backtest || {}, L = e.live || {}; rows.push({ sym, side: +sideKey, B, L, dropped: e.dropped }); }
  rows.sort((a, c) => (c.B.total_ret || 0) - (a.B.total_ret || 0));
  $('#nameTable').innerHTML = '<thead><tr><th>symbol</th><th>side</th><th>bt trades</th><th>bt total</th><th>bt on peak</th><th>bt Sharpe</th><th>B&amp;H</th><th>live trades</th><th>live total</th><th>live max DD</th><th></th></tr></thead><tbody>' +
    rows.map(r => `<tr class="click" data-sym="${r.sym}" data-side="${r.side}"><td>${r.sym}</td><td>${r.side > 0 ? 'long' : 'short'}</td><td>${r.B.trades ?? '–'}</td><td class="${sgn(r.B.total_ret)}">${fmt(r.B.total_ret, 3)}</td><td class="${sgn(r.B.ret_peak)}">${pct(r.B.ret_peak)}</td><td class="${sgn(r.B.sharpe)}">${fmt(r.B.sharpe)}</td><td class="${sgn(r.B.bh_ret)}">${pct(r.B.bh_ret)}</td><td>${r.L.trades ?? '–'}</td><td class="${sgn(r.L.total_ret)}">${r.L.total_ret == null ? '–' : fmt(r.L.total_ret, 3)}</td><td>${r.L.max_dd == null ? '–' : fmt(r.L.max_dd, 3)}</td><td>${r.dropped ? '<span class="badge warn">dropped ' + r.dropped + '</span>' : ''}</td></tr>`).join('') + '</tbody>';
  $$('tr.click', $('#nameTable')).forEach(r => r.addEventListener('click', () => { $('#nameSearch').value = ''; renderNameSelect(r.dataset.sym + '|' + r.dataset.side); showTab('name'); }));
}

/* ---------- name tab ---------- */
function nameOptions() {
  const b = S.book; if (!b) return [];
  const opts = []; for (const [sym, sides] of Object.entries(b.names)) for (const sideKey of Object.keys(sides)) opts.push({ v: sym + '|' + sideKey, sym, side: +sideKey, dropped: !!sides[sideKey].dropped });
  // longs A→Z, then shorts A→Z
  return opts.sort((a, c) => (c.side - a.side) || a.sym.localeCompare(c.sym));
}
function renderNameSelect(keep) {
  const sel = $('#nameSel'), q = ($('#nameSearch').value || '').trim().toUpperCase();
  const all = nameOptions(), opts = q ? all.filter(o => o.sym.includes(q)) : all;
  const cur = keep || sel.value;
  sel.innerHTML = opts.map(o => `<option value="${o.v}">${o.sym} · ${o.side > 0 ? 'long' : 'short'}${o.dropped ? ' (dropped)' : ''}</option>`).join('');
  if (!opts.length) { $('#nameMeta').textContent = all.length ? 'no match' : ''; return; }
  sel.value = opts.some(o => o.v === cur) ? cur : opts[0].v;
  if (sel.value !== S.lastName) loadName(sel.value);
}
$('#nameSearch').addEventListener('input', () => renderNameSelect());
$('#nameSearch').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); const sel = $('#nameSel'); if (sel.options.length) loadName(sel.value); } });
async function loadName(v) {
  if (!v) return; S.lastName = v; const [sym, side] = v.split('|'); let d; try { d = await api(`/screens/${S.sid}/series/${sym}?side=${side}`); } catch (e) { $('#nameChart').innerHTML = `<div class="empty">${e.message}</div>`; return; }
  const U = S.screen.unit_dollars, n = d.dates.length, dates = d.dates; const bt = d.backtest, lv = d.live;
  $('#nameMeta').textContent = `${bt ? bt.trades.length + ' backtest trades' : ''}${lv ? ' · ' + lv.trades.length + ' live trades since ' + d.inception + ' · open ' + fmt(lv.open.reduce((a, o) => a + o.size, 0), 2) + ' u' : ''}`;
  // period + trade statistics per ledger
  const idx0 = Object.fromEntries(dates.map((dt, i) => [dt, i]));
  const ledgerStats = (L, label) => {
    const cls = label === 'live' ? 'lv' : 'bt';
    if (!L || !L.dates.length) return `<div class="tile ${cls}"><div class="k">${label}</div><div class="v">–</div><div class="s">not started</div></div>`;
    const a = idx0[L.dates[0]], b2 = idx0[L.dates[L.dates.length - 1]];
    const priceRet = (d.close[b2] / d.close[a] - 1) * (+side);
    const tr = L.trades; const sumPnl = tr.reduce((s, t) => s + t.pnl, 0);
    const peak = Math.max(0, ...L.capital); const first = tr.length ? tr.reduce((m, t) => t.entry < m ? t.entry : m, tr[0].entry) : null, last = tr.length ? tr.reduce((m, t) => t.exit > m ? t.exit : m, tr[0].exit) : null;
    // day-weighted return over the trade span: P&L / average capital deployed between first entry and last exit
    const li = Object.fromEntries(L.dates.map((dt, i) => [dt, i])); const s0 = first ? li[first] : 0, s1 = last ? li[last] : L.dates.length - 1;
    const span = L.capital.slice(s0, s1 + 1); const avgCap = span.length ? span.reduce((a, c) => a + c, 0) / span.length : 0;
    const openPnl = L.open.reduce((s, o) => s + o.size * (Math.log(d.close[b2]) - Math.log(d.close[idx0[o.date]])) * (+side), 0);
    const eq = L.equity; let pk = -Infinity, dd = 0; for (const e of eq) { pk = Math.max(pk, e); dd = Math.max(dd, pk - e); }
    const dp = eq.map((e, i) => i ? e - eq[i - 1] : e); const mu = dp.reduce((s, x) => s + x, 0) / dp.length; const sd = Math.sqrt(dp.reduce((s, x) => s + (x - mu) ** 2, 0) / Math.max(dp.length - 1, 1)); const sharpe = sd > 0 && dp.length > 1 ? mu / sd * Math.sqrt(252) : null;
    return `<div class="tile ${cls}"><div class="k">${label} · period</div><div class="v">${L.dates.length}d</div><div class="s">${L.dates[0]} → ${L.dates[L.dates.length - 1]}</div></div>
      <div class="tile ${cls}"><div class="k">${label} · price return</div><div class="v ${sgn(priceRet)}">${pct(priceRet)}</div><div class="s">${+side > 0 ? 'holding' : 'short-and-holding'} over the period</div></div>
      <div class="tile ${cls}"><div class="k">${label} · trade P&amp;L</div><div class="v ${sgn(sumPnl)}">${fmt(sumPnl, 3)} u</div><div class="s">${tr.length} closed trades${L.open.length ? ` · open ${fmt(openPnl, 3)} u` : ''}</div></div>
      <div class="tile ${cls}"><div class="k">${label} · cumulative trade return</div><div class="v ${sgn(sumPnl)}">${avgCap > 0 ? pct(sumPnl / avgCap) : '–'}</div><div class="s">${first ? first + ' → ' + last : '–'} · on ${fmt(avgCap, 2)} u avg deployed</div></div>
      <div class="tile ${cls}"><div class="k">${label} · on peak capital</div><div class="v ${sgn(sumPnl)}">${peak > 0 ? pct(sumPnl / peak) : '–'}</div><div class="s">peak ${fmt(peak, 2)} u deployed</div></div>
      <div class="tile ${cls}"><div class="k">${label} · Sharpe</div><div class="v ${sgn(sharpe)}">${fmt(sharpe)}</div><div class="s">daily P&amp;L, annualised</div></div>
      <div class="tile ${cls}"><div class="k">${label} · max drawdown</div><div class="v neg">${fmt(dd, 3)} u</div><div class="s">${peak > 0 ? pct(dd / peak) + ' of peak capital' : '–'}</div></div>`;
  };
  $('#nameTiles').innerHTML = ledgerStats(bt, 'backtest') + ledgerStats(lv, 'live');
  const W = 1200, H = 520, m = { l: 56, r: 60, t: 12, b: 26 }, PH = 300, CH = 60, EH = 70, gap = 14; const { svg, x } = frame(W, H, m, dates);
  const lo = Math.min(...d.close, ...d.q10), hi = Math.max(...d.close, ...d.q90); const y = v => m.t + PH * (1 - (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)));
  el('polygon', { points: d.q90.map((v, i) => `${x(i)},${y(v)}`).concat(d.q10.map((v, i) => `${x(i)},${y(v)}`).reverse()).join(' '), fill: 'var(--gravsoft)', stroke: 'var(--grav)', 'stroke-width': .8, 'stroke-opacity': .6 }, svg);
  for (let k = 0; k <= 4; k++) { const v = Math.exp(Math.log(lo) + (Math.log(hi) - Math.log(lo)) * k / 4); el('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: 'var(--grid)' }, svg); const t = el('text', { x: m.l - 6, y: y(v) + 4, 'text-anchor': 'end' }, svg); t.textContent = v.toFixed(0); }
  el('polyline', { points: d.close.map((v, i) => `${x(i)},${y(v)}`).join(' '), fill: 'none', stroke: 'var(--ink)', 'stroke-width': 1.5, 'stroke-linejoin': 'round' }, svg);
  const idx = Object.fromEntries(dates.map((dt, i) => [dt, i]));
  const drawTrades = (tr, filled) => { for (const t of tr) { const a = idx[t.entry], b2 = idx[t.exit]; if (a == null || b2 == null) continue;
    el('line', { x1: x(a), y1: y(t.entry_px), x2: x(b2), y2: y(t.exit_px), stroke: t.pnl > 0 ? 'var(--good)' : 'var(--bad)', 'stroke-opacity': filled ? .5 : .25 }, svg);
    const r = 3 + 4 * Math.sqrt(t.size); const e1 = el('circle', { cx: x(a), cy: y(t.entry_px), r, fill: filled ? 'var(--good)' : 'var(--panel)', stroke: 'var(--good)', 'stroke-width': 1.5 }, svg); const e2 = el('circle', { cx: x(b2), cy: y(t.exit_px), r: 3.5, fill: filled ? 'var(--bad)' : 'var(--panel)', stroke: 'var(--bad)', 'stroke-width': 1.5 }, svg);
    const html = `${filled ? 'LIVE' : 'backtest'} ${t.side > 0 ? 'long' : 'short'}<br>in ${t.entry} @ ${fmt(t.entry_px)} · ${fmt(t.size, 2)} u (${usd(t.size * U)})<br>out ${t.exit} @ ${fmt(t.exit_px)} · ${t.reason}<br>held ${t.held}d · ${pct(t.ret)} · P&L ${fmt(t.pnl, 3)} u`; for (const c of [e1, e2]) { c.addEventListener('mousemove', ev => showTip(ev, html)); c.addEventListener('mouseleave', hideTip); } } };
  if (bt) drawTrades(bt.trades, false); if (lv) { drawTrades(lv.trades, true); for (const o of lv.open) { const a = idx[o.date]; if (a == null) continue; el('circle', { cx: x(a), cy: y(d.close[a]), r: 3 + 4 * Math.sqrt(o.size), fill: 'var(--good)', stroke: 'var(--panel)', 'stroke-width': 1.5 }, svg); } }
  if (d.inception && idx[d.inception] != null) el('line', { x1: x(idx[d.inception]), x2: x(idx[d.inception]), y1: m.t, y2: H - m.b, stroke: 'var(--lv)', 'stroke-dasharray': '4 3' }, svg);
  for (const vd of (d.version_dates || [])) { const k = dates.findIndex(dt => dt >= vd); if (k < 0) continue; el('line', { x1: x(k), x2: x(k), y1: m.t, y2: H - m.b, stroke: 'var(--lv)', 'stroke-dasharray': '1 3' }, svg); const t = el('text', { x: x(k) + 4, y: m.t + 12 }, svg); t.textContent = 'strategy → ' + vd; t.style.fill = 'var(--lv)'; }
  // capital + equity strips (backtest muted, live bold)
  const strip = (y0, h, label, btArr, lvArr, lvStart, fmtv) => { const lab = el('text', { x: m.l, y: y0 - 3 }, svg); lab.textContent = label; const all = [...(btArr || []), ...(lvArr || [])]; const mn = Math.min(0, ...all), mx = Math.max(.01, ...all); const yy = v => y0 + h * (1 - (v - mn) / (mx - mn));
    for (const tv of [mn, (mn + mx) / 2, mx]) { el('line', { x1: m.l, x2: W - m.r, y1: yy(tv), y2: yy(tv), stroke: tv === 0 ? 'var(--line)' : 'var(--grid)' }, svg); const t = el('text', { x: m.l - 6, y: yy(tv) + 4, 'text-anchor': 'end' }, svg); t.textContent = fmtv(tv); }
    if (mn < 0 && mx > 0) el('line', { x1: m.l, x2: W - m.r, y1: yy(0), y2: yy(0), stroke: 'var(--line)' }, svg);
    if (btArr) el('polyline', { points: btArr.map((v, i) => `${x(i)},${yy(v)}`).join(' '), fill: 'none', stroke: 'var(--bt)', 'stroke-width': 1.2 }, svg);
    if (lvArr && lvStart != null) el('polyline', { points: lvArr.map((v, j) => `${x(lvStart + j)},${yy(v)}`).join(' '), fill: 'none', stroke: 'var(--lv)', 'stroke-width': 2 }, svg);
    const lastV = lvArr && lvArr.length ? lvArr[lvArr.length - 1] : (btArr ? btArr[btArr.length - 1] : 0); const t = el('text', { x: W - m.r + 6, y: yy(lastV) + 4 }, svg); t.textContent = fmtv(lastV); t.style.fill = lvArr && lvArr.length ? 'var(--lv)' : 'var(--bt)'; };
  const btStart = bt ? idx[bt.dates[0]] : null, lvStart = lv ? idx[lv.dates[0]] : null;
  const pad = (arr, start, fill = 0) => { if (!arr || start == null) return null; const out = new Array(n).fill(fill); arr.forEach((v, j) => { if (start + j < n) out[start + j] = v; }); return out; };
  const btCap = pad(bt && bt.capital, btStart), btEq = pad(bt && bt.equity, btStart), lvCap = pad(lv && lv.capital, lvStart, null), lvEq = pad(lv && lv.equity, lvStart, null);
  strip(m.t + PH + gap + 10, CH, 'capital deployed (u)', btCap, lv && lv.capital, lvStart, v => fmt(v, 2));
  strip(m.t + PH + gap + 10 + CH + gap + 10, EH, 'equity (u)', btEq, lv && lv.equity, lvStart, v => fmt(v, 3));
  hover(svg, W, H, m, x, n, i => `<b>${dates[i]}</b> close ${fmt(d.close[i])} · z ${fmt(d.z[i])}<br>band ${fmt(d.q10[i], 1)} – ${fmt(d.q50[i], 1)} – ${fmt(d.q90[i], 1)}` +
    (btCap ? `<br><span style="color:var(--bt)">backtest</span> capital ${fmt(btCap[i], 2)} · equity ${fmt(btEq[i], 3)}` : '') +
    (lvCap && lvCap[i] != null ? `<br><span style="color:var(--lv)">live</span> capital ${fmt(lvCap[i], 2)} · equity ${fmt(lvEq[i], 3)}` : ''));
  $('#nameChart').innerHTML = ''; $('#nameChart').appendChild(svg);
  const chrono = (a, b2) => a.exit.localeCompare(b2.exit) || a.entry.localeCompare(b2.entry);
  const withCum = (trs, ledger) => { let c = 0; return trs.slice().sort(chrono).map(t => ({ ...t, ledger, cum: (c += t.pnl) })); };
  S.trades = [...withCum(lv ? lv.trades : [], 'live'), ...withCum(bt ? bt.trades : [], 'backtest')];
  S.tradeDir = S.tradeDir || 'desc';
  renderTrades();
}
function renderTrades() {
  const dir = S.tradeDir; const chrono = (a, b2) => a.exit.localeCompare(b2.exit) || a.entry.localeCompare(b2.entry);
  const tr = S.trades.slice().sort((a, b2) => dir === 'desc' ? chrono(b2, a) : chrono(a, b2));
  $('#tradeTable').innerHTML = `<thead><tr><th>ledger</th><th class="sortable" title="click to flip order">entry → exit ${dir === 'desc' ? '▼' : '▲'}</th><th>size</th><th>entry px</th><th>exit px</th><th>held</th><th>return</th><th>P&amp;L (u)</th><th>cum P&amp;L (u)</th><th>reason</th></tr></thead><tbody>` +
    tr.map(t => `<tr class="${t.ledger === 'live' ? 'lv' : 'bt'}"><td><span class="ledger ${t.ledger === 'live' ? 'lv' : 'bt'}">${t.ledger}</span></td><td class="mono">${t.entry} → ${t.exit}</td><td>${fmt(t.size, 2)}</td><td>${fmt(t.entry_px)}</td><td>${fmt(t.exit_px)}</td><td>${t.held}</td><td class="${sgn(t.ret)}">${pct(t.ret)}</td><td class="${sgn(t.pnl)}">${fmt(t.pnl, 3)}</td><td class="${sgn(t.cum)}"><b>${fmt(t.cum, 3)}</b></td><td>${t.reason}</td></tr>`).join('') + '</tbody>';
  $('th.sortable', $('#tradeTable')).addEventListener('click', () => { S.tradeDir = S.tradeDir === 'desc' ? 'asc' : 'desc'; renderTrades(); });
}

boot().catch(e => { console.error(e); alert('boot failed: ' + e.message); });
