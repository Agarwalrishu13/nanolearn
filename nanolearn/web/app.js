/* nanoLearn front end — plain JavaScript, no framework, no build step.
   Five steps: give it a file, say what to predict, let it practise, see the
   result, use it. */

'use strict';

const $ = (id) => document.getElementById(id);
const api = {
  async get(url) {
    const response = await fetch(url);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  },
  async post(url, body) {
    const response = await fetch(url, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  },
};

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function toast(message, kind) {
  const node = document.createElement('div');
  node.className = 'toast' + (kind ? ' ' + kind : '');
  node.textContent = message;
  $('toasts').appendChild(node);
  setTimeout(() => {
    node.style.opacity = '0';
    node.style.transition = 'opacity .3s';
    setTimeout(() => node.remove(), 300);
  }, kind === 'bad' ? 10000 : 5000);
}

async function stream(url, body, onEvent) {
  const response = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let split;
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of raw.split('\n')) {
        if (!line.startsWith('data:')) continue;
        const payload = line.slice(5).trim();
        if (payload === '[DONE]') return;
        try { onEvent(JSON.parse(payload)); } catch (err) { /* ignore partial */ }
      }
    }
  }
}

// ---------------------------------------------------------------------- state
const state = {
  status: null,
  dataset: null,     // the /api/inspect shape
  target: '',
  task: 'classification',
  job: null,
  poll: null,
  result: null,
  model: null,
  step: 'data',
};

const STEPS = ['data', 'understand', 'learn', 'result', 'try'];

function setStep(step) {
  state.step = step;
  for (const name of STEPS) {
    $('panel-' + name).hidden = name !== step;
  }
  document.querySelectorAll('.step').forEach((button) => {
    const name = button.dataset.step;
    button.classList.toggle('active', name === step);
    const reached = STEPS.indexOf(name) <= STEPS.indexOf(step);
    button.classList.toggle('done', reached && name !== step);
    button.disabled = !state.dataset && name !== 'data';
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ------------------------------------------------------------------ step one
async function loadStatus() {
  try {
    state.status = await api.get('/api/status');
  } catch (err) {
    toast('The app stopped answering. Restart it.', 'bad');
    return;
  }
  const full = state.status.full_engine || {};
  $('enginePill').textContent = full.available
    ? `Full engine ready (scikit-learn ${full.version})`
    : 'Built-in engine';
  $('enginePill').className = 'pill ' + (full.available ? 'good' : 'warn');
  $('installBtn').style.display = full.available ? 'none' : '';
  $('engineHelp').textContent = full.available
    ? `The full engine is installed (scikit-learn ${full.version}). nanoLearn is using it automatically.`
    : 'The built-in engine needs nothing installed. The full engine (scikit-learn) is optional and makes '
      + 'some predictions better; press “Install full engine” to add it.';
  $('aboutInfo').textContent = `Your files are kept in ${state.status.data_folder} — uploaded spreadsheets, `
    + 'trained models, reports and guesses. Nothing is ever sent anywhere.';
  renderSamples(state.status.samples || []);
  await loadFiles();
  $('testFraction').value = state.status.settings.test_fraction;
  $('seedInput').value = state.status.settings.seed;
}

function renderSamples(samples) {
  const holder = $('sampleList');
  holder.innerHTML = '';
  for (const sample of samples) {
    const button = document.createElement('button');
    button.className = 'sample-item';
    button.innerHTML = `<span class="info"><b>${escapeHtml(sample.title)}</b>
      <span>${escapeHtml(sample.blurb)}</span></span><span>→</span>`;
    button.onclick = () => loadSample(sample.id);
    holder.appendChild(button);
  }
}

async function loadFiles() {
  const holder = $('fileList');
  holder.innerHTML = '';
  let data;
  try { data = await api.get('/api/files'); } catch (err) { return; }
  if (!data.files.length) {
    holder.innerHTML = '<p class="muted" style="font-size:13.5px">Nothing yet. Files you add will be listed here.</p>';
    return;
  }
  for (const file of data.files.slice(0, 8)) {
    const button = document.createElement('button');
    button.className = 'sample-item';
    button.innerHTML = `<span class="info"><b>${escapeHtml(file.name)}</b>
      <span>${file.size_mb} MB</span></span><span>→</span>`;
    button.onclick = () => inspect(file.path);
    holder.appendChild(button);
  }
}

async function loadSample(id) {
  try {
    const data = await api.post('/api/samples/load', { id });
    acceptDataset(data, data.message);
  } catch (err) { toast(err.message, 'bad'); }
}

async function inspect(path) {
  try {
    const data = await api.post('/api/inspect', { path });
    acceptDataset(data);
  } catch (err) { toast(err.message, 'bad'); }
}

// ------------------------------------------------------------------ step two
function acceptDataset(data, message) {
  state.dataset = data;
  state.target = data.target_guess.name || '';
  state.task = data.target_guess.task || 'classification';
  if (message) toast(message, 'good');

  $('fileName').textContent = data.name;
  const summary = data.summary;
  $('summary').textContent =
    `${summary.rows} rows and ${summary.columns} columns. ` +
    `${summary.numbers} number column${summary.numbers === 1 ? '' : 's'}, ` +
    `${summary.categories} with categories, ${summary.text} of free text.` +
    (summary.missing_cells
      ? ` ${summary.missing_cells} cells are empty (${summary.missing_pct}%) — that is handled automatically.`
      : ' No empty cells.');

  const stats = [
    ['Rows', summary.rows], ['Columns', summary.columns],
    ['Numbers', summary.numbers], ['Categories', summary.categories],
    ['Empty cells', summary.missing_cells],
  ];
  $('stats').innerHTML = stats.map(([k, v]) => `<div class="stat"><div class="v">${escapeHtml(v)}</div><div class="k">${escapeHtml(k)}</div></div>`).join('');

  // Target picker
  const select = $('targetSelect');
  select.innerHTML = '';
  for (const column of data.columns) {
    const option = document.createElement('option');
    option.value = column.name;
    option.textContent = `${column.name}  (${column.kind === 'number' ? 'numbers' : column.kind === 'category' ? 'categories' : column.kind})`;
    select.appendChild(option);
  }
  select.value = state.target;
  $('taskSelect').value = state.task;
  $('guessReason').textContent = data.target_guess.reason;

  renderPreview(data.preview);
  renderColumns(data.columns);
  renderChart(state.target);
  setStep('understand');
}

function renderPreview(preview) {
  const head = preview.headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('');
  const body = preview.rows.map((row) =>
    `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('');
  $('preview').innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderColumns(columns) {
  const holder = $('columnList');
  holder.innerHTML = '';
  for (const column of columns) {
    const chip = document.createElement('button');
    chip.className = 'column-chip' + (column.name === state.target ? ' active' : '');
    const kind = column.kind === 'number' ? 'numbers' : column.kind === 'category' ? 'categories' : column.kind;
    chip.innerHTML = `${escapeHtml(column.name)}<em>${escapeHtml(kind)}</em>`;
    chip.onclick = () => {
      state.target = column.name;
      $('targetSelect').value = column.name;
      $('taskSelect').value = column.kind === 'number' ? 'regression' : 'classification';
      renderChart(column.name);
      renderColumns(columns);
    };
    holder.appendChild(chip);
  }
}

/* A small SVG bar chart: horizontal bars, because labels read better that way. */
function renderChart(columnName) {
  const spec = (state.dataset && state.dataset.charts && state.dataset.charts[columnName]) || null;
  const holder = $('chart');
  if (!spec || spec.kind === 'empty') {
    holder.innerHTML = '<p class="muted" style="font-size:13px">No chart for this column.</p>';
    return;
  }
  const rows = spec.labels.map((label, index) => ({ label, value: spec.values[index] || 0 }));
  const biggest = Math.max(...rows.map((row) => row.value)) || 1;
  const rowHeight = 26;
  const labelWidth = 150;
  const chartWidth = 620;
  const barSpace = chartWidth - labelWidth - 60;

  const bars = rows.map((row, index) => {
    const y = index * rowHeight;
    const width = Math.max(2, (row.value / biggest) * barSpace);
    return `<text x="0" y="${y + 15}">${escapeHtml(shorten(row.label, 20))}</text>
      <rect class="bar" x="${labelWidth}" y="${y + 4}" width="${width}" height="${rowHeight - 10}" rx="4"></rect>
      <text x="${labelWidth + width + 7}" y="${y + 15}">${row.value}</text>`;
  }).join('');

  holder.innerHTML = `
    <div class="caption"><b>${escapeHtml(columnName)}</b> — ${escapeHtml(spec.caption || '')}</div>
    <svg viewBox="0 0 ${chartWidth} ${Math.max(rows.length * rowHeight, 30)}" role="img"
         aria-label="Chart of ${escapeHtml(columnName)}">
      <defs><linearGradient id="barFill" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="#3ddc97" stop-opacity="0.85"></stop>
        <stop offset="100%" stop-color="#1f9c74" stop-opacity="0.75"></stop>
      </linearGradient></defs>
      ${bars}
    </svg>`;
}

function shorten(text, limit) {
  const value = String(text);
  return value.length > limit ? value.slice(0, limit - 1) + '…' : value;
}

// ---------------------------------------------------------------- step three
async function teach() {
  if (!state.dataset) return;
  state.target = $('targetSelect').value;
  state.task = $('taskSelect').value;
  setStep('learn');
  $('learnTitle').textContent = `Teaching it to predict “${state.target}”…`;
  $('learnLog').textContent = '';
  $('learnBar').style.width = '0%';

  try {
    const started = await api.post('/api/train', {
      path: state.dataset.path,
      target: state.target,
      task: state.task,
      engine: $('useFull').checked ? 'full' : 'auto',
    });
    state.job = started.job_id;
    pollJob();
  } catch (err) {
    toast(err.message, 'bad');
    setStep('understand');
  }
}

function pollJob() {
  clearInterval(state.poll);
  state.poll = setInterval(async () => {
    let payload;
    try {
      payload = await api.get('/api/job/' + encodeURIComponent(state.job));
    } catch (err) {
      clearInterval(state.poll);
      toast(err.message, 'bad');
      setStep('understand');
      return;
    }
    $('learnBar').style.width = payload.progress + '%';
    const log = $('learnLog');
    log.textContent = (payload.logs || []).join('\n');
    log.scrollTop = log.scrollHeight;

    if (payload.state === 'done') {
      clearInterval(state.poll);
      $('learnBar').style.width = '100%';
      state.result = payload.result;
      showResult(payload.result);
    } else if (payload.state === 'error') {
      clearInterval(state.poll);
      $('learnTitle').textContent = 'That did not work';
      $('learnSub').textContent = payload.error || 'Something went wrong.';
      toast(payload.error || 'Training failed.', 'bad');
      setTimeout(() => setStep('understand'), 1400);
    }
  }, 500);
}

// ----------------------------------------------------------------- step four
function showResult(result) {
  const task = result.task;
  if (task === 'classification') {
    $('scoreBig').textContent = `${result.metrics.correct_pct}% correct`;
    $('resultSentence').textContent =
      `${result.explanation} (Always answering “${mostCommonGuess(result)}” would have scored `
      + `${result.baseline}%.)`;
  } else {
    $('scoreBig').textContent = `Typical error ${result.metrics.mae}`;
    $('resultSentence').textContent = result.explanation;
  }

  const leaderboard = result.leaderboard || [];
  $('leaderboard').innerHTML = leaderboard.length
    ? `<table><thead><tr><th>Approach</th><th>Score</th></tr></thead><tbody>${
        leaderboard.map((row) => {
          const score = row.plain
            || Object.entries(row.score || {}).map(([k, v]) => `${k} ${v}`).join(', ');
          return `<tr class="${row.is_best ? 'best' : ''}"><td>${escapeHtml(row.name)}</td>
            <td>${escapeHtml(score)}</td></tr>`;
        }).join('')}</tbody></table>`
    : '<p class="muted">No comparison for this run.</p>';

  const importance = (result.importance || []).filter((row) => row.weight > 0).slice(0, 10);
  const biggest = importance.length ? importance[0].weight : 1;
  $('importance').innerHTML = importance.length
    ? importance.map((row) => `
        <div class="importance-row">
          <div class="top"><span>${escapeHtml(row.name)}</span><span>${Math.round((row.weight || 0) * 100)}%</span></div>
          <div class="bar"><span style="width:${Math.max(2, 100 * row.weight / biggest)}%"></span></div>
        </div>`).join('')
    : '<p class="muted">This model cannot say which columns mattered.</p>';

  const examples = result.examples || [];
  if (examples.length) {
    const classification = task === 'classification';
    const head = classification
      ? '<th>Truth</th><th>Its guess</th><th>Confidence</th><th></th>'
      : '<th>Truth</th><th>Its guess</th><th>Off by</th>';
    const body = examples.map((row) => classification
      ? `<tr><td>${escapeHtml(row.actual)}</td><td>${escapeHtml(row.predicted)}</td>
         <td class="num">${row.confidence != null ? row.confidence + '%' : '—'}</td>
         <td class="${row.correct ? 'yes' : 'no'}">${row.correct ? 'right' : 'wrong'}</td></tr>`
      : `<tr><td class="num">${escapeHtml(row.actual)}</td><td class="num">${escapeHtml(row.predicted)}</td>
         <td class="num">${escapeHtml(row.difference)}</td></tr>`).join('');
    $('examples').innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  } else {
    $('examples').innerHTML = '<p class="muted">Not enough rows to hold any back for testing.</p>';
  }

  const dropped = result.dropped || [];
  $('droppedCard').hidden = dropped.length === 0;
  if (dropped.length) {
    $('droppedReason').textContent = 'These columns cannot help predict anything, so they were excluded. '
      + 'Keeping them in would make the score look better than it really is.';
    $('droppedList').innerHTML = '<div class="column-list">'
      + dropped.map((item) => `<span class="column-chip">${escapeHtml(item)}</span>`).join('')
      + '</div>';
  }
  setStep('result');
}

function mostCommonGuess(result) {
  const classes = result.classes || [];
  return classes.length ? classes[0] : 'the same thing';
}

// ------------------------------------------------------------------ step five
async function showTry() {
  setStep('try');
  $('predictResult').innerHTML = '';
  $('answer').hidden = true;
  let model;
  try {
    model = await api.get('/api/model');
  } catch (err) { return; }
  state.model = model;
  if (!model.ready) {
    $('tryTitle').textContent = 'Nothing has been taught yet';
    $('predictForm').innerHTML = '<p class="muted">Go back to step 2 and press “Teach it” first.</p>';
    return;
  }
  $('tryTitle').textContent = `Ask it about a new case — ${model.target}`;

  const form = $('predictForm');
  form.innerHTML = '';
  for (const field of model.fields) {
    const wrap = document.createElement('div');
    wrap.className = 'field';
    const label = document.createElement('label');
    label.textContent = field.name;
    label.htmlFor = 'field-' + field.name;
    wrap.appendChild(label);

    let input;
    if (field.type === 'category' && field.categories.length) {
      input = document.createElement('select');
      for (const category of field.categories) {
        const option = document.createElement('option');
        option.value = category;
        option.textContent = category;
        input.appendChild(option);
      }
    } else {
      input = document.createElement('input');
      input.type = 'number';
      input.step = 'any';
      if (field.min != null || field.max != null) {
        input.placeholder = `${field.min ?? '?'} to ${field.max ?? '?'}`;
      }
    }
    input.id = 'field-' + field.name;
    input.dataset.field = field.name;
    wrap.appendChild(input);
    form.appendChild(wrap);
  }
  $('modelInfo').innerHTML = `
    <p><b>Answering:</b> ${escapeHtml(model.target)} (${model.task === 'classification' ? 'a category' : 'a number'})<br>
    <b>Engine:</b> ${escapeHtml(model.engine)}<br>
    <b>Columns it uses:</b> ${model.fields.length}</p>
    <p class="muted" style="font-size:13px">A saved copy of this model is in your nanoLearn folder, so you can
    come back to it later without teaching again.</p>`;
}

async function predict() {
  const values = {};
  document.querySelectorAll('#predictForm [data-field]').forEach((input) => {
    values[input.dataset.field] = input.value;
  });
  try {
    const data = await api.post('/api/predict', { values });
    const answer = data.answer || {};
    const box = $('answer');
    box.hidden = false;
    let html = `<div class="label">Its answer</div>`;
    if (data.task === 'classification') {
      html += `<div class="big">${escapeHtml(answer.label ?? '—')}</div>
        <p class="muted">${escapeHtml(data.sentence)}</p>`;
      if (answer.probabilities && answer.probabilities.length) {
        html += '<div class="prob">' + answer.probabilities.map((item) => `
          <div class="prob-row"><span>${escapeHtml(item.label)}</span>
            <div class="prob-bar"><span style="width:${item.percent}%"></span></div>
            <span class="num">${item.percent}%</span></div>`).join('') + '</div>';
      }
    } else {
      html += `<div class="big">${escapeHtml(answer.value ?? '—')}</div>
        <p class="muted">${escapeHtml(data.sentence)}</p>`;
    }
    box.innerHTML = html;
  } catch (err) { toast(err.message, 'bad'); }
}

async function predictFile(file) {
  const holder = $('predictResult');
  holder.innerHTML = '<p class="muted">Reading the file…</p>';
  try {
    const saved = await uploadFile(file);
    const data = await api.post('/api/predict/file', { path: saved.path || saved.file_path || '' });
    const preview = data.preview;
    holder.innerHTML = `<p>${escapeHtml(data.sentence)}</p>
      <div class="table-wrap" style="margin-bottom:12px"><table><thead><tr>${
        preview.headers.map((h) => `<th>${escapeHtml(h)}</th>`).join('')}</tr></thead><tbody>${
        preview.rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`).join('')
      }</tbody></table></div>
      <a class="btn primary" href="${escapeHtml(data.download)}" download>Download the results (${data.count} rows)</a>`;
  } catch (err) {
    holder.innerHTML = '';
    toast(err.message, 'bad');
  }
}

// ------------------------------------------------------------------- uploads
async function uploadFile(file, onProgress) {
  const start = await api.post('/api/upload/start', { name: file.name });
  const chunkSize = start.chunk_bytes || (4 * 1024 * 1024);
  let offset = 0;
  while (offset < file.size) {
    const slice = file.slice(offset, Math.min(offset + chunkSize, file.size));
    const response = await fetch(`/api/upload/chunk?id=${encodeURIComponent(start.id)}&offset=${offset}`,
      { method: 'POST', body: slice });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || 'The transfer stopped.');
    }
    offset += slice.size;
    if (onProgress) onProgress(Math.round(100 * offset / file.size));
  }
  const done = await api.post('/api/upload/finish', { id: start.id });
  done.file_path = done.path;
  return done;
}

// ------------------------------------------------------------------- report
async function downloadReport() {
  try {
    const data = await api.post('/api/report', { path: state.dataset ? state.dataset.path : '' });
    window.location.href = data.download;
    toast('Report downloaded — open it in any browser.', 'good');
  } catch (err) { toast(err.message, 'bad'); }
}

async function installFullEngine() {
  const log = $('installLog');
  log.textContent = '';
  $('settingsModal').classList.add('open');
  toast('Installing — this downloads about 100 MB.');
  try {
    await stream('/api/install', {}, (event) => {
      if (event.type === 'log') { log.textContent += event.message + '\n'; log.scrollTop = log.scrollHeight; }
      else if (event.type === 'error') { log.textContent += event.message + '\n'; toast(event.message, 'bad'); }
      else if (event.type === 'done') { log.textContent += event.message + '\n'; toast(event.message, 'good'); }
    });
    await loadStatus();
  } catch (err) { toast(err.message, 'bad'); log.textContent += err.message + '\n'; }
}

// --------------------------------------------------------------------- wiring
function wireDrop(element, handler) {
  ['dragenter', 'dragover'].forEach((name) => element.addEventListener(name, (event) => {
    event.preventDefault();
    element.classList.add('over');
  }));
  ['dragleave', 'drop'].forEach((name) => element.addEventListener(name, (event) => {
    event.preventDefault();
    element.classList.remove('over');
  }));
  element.addEventListener('drop', (event) => {
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length) handler(files[0]);
  });
  element.addEventListener('click', (event) => {
    if (event.target.closest('button, a, input')) return;
    $('fileInput').click();
  });
}

function wire() {
  wireDrop($('drop'), async (file) => {
    toast(`Reading ${file.name}…`);
    try {
      const saved = await uploadFile(file);
      acceptDataset(saved, saved.message);
    } catch (err) { toast(err.message, 'bad'); }
  });
  wireDrop($('dropPredict'), (file) => predictFile(file));

  $('browseBtn').onclick = (event) => { event.stopPropagation(); $('fileInput').click(); };
  $('fileInput').onchange = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    toast(`Reading ${file.name}…`);
    try {
      const saved = await uploadFile(file);
      acceptDataset(saved, saved.message);
    } catch (err) { toast(err.message, 'bad'); }
  };

  $('targetSelect').onchange = (event) => {
    state.target = event.target.value;
    const column = (state.dataset.columns || []).find((item) => item.name === state.target);
    if (column) $('taskSelect').value = column.kind === 'number' ? 'regression' : 'classification';
    renderChart(state.target);
    renderColumns(state.dataset.columns);
  };
  $('teachBtn').onclick = teach;
  $('useItBtn').onclick = showTry;
  $('reportBtn').onclick = downloadReport;
  $('retryBtn').onclick = () => setStep('understand');
  $('predictBtn').onclick = predict;
  $('clearBtn').onclick = () => {
    document.querySelectorAll('#predictForm [data-field]').forEach((input) => {
      if (input.tagName === 'SELECT') input.selectedIndex = 0; else input.value = '';
    });
    $('answer').hidden = true;
  };
  $('installBtn').onclick = installFullEngine;
  $('settingsBtn').onclick = () => $('settingsModal').classList.add('open');

  document.querySelectorAll('.step').forEach((button) => {
    button.onclick = () => {
      const name = button.dataset.step;
      if (name === 'data' || state.dataset) setStep(name);
    };
  });
  document.querySelectorAll('[data-close]').forEach((button) => {
    button.onclick = async () => {
      const id = button.dataset.close;
      $(id).classList.remove('open');
      if (id === 'settingsModal') {
        try {
          state.status.settings = await api.post('/api/settings', {
            test_fraction: Number($('testFraction').value) || 0.2,
            seed: Number($('seedInput').value) || 0,
          });
        } catch (err) { toast(err.message, 'bad'); }
      }
    };
  });
  document.querySelectorAll('.backdrop').forEach((backdrop) => {
    backdrop.addEventListener('click', (event) => {
      if (event.target === backdrop) backdrop.classList.remove('open');
    });
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      document.querySelectorAll('.backdrop.open').forEach((backdrop) => backdrop.classList.remove('open'));
    }
  });
}

(async function main() {
  wire();
  await loadStatus();
  setStep('data');
})();
