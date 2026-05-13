/* Lead Intent Predictor — frontend logic */

const API = '';  // same origin as FastAPI

const CIRCUMFERENCE = 2 * Math.PI * 54;  // SVG gauge circle r=54

const FLOAT_FIELDS = new Set([
  'lead_score', 'engagement_score', 'response_time_hours',
  'email_open_rate', 'email_click_rate',
]);
const INT_FIELDS = new Set([
  'company_size', 'page_views', 'time_since_signup_days',
]);
const ALL_FIELDS = [
  'lead_score', 'company_size', 'engagement_score', 'response_time_hours',
  'email_open_rate', 'email_click_rate', 'page_views', 'time_since_signup_days',
  'industry', 'company_type', 'location', 'product_interest', 'source', 'sales_stage',
];

let batchRows    = [];
let batchResults = [];

// ── Initialization ─────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  setInterval(checkHealth, 30_000);
  setupDragDrop();
});

// ── API Health ──────────────────────────────────────────────────────────────

async function checkHealth() {
  const badge = document.getElementById('api-status');
  badge.className = 'status-badge checking';
  try {
    const res  = await fetch(`${API}/health`);
    const data = await res.json();
    const ok   = data.status === 'healthy';
    badge.className = `status-badge ${ok ? 'healthy' : 'unhealthy'}`;
    badge.querySelector('.status-dot').textContent  = '●';
    badge.querySelector('.status-text').textContent = ok
      ? `Live · v${data.version}${data.model_loaded ? ' · Model Ready' : ' · No Model'}`
      : 'Degraded';
  } catch {
    badge.className = 'status-badge unhealthy';
    badge.querySelector('.status-text').textContent = 'Offline';
  }
}

// ── Tabs ────────────────────────────────────────────────────────────────────

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');
}

// ── Single Prediction ───────────────────────────────────────────────────────

async function submitSinglePrediction(e) {
  e.preventDefault();
  const form = e.target;
  const btn  = document.getElementById('predict-btn');

  const payload = {
    lead_score:             parseFloat(form.lead_score.value),
    company_size:           parseInt(form.company_size.value),
    engagement_score:       parseFloat(form.engagement_score.value),
    response_time_hours:    parseFloat(form.response_time_hours.value),
    email_open_rate:        parseFloat(form.email_open_rate.value),
    email_click_rate:       parseFloat(form.email_click_rate.value),
    page_views:             parseInt(form.page_views.value),
    time_since_signup_days: parseInt(form.time_since_signup_days.value),
    industry:               form.industry.value,
    company_type:           form.company_type.value,
    location:               form.location.value,
    product_interest:       form.product_interest.value,
    source:                 form.source.value,
    sales_stage:            form.sales_stage.value,
  };

  setLoading(btn, true, 'Predicting…');

  try {
    const res = await fetch(`${API}/api/predict`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${res.status}`);
    }
    displaySingleResult(await res.json());
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(btn, false, 'Predict Intent');
  }
}

function displaySingleResult(result) {
  const { probability, predicted_label, confidence, model_version, prediction_id } = result;
  const isHigh = predicted_label;
  const pct    = Math.round(probability * 100);

  // Gauge
  const fill = probability * CIRCUMFERENCE;
  const gaugeFill = document.querySelector('.gauge-fill');
  gaugeFill.setAttribute('stroke-dasharray', `${fill} ${CIRCUMFERENCE}`);
  gaugeFill.style.stroke = isHigh ? '#10B981' : '#EF4444';
  document.querySelector('.gauge-pct').textContent = `${pct}%`;

  // Intent badge
  const badge = document.getElementById('intent-badge');
  badge.textContent = isHigh ? 'High Intent' : 'Low Intent';
  badge.className   = `intent-badge ${isHigh ? 'high' : 'low'}`;

  // Details
  document.querySelector('.confidence').textContent    = confidence != null ? `${(confidence * 100).toFixed(1)}%` : 'N/A';
  document.querySelector('.model-version').textContent = model_version  || 'N/A';
  document.querySelector('.pred-id').textContent       = prediction_id  || 'N/A';

  const card = document.getElementById('single-result');
  card.style.display = 'flex';
  card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Batch: file upload ───────────────────────────────────────────────────────

function handleCsvFile(e) {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = ev => parseCsv(ev.target.result);
  reader.readAsText(file);
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) { showError('CSV must have a header row and at least one data row.'); return; }

  const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
  const missing = ALL_FIELDS.filter(f => !headers.includes(f));
  if (missing.length > 0) {
    showError(`Missing columns: ${missing.join(', ')}`);
    return;
  }

  batchRows = lines.slice(1)
    .filter(l => l.trim())
    .map(line => {
      const vals = line.split(',').map(v => v.trim().replace(/^"|"$/g, ''));
      const row  = {};
      headers.forEach((h, i) => {
        const v = vals[i] ?? '';
        if      (FLOAT_FIELDS.has(h)) row[h] = parseFloat(v) || 0;
        else if (INT_FIELDS.has(h))   row[h] = parseInt(v)   || 0;
        else                          row[h] = v;
      });
      return row;
    });

  if (batchRows.length === 0) { showError('No valid data rows found in CSV.'); return; }

  renderBatchPreview();
  document.getElementById('run-batch-btn').disabled = false;
}

function renderBatchPreview() {
  const el = document.getElementById('batch-preview');
  el.style.display = 'block';
  document.getElementById('preview-count').textContent =
    `${batchRows.length} lead${batchRows.length !== 1 ? 's' : ''} loaded`;

  const preview = batchRows.slice(0, 5);
  const more    = batchRows.length - preview.length;

  document.getElementById('preview-table').innerHTML = `
    <thead><tr>${ALL_FIELDS.map(c => `<th>${c}</th>`).join('')}</tr></thead>
    <tbody>
      ${preview.map(r => `<tr>${ALL_FIELDS.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}
      ${more > 0 ? `<tr><td colspan="${ALL_FIELDS.length}" style="text-align:center;color:var(--muted);font-style:italic">… and ${more} more row${more !== 1 ? 's' : ''}</td></tr>` : ''}
    </tbody>`;
}

function clearBatch() {
  batchRows    = [];
  batchResults = [];
  document.getElementById('batch-preview').style.display = 'none';
  document.getElementById('batch-results').style.display = 'none';
  document.getElementById('csv-input').value             = '';
  document.getElementById('run-batch-btn').disabled      = true;
}

// ── Batch: run predictions ───────────────────────────────────────────────────

async function runBatchPredictions() {
  const btn = document.getElementById('run-batch-btn');
  setLoading(btn, true, 'Running…');

  try {
    const res = await fetch(`${API}/api/predict/batch`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(batchRows),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `API error ${res.status}`);
    }
    batchResults = await res.json();
    renderBatchResults();
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(btn, false, 'Run Batch Predictions');
  }
}

function renderBatchResults() {
  const highCount = batchResults.filter(r => r.predicted_label).length;
  const total     = batchResults.length;

  const rows = batchResults.map((r, i) => {
    const pct    = Math.round(r.probability * 100);
    const isHigh = r.predicted_label;
    const color  = isHigh ? '#10B981' : '#EF4444';
    return `
      <tr>
        <td>${i + 1}</td>
        <td class="prob-cell">
          <div class="prob-bar-wrap">
            <div class="prob-bar"><div class="prob-fill" style="width:${pct}%;background:${color}"></div></div>
            <span class="prob-label">${pct}%</span>
          </div>
        </td>
        <td><span class="badge ${isHigh ? 'high' : 'low'}">${isHigh ? 'High' : 'Low'}</span></td>
        <td>${r.confidence != null ? (r.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
        <td>${r.model_version || 'N/A'}</td>
      </tr>`;
  }).join('');

  document.getElementById('results-table').innerHTML = `
    <thead>
      <tr>
        <th>#</th>
        <th>Probability</th>
        <th>Intent</th>
        <th>Confidence</th>
        <th>Model</th>
      </tr>
    </thead>
    <tbody>${rows}</tbody>
    <tfoot>
      <tr>
        <td colspan="2">${total} leads · ${highCount} high intent (${Math.round(highCount / total * 100)}%)</td>
        <td colspan="3"></td>
      </tr>
    </tfoot>`;

  const container = document.getElementById('batch-results');
  container.style.display = 'block';
  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── CSV downloads ────────────────────────────────────────────────────────────

function downloadSampleCsv() {
  const header  = ALL_FIELDS.join(',');
  const samples = [
    '72.5,150,85.0,2.5,0.45,0.12,23,30,Technology,SaaS,US,Enterprise,LinkedIn,Consideration',
    '35.0,50,40.0,12.0,0.20,0.03,5,120,Retail,SMB,EU,SMB,Direct,Awareness',
    '91.0,500,95.0,0.5,0.78,0.35,67,7,Finance,Enterprise,US,Enterprise,Partner,Decision',
    '55.0,200,60.0,6.0,0.30,0.08,12,60,Healthcare,Startup,APAC,Startup,Event,Qualification',
    '18.0,25,22.0,24.0,0.10,0.01,2,300,Retail,SMB,LATAM,SMB,Direct,Awareness',
  ];
  triggerDownload([header, ...samples].join('\n'), 'sample_leads.csv');
}

function downloadResultsCsv() {
  if (!batchResults.length) return;
  const headers = ['row', 'probability', 'predicted_label', 'confidence', 'model_version', 'prediction_id'];
  const rows    = batchResults.map((r, i) =>
    [i + 1, r.probability.toFixed(4), r.predicted_label,
     r.confidence != null ? r.confidence.toFixed(4) : '',
     r.model_version || '', r.prediction_id || ''].join(',')
  );
  triggerDownload([headers.join(','), ...rows].join('\n'), 'batch_predictions.csv');
}

function triggerDownload(content, filename) {
  const blob = new Blob([content], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = Object.assign(document.createElement('a'), { href: url, download: filename });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ── Drag & drop ──────────────────────────────────────────────────────────────

function setupDragDrop() {
  const zone = document.getElementById('upload-zone');
  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith('.csv')) {
      const reader = new FileReader();
      reader.onload = ev => parseCsv(ev.target.result);
      reader.readAsText(file);
    } else {
      showError('Please drop a .csv file.');
    }
  });
}

// ── Utilities ────────────────────────────────────────────────────────────────

function setLoading(btn, loading, label) {
  btn.disabled = loading;
  btn.innerHTML = loading
    ? `<span class="spinner"></span>${label}`
    : label;
}

function showError(msg) {
  document.getElementById('error-msg').textContent = msg;
  const toast = document.getElementById('error-toast');
  toast.style.display = 'flex';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(hideError, 6000);
}

function hideError() {
  document.getElementById('error-toast').style.display = 'none';
}
