/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById('dropZone');
const photoInput     = document.getElementById('photoInput');
const previewWrap    = document.getElementById('previewWrap');
const previewCanvas  = document.getElementById('previewCanvas');
const detectBtn      = document.getElementById('detectBtn');
const resultSection  = document.getElementById('resultSection');
const resultBadge    = document.getElementById('resultBadge');
const warningBox     = document.getElementById('warningBox');
const confidenceBar  = document.getElementById('confidenceBar');
const confidenceText = document.getElementById('confidenceText');
const interEyeText   = document.getElementById('interEyeText');
const faceSizeHint   = document.getElementById('faceSizeHint');
const featuresTable  = document.getElementById('featuresTable');
const keypointsTable = document.getElementById('keypointsTable');
const errorBox       = document.getElementById('errorBox');

const feedbackYes    = document.getElementById('feedbackYes');
const feedbackNo     = document.getElementById('feedbackNo');
const feedbackStatus = document.getElementById('feedbackStatus');

const feedbackCount  = document.getElementById('feedbackCount');
const retrainBtn     = document.getElementById('retrainBtn');
const retrainResult  = document.getElementById('retrainResult');

/* ── State ────────────────────────────────────────────────────────────────── */
let selectedFile  = null;
let originalImage = null;
let lastResult    = null;   // last /detect response (holds keypoints for feedback)

/* ── Init: load feedback count ───────────────────────────────────────────── */
(async () => {
  try {
    const res  = await fetch('/stats');
    const data = await res.json();
    feedbackCount.textContent = data.feedback_count;
  } catch (_) {}
})();

/* ── Drag & drop ──────────────────────────────────────────────────────────── */
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
photoInput.addEventListener('change', () => {
  if (photoInput.files[0]) handleFile(photoInput.files[0]);
});

/* ── File handling ────────────────────────────────────────────────────────── */
function handleFile(file) {
  selectedFile = file;
  hideError();
  hideResult();

  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      originalImage = img;
      drawPreview(img, null);
      previewWrap.hidden = false;
      detectBtn.disabled = false;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/* ── Draw preview (with optional keypoints overlay) ──────────────────────── */
function drawPreview(img, keypoints) {
  const MAX_W = previewWrap.clientWidth || 700;
  const scale = Math.min(1, MAX_W / img.naturalWidth);
  previewCanvas.width  = img.naturalWidth  * scale;
  previewCanvas.height = img.naturalHeight * scale;

  const ctx = previewCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);

  if (!keypoints) return;

  const groups = {
    eye:   ['left_eye_outer','left_eye_inner','left_eye_top','left_eye_bottom',
            'right_eye_inner','right_eye_outer','right_eye_top','right_eye_bottom'],
    mouth: ['mouth_left','mouth_right','mouth_top_ctr','mouth_bot_ctr',
            'mouth_top_left','mouth_top_right','lower_lip'],
  };
  const colors = { eye: '#6366f1', mouth: '#f59e0b' };

  for (const [region, names] of Object.entries(groups)) {
    ctx.fillStyle   = colors[region];
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1;
    for (const name of names) {
      const pt = keypoints[name];
      if (!pt) continue;
      const [x, y] = pt;
      ctx.beginPath();
      ctx.arc(x * scale, y * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }

  // Connect mouth corners
  const ml = keypoints['mouth_left'];
  const mr = keypoints['mouth_right'];
  const mt = keypoints['mouth_top_ctr'];
  if (ml && mr && mt) {
    ctx.beginPath();
    ctx.moveTo(ml[0] * scale, ml[1] * scale);
    ctx.quadraticCurveTo(mt[0] * scale, mt[1] * scale, mr[0] * scale, mr[1] * scale);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth   = 2;
    ctx.stroke();
  }
}

/* ── Detect button ────────────────────────────────────────────────────────── */
detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  const spinner = showSpinner();
  hideError();
  hideResult();
  lastResult = null;

  const formData = new FormData();
  formData.append('photo', selectedFile);

  try {
    const res  = await fetch('/detect', { method: 'POST', body: formData });
    const data = await res.json();
    removeSpinner(spinner);

    if (data.error) { showError(data.error); return; }

    lastResult = data;
    renderResult(data);
  } catch (err) {
    removeSpinner(spinner);
    showError('Network error: ' + err.message);
  }
});

/* ── Render result ────────────────────────────────────────────────────────── */
function renderResult(data) {
  // Badge
  if (data.smile) {
    resultBadge.textContent = '😄 SMILE DETECTED — TRUE';
    resultBadge.className   = 'result-badge smile';
  } else {
    resultBadge.textContent = '😐 NO SMILE — FALSE';
    resultBadge.className   = 'result-badge no-smile';
  }

  // Warning
  if (data.warning) {
    warningBox.textContent = '⚠️  ' + data.warning;
    warningBox.hidden = false;
  } else {
    warningBox.hidden = true;
  }

  // Confidence
  const pct = Math.round(data.confidence * 100);
  confidenceBar.style.width      = pct + '%';
  confidenceBar.style.background = data.low_confidence
    ? 'linear-gradient(90deg, #f59e0b, #fcd34d)'
    : 'linear-gradient(90deg, #6366f1, #a78bfa)';
  confidenceText.textContent = pct + '%';

  // Face size
  const px = data.inter_eye_px;
  interEyeText.textContent = px + ' px';
  if (px < 50) {
    interEyeText.style.color = '#f87171';
    faceSizeHint.textContent = '⚠ Too small — move closer for better accuracy';
  } else if (px < 80) {
    interEyeText.style.color = '#fcd34d';
    faceSizeHint.textContent = 'Acceptable — closer is better';
  } else {
    interEyeText.style.color = '#4ade80';
    faceSizeHint.textContent = '✓ Good face size';
  }

  // Features
  const featureLabels = {
    mouth_w_eye_span_ratio:  'Mouth width / eye span',
    mouth_openness:          'Mouth openness',
    eye_to_mouth_face_ratio: 'Eye-to-mouth / face height',
  };
  featuresTable.innerHTML = Object.entries(data.features)
    .map(([k, v]) => `<tr><td>${featureLabels[k] || k}</td><td>${v}</td></tr>`)
    .join('');

  // Keypoints
  keypointsTable.innerHTML = Object.entries(data.keypoints)
    .map(([name, [x, y]]) =>
      `<tr><td>${name.replace(/_/g, ' ')}</td><td>(${x}, ${y})</td></tr>`)
    .join('');

  // Overlay
  if (originalImage) drawPreview(originalImage, data.keypoints);

  // Reset feedback buttons
  feedbackYes.disabled    = false;
  feedbackNo.disabled     = false;
  feedbackStatus.textContent = '';

  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Feedback ─────────────────────────────────────────────────────────────── */
feedbackYes.addEventListener('click', () => sendFeedback(true));
feedbackNo.addEventListener('click',  () => sendFeedback(false));

async function sendFeedback(userSaysCorrect) {
  if (!lastResult || !lastResult.keypoints) return;

  // Convert named keypoints back to ordered list
  const KP_ORDER = [
    'left_eye_outer','left_eye_inner','left_eye_top','left_eye_bottom',
    'right_eye_inner','right_eye_outer','right_eye_top','right_eye_bottom',
    'mouth_left','mouth_right','mouth_top_ctr','mouth_bot_ctr',
    'mouth_top_left','mouth_top_right','lower_lip',
  ];
  const kpList = KP_ORDER.map(name => lastResult.keypoints[name]);

  // If user says "Yes (correct)", the correct label = what the model predicted.
  // If user says "No (wrong)",    the correct label = opposite of what model predicted.
  const correctLabel = userSaysCorrect ? lastResult.smile : !lastResult.smile;

  feedbackYes.disabled = true;
  feedbackNo.disabled  = true;
  feedbackStatus.textContent = 'Saving…';

  try {
    const res  = await fetch('/feedback', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ keypoints: kpList, correct_label: correctLabel }),
    });
    const data = await res.json();

    if (data.error) {
      feedbackStatus.textContent = '⚠ ' + data.error;
    } else {
      feedbackStatus.textContent = '✓ Saved';
      feedbackCount.textContent  = data.feedback_count;
    }
  } catch (err) {
    feedbackStatus.textContent = '⚠ Network error';
  }
}

/* ── Retrain ──────────────────────────────────────────────────────────────── */
retrainBtn.addEventListener('click', async () => {
  retrainBtn.disabled        = true;
  retrainBtn.textContent     = 'Retraining…';
  retrainResult.hidden       = true;
  retrainResult.className    = 'retrain-result';

  try {
    const res  = await fetch('/retrain', { method: 'POST' });
    const data = await res.json();

    retrainResult.hidden = false;

    if (!data.retrained) {
      retrainResult.classList.add('error');
      retrainResult.textContent = '⚠ ' + data.message;
    } else {
      retrainResult.classList.add('success');
      retrainResult.innerHTML =
        `✓ ${data.message}<br>` +
        `Original training samples: ${data.original_train_samples} &nbsp;|&nbsp; ` +
        `Feedback samples: ${data.feedback_samples} (weight ×${data.feedback_weight})<br>` +
        `Accuracy on original data: ${(data.original_train_accuracy * 100).toFixed(1)}% &nbsp;|&nbsp; ` +
        `Accuracy on feedback: ${(data.feedback_accuracy * 100).toFixed(1)}%`;
    }
  } catch (err) {
    retrainResult.hidden = false;
    retrainResult.classList.add('error');
    retrainResult.textContent = '⚠ Network error: ' + err.message;
  } finally {
    retrainBtn.disabled    = false;
    retrainBtn.textContent = '🔄 Retrain Model';
  }
});

/* ── Helpers ──────────────────────────────────────────────────────────────── */
function showError(msg) {
  errorBox.textContent = '⚠️  ' + msg;
  errorBox.hidden = false;
}
function hideError()  { errorBox.hidden = true; }
function hideResult() { resultSection.hidden = true; }

function showSpinner() {
  const el = document.createElement('div');
  el.className = 'spinner-overlay';
  el.innerHTML = '<div class="spinner"></div>';
  document.body.appendChild(el);
  return el;
}
function removeSpinner(el) { el?.remove(); }
