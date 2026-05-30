/* ── DOM refs ─────────────────────────────────────────────────────────────── */
const dropZone      = document.getElementById('dropZone');
const photoInput    = document.getElementById('photoInput');
const previewWrap   = document.getElementById('previewWrap');
const previewCanvas = document.getElementById('previewCanvas');
const detectBtn     = document.getElementById('detectBtn');
const resultSection = document.getElementById('resultSection');
const facesGrid     = document.getElementById('facesGrid');
const faceCountBadge= document.getElementById('faceCountBadge');
const errorBox      = document.getElementById('errorBox');

const feedbackCount = document.getElementById('feedbackCount');
const retrainBtn    = document.getElementById('retrainBtn');
const retrainResult = document.getElementById('retrainResult');

/* ── State ────────────────────────────────────────────────────────────────── */
let selectedFile  = null;
let originalImage = null;
let lastFaces     = [];   // array of face result objects from /detect

/* ── Keypoint draw order ──────────────────────────────────────────────────── */
const KP_ORDER = [
  'left_eye_outer','left_eye_inner','left_eye_top','left_eye_bottom',
  'right_eye_inner','right_eye_outer','right_eye_top','right_eye_bottom',
  'mouth_left','mouth_right','mouth_top_ctr','mouth_bot_ctr',
  'mouth_top_left','mouth_top_right','lower_lip',
];

const FACE_COLORS = [
  { dot: '#6366f1', mouth: '#f59e0b' },
  { dot: '#ec4899', mouth: '#f97316' },
  { dot: '#10b981', mouth: '#facc15' },
  { dot: '#3b82f6', mouth: '#a78bfa' },
  { dot: '#f43f5e', mouth: '#34d399' },
];

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
      drawPreview(img, []);
      previewWrap.hidden = false;
      detectBtn.disabled = false;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/* ── Draw preview with keypoints for all faces ────────────────────────────── */
function drawPreview(img, faces) {
  const MAX_W = previewWrap.clientWidth || 700;
  const scale = Math.min(1, MAX_W / img.naturalWidth);
  previewCanvas.width  = img.naturalWidth  * scale;
  previewCanvas.height = img.naturalHeight * scale;

  const ctx = previewCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0, previewCanvas.width, previewCanvas.height);

  if (!faces || faces.length === 0) return;

  const eyeNames   = ['left_eye_outer','left_eye_inner','left_eye_top','left_eye_bottom',
                       'right_eye_inner','right_eye_outer','right_eye_top','right_eye_bottom'];
  const mouthNames = ['mouth_left','mouth_right','mouth_top_ctr','mouth_bot_ctr',
                      'mouth_top_left','mouth_top_right','lower_lip'];

  faces.forEach((face, i) => {
    if (!face.keypoints || Object.keys(face.keypoints).length === 0) return;
    const palette = FACE_COLORS[i % FACE_COLORS.length];

    // Draw eye dots
    ctx.fillStyle   = palette.dot;
    ctx.strokeStyle = '#fff';
    ctx.lineWidth   = 1;
    for (const name of eyeNames) {
      const pt = face.keypoints[name];
      if (!pt) continue;
      ctx.beginPath();
      ctx.arc(pt[0] * scale, pt[1] * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Draw mouth dots
    ctx.fillStyle = palette.mouth;
    for (const name of mouthNames) {
      const pt = face.keypoints[name];
      if (!pt) continue;
      ctx.beginPath();
      ctx.arc(pt[0] * scale, pt[1] * scale, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }

    // Connect mouth corners with a curve
    const ml = face.keypoints['mouth_left'];
    const mr = face.keypoints['mouth_right'];
    const mt = face.keypoints['mouth_top_ctr'];
    if (ml && mr && mt) {
      ctx.beginPath();
      ctx.moveTo(ml[0] * scale, ml[1] * scale);
      ctx.quadraticCurveTo(mt[0] * scale, mt[1] * scale, mr[0] * scale, mr[1] * scale);
      ctx.strokeStyle = palette.mouth;
      ctx.lineWidth   = 2;
      ctx.stroke();
    }

    // Face index label near the left eye outer corner
    const anchor = face.keypoints['left_eye_outer'];
    if (anchor) {
      ctx.font      = 'bold 13px Segoe UI, sans-serif';
      ctx.fillStyle = palette.dot;
      ctx.strokeStyle = '#000';
      ctx.lineWidth   = 3;
      const label = `Face ${i + 1}`;
      ctx.strokeText(label, anchor[0] * scale, anchor[1] * scale - 10);
      ctx.fillText(label,   anchor[0] * scale, anchor[1] * scale - 10);
    }
  });
}

/* ── Detect button ────────────────────────────────────────────────────────── */
detectBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  const spinner = showSpinner();
  hideError();
  hideResult();
  lastFaces = [];

  const formData = new FormData();
  formData.append('photo', selectedFile);

  try {
    const res  = await fetch('/detect', { method: 'POST', body: formData });
    const data = await res.json();
    removeSpinner(spinner);

    if (data.error) { showError(data.error); return; }

    lastFaces = data.faces;
    renderResults(data);
  } catch (err) {
    removeSpinner(spinner);
    showError('Network error: ' + err.message);
  }
});

/* ── Render all face results ──────────────────────────────────────────────── */
function renderResults(data) {
  // Update face count badge
  faceCountBadge.textContent = `${data.face_count} face${data.face_count !== 1 ? 's' : ''} detected`;
  faceCountBadge.hidden = false;

  // Draw all keypoints on canvas
  if (originalImage) drawPreview(originalImage, data.faces);

  // Build per-face cards
  facesGrid.innerHTML = '';
  data.faces.forEach((face, i) => {
    const palette = FACE_COLORS[i % FACE_COLORS.length];
    facesGrid.appendChild(buildFaceCard(face, i, palette));
  });

  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Build a single face result card ─────────────────────────────────────── */
function buildFaceCard(face, i, palette) {
  const card = document.createElement('div');
  card.className = 'face-card';
  card.style.setProperty('--face-color', palette.dot);

  // ── Header ──
  const header = document.createElement('div');
  header.className = 'face-card-header';

  const title = document.createElement('span');
  title.className = 'face-card-title';
  title.textContent = `Face ${i + 1}`;
  title.style.color = palette.dot;

  header.appendChild(title);

  if (face.error) {
    const errMsg = document.createElement('p');
    errMsg.className = 'face-card-error';
    errMsg.textContent = '⚠️ ' + face.error;
    card.appendChild(header);
    card.appendChild(errMsg);
    return card;
  }

  // ── Badge ──
  const badge = document.createElement('div');
  badge.className = `face-badge ${face.smile ? 'smile' : 'no-smile'}`;
  badge.textContent = face.smile ? '😄 SMILING' : '😐 NOT SMILING';

  // ── Warning ──
  if (face.warning) {
    const warn = document.createElement('div');
    warn.className = 'face-warning';
    warn.textContent = '⚠️ ' + face.warning;
    card.appendChild(header);
    card.appendChild(badge);
    card.appendChild(warn);
  } else {
    card.appendChild(header);
    card.appendChild(badge);
  }

  // ── Details grid ──
  const details = document.createElement('div');
  details.className = 'face-details';

  // Confidence
  const pct = Math.round(face.confidence * 100);
  const confCard = document.createElement('div');
  confCard.className = 'face-detail-card';
  confCard.innerHTML = `
    <h3>Confidence</h3>
    <div class="confidence-bar-wrap">
      <div class="confidence-bar" style="width:${pct}%; background:${
        face.low_confidence
          ? 'linear-gradient(90deg,#f59e0b,#fcd34d)'
          : `linear-gradient(90deg,${palette.dot},#a78bfa)`
      }"></div>
    </div>
    <span>${pct}%</span>
  `;

  // Face size
  const px = face.inter_eye_px;
  const sizeColor = px < 50 ? '#f87171' : px < 80 ? '#fcd34d' : '#4ade80';
  const sizeHint  = px < 50 ? '⚠ Too small' : px < 80 ? 'Acceptable — closer is better' : '✓ Good face size';
  const sizeCard = document.createElement('div');
  sizeCard.className = 'face-detail-card';
  sizeCard.innerHTML = `
    <h3>Face Size</h3>
    <p class="face-size-label">Inter-eye distance</p>
    <p class="face-size-value" style="color:${sizeColor}">${px} px</p>
    <p class="face-size-hint">${sizeHint}</p>
  `;

  // Features
  const featureLabels = {
    mouth_w_eye_span_ratio:  'Mouth width / eye span',
    mouth_openness:          'Mouth openness',
    eye_to_mouth_face_ratio: 'Eye-to-mouth / face height',
  };
  const featCard = document.createElement('div');
  featCard.className = 'face-detail-card';
  featCard.innerHTML = `
    <h3>Facial Features</h3>
    <table>${Object.entries(face.features)
      .map(([k, v]) => `<tr><td>${featureLabels[k] || k}</td><td>${v}</td></tr>`)
      .join('')}</table>
  `;

  details.appendChild(confCard);
  details.appendChild(sizeCard);
  details.appendChild(featCard);
  card.appendChild(details);

  // ── Feedback ──
  const fbBar = document.createElement('div');
  fbBar.className = 'feedback-bar';

  const fbLabel = document.createElement('span');
  fbLabel.className = 'feedback-label';
  fbLabel.textContent = 'Was this correct?';

  const btnYes = document.createElement('button');
  btnYes.className = 'btn-feedback btn-yes';
  btnYes.textContent = '👍 Yes';

  const btnNo = document.createElement('button');
  btnNo.className = 'btn-feedback btn-no';
  btnNo.textContent = '👎 No';

  const fbStatus = document.createElement('span');
  fbStatus.className = 'feedback-status';

  btnYes.addEventListener('click', () => sendFeedback(face, true,  btnYes, btnNo, fbStatus));
  btnNo.addEventListener('click',  () => sendFeedback(face, false, btnYes, btnNo, fbStatus));

  fbBar.appendChild(fbLabel);
  fbBar.appendChild(btnYes);
  fbBar.appendChild(btnNo);
  fbBar.appendChild(fbStatus);
  card.appendChild(fbBar);

  return card;
}

/* ── Feedback ─────────────────────────────────────────────────────────────── */
async function sendFeedback(face, userSaysCorrect, btnYes, btnNo, fbStatus) {
  if (!face.keypoints) return;

  const kpList = KP_ORDER.map(name => face.keypoints[name]);
  const correctLabel = userSaysCorrect ? face.smile : !face.smile;

  btnYes.disabled = true;
  btnNo.disabled  = true;
  fbStatus.textContent = 'Saving…';

  try {
    const res  = await fetch('/feedback', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ keypoints: kpList, correct_label: correctLabel }),
    });
    const data = await res.json();

    if (data.error) {
      fbStatus.textContent = '⚠ ' + data.error;
    } else {
      fbStatus.textContent = '✓ Saved';
      feedbackCount.textContent = data.feedback_count;
    }
  } catch (err) {
    fbStatus.textContent = '⚠ Network error';
  }
}

/* ── Retrain ──────────────────────────────────────────────────────────────── */
retrainBtn.addEventListener('click', async () => {
  retrainBtn.disabled    = true;
  retrainBtn.textContent = 'Retraining…';
  retrainResult.hidden   = true;
  retrainResult.className = 'retrain-result';

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
function hideResult() {
  resultSection.hidden = true;
  faceCountBadge.hidden = true;
}

function showSpinner() {
  const el = document.createElement('div');
  el.className = 'spinner-overlay';
  el.innerHTML = '<div class="spinner"></div>';
  document.body.appendChild(el);
  return el;
}
function removeSpinner(el) { el?.remove(); }
