/**
 * AI Vision Studio — Frontend JS
 * 13 AI modules, recording, filters, artistic modes, camera switching, analytics.
 */

const API = {
    STATUS: '/api/status', MODULES: '/api/modules', TOGGLE: '/api/toggle/',
    SCREENSHOT: '/api/screenshot', RECORD_START: '/api/record/start',
    RECORD_STOP: '/api/record/stop', FILTERS: '/api/filters',
    ANALYTICS: '/api/analytics', ARTISTIC_SET: '/api/artistic/set',
    ARTISTIC_CYCLE: '/api/artistic/cycle', BG_SET: '/api/bg/set',
    CAMERA_SWITCH: '/api/camera/switch',
};

let modules = [], allModulesActive = false, isRecording = false, chartData = [];

// Toast
function showToast(msg, type = 'success') {
    const icon = type === 'success' ? '✅' : '❌';
    const c = document.getElementById('toast-container');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span class="toast-icon">${icon}</span><span>${msg}</span>`;
    c.appendChild(t);
    setTimeout(() => { if (t.parentNode) t.remove(); }, 3000);
}

// Modules
async function loadModules() {
    try {
        const r = await fetch(API.MODULES);
        modules = await r.json();
        renderModuleCards();
    } catch (e) { console.error(e); }
}

function renderModuleCards() {
    document.getElementById('module-cards').innerHTML = modules.map(m => `
        <div class="module-card ${m.active ? 'active' : ''}" data-module-id="${m.id}"
             style="--module-color:${m.color}" id="module-card-${m.id}" onclick="toggleModule('${m.id}')">
            <div class="module-icon">${m.icon}</div>
            <div class="module-info"><div class="module-name">${m.name}</div><div class="module-desc">${m.description}</div></div>
            <div class="module-toggle"></div>
        </div>
    `).join('');
    updateModuleCount();
}

async function toggleModule(id) {
    try {
        const r = await fetch(API.TOGGLE + id, { method: 'POST' });
        const d = await r.json();
        if (d.success) {
            const m = modules.find(x => x.id === id);
            if (m) m.active = d.active;
            const card = document.getElementById(`module-card-${id}`);
            if (card) card.classList.toggle('active', d.active);
            showToast(`${m ? m.name : id}: ${d.active ? 'ON' : 'OFF'}`);
            updateModuleCount();
        }
    } catch (e) { showToast('Toggle failed', 'error'); }
}

async function toggleAllModules() {
    allModulesActive = !allModulesActive;
    for (const m of modules) {
        if (m.active !== allModulesActive) {
            await toggleModule(m.id);
            await new Promise(r => setTimeout(r, 80));
        }
    }
}

function updateModuleCount() {
    const active = modules.filter(m => m.active).length;
    const el = document.getElementById('module-count');
    if (el) el.textContent = `${active}/${modules.length}`;
}

// Status Polling
async function pollStatus() {
    try {
        const r = await fetch(API.STATUS);
        const d = await r.json();
        updateUI(d);
    } catch (e) { updateCameraStatus(false); }
}

function updateUI(d) {
    const fps = document.getElementById('fps-value');
    if (fps) fps.textContent = d.fps || 0;
    updateCameraStatus(d.camera_active);

    const det = d.detections || {};
    updateStat('face', det.face_detection || 0);
    updateStat('hand', det.hand_tracking || 0);
    updateStat('object', det.object_detection || 0);
    updateStat('emotion', det.emotion_detection || 0);
    updateStat('motion', det.motion_detection || 0);
    updateStat('color', det.color_analysis || 0);
    updateStat('mesh', det.face_mesh || 0);
    updateStat('pose', det.pose_estimation || 0);
    updateStat('qr', det.qr_scanner || 0);
    updateStat('artistic', det.artistic_filters || 0);
    updateStat('bg', det.background_segmentation || 0);
    updateStat('age', det.age_gender || 0);
    updateStat('speed', det.speed_tracking || 0);

    const ms = d.modules || {};
    for (const [id, active] of Object.entries(ms)) {
        const card = document.getElementById(`module-card-${id}`);
        if (card) card.classList.toggle('active', active);
        const m = modules.find(x => x.id === id);
        if (m) m.active = active;
    }
    updateModuleCount();

    if (d.recording !== isRecording) { isRecording = d.recording; updateRecordingUI(); }
    if (isRecording && d.recording_duration !== undefined) updateRecTimer(d.recording_duration);

    // Update artistic mode buttons
    if (d.artistic_mode) {
        document.querySelectorAll('.art-mode-btn').forEach(b => {
            b.classList.toggle('active', b.dataset.mode === d.artistic_mode);
        });
    }
}

function updateStat(name, count) {
    const v = document.getElementById(`stat-${name}`);
    const b = document.getElementById(`bar-${name}`);
    if (v) {
        v.textContent = count;
        if (parseInt(v.dataset.prev || '0') !== count) {
            v.style.transform = 'scale(1.15)';
            setTimeout(() => v.style.transform = 'scale(1)', 180);
            v.dataset.prev = count;
        }
    }
    if (b) b.style.width = `${Math.min(count * 10, 100)}%`;
}

function updateCameraStatus(active) {
    const p = document.getElementById('camera-status');
    if (!p) return;
    const dot = p.querySelector('.status-dot'), txt = p.querySelector('.status-text');
    if (active) { dot.className = 'status-dot active'; txt.textContent = 'Camera Active'; }
    else { dot.className = 'status-dot error'; txt.textContent = 'No Camera'; }
}

// Recording
async function toggleRecording() {
    try {
        if (!isRecording) {
            const r = await fetch(API.RECORD_START, { method: 'POST' });
            const d = await r.json();
            if (d.success) { isRecording = true; updateRecordingUI(); showToast('Recording started'); }
        } else {
            const r = await fetch(API.RECORD_STOP, { method: 'POST' });
            const d = await r.json();
            if (d.success) { isRecording = false; updateRecordingUI(); showToast('Recording saved'); }
        }
    } catch (e) { showToast('Recording failed', 'error'); }
}

function updateRecordingUI() {
    const btn = document.getElementById('record-btn');
    const txt = document.getElementById('record-btn-text');
    const badge = document.getElementById('recording-badge');
    if (isRecording) { btn.classList.add('recording'); txt.textContent = 'Stop'; badge.style.display = 'flex'; }
    else { btn.classList.remove('recording'); txt.textContent = 'Record'; badge.style.display = 'none'; }
}

function updateRecTimer(sec) {
    const t = document.getElementById('rec-timer');
    if (t) t.textContent = `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;
}

// Screenshot
async function takeScreenshot() {
    try {
        const r = await fetch(API.SCREENSHOT, { method: 'POST' });
        if (r.ok) {
            const blob = await r.blob();
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `ai_vision_${Date.now()}.jpg`;
            document.body.appendChild(a); a.click(); a.remove();
            showToast('Screenshot saved!');
        }
    } catch (e) { showToast('Screenshot failed', 'error'); }
}

// Filters
function initFilters() {
    const bs = document.getElementById('brightness-slider'), cs = document.getElementById('contrast-slider');
    const bv = document.getElementById('brightness-value'), cv = document.getElementById('contrast-value');
    if (bs) bs.addEventListener('input', function () { bv.textContent = this.value; sendFilter('brightness', +this.value); });
    if (cs) cs.addEventListener('input', function () { const v = (+this.value / 100).toFixed(1); cv.textContent = v; sendFilter('contrast', +v); });
}

async function sendFilter(name, value) {
    try { await fetch(API.FILTERS, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [name]: value }) }); } catch (e) { }
}

// Artistic Filters
function initArtistic() {
    document.querySelectorAll('.art-mode-btn').forEach(btn => {
        btn.addEventListener('click', async function () {
            const mode = this.dataset.mode;
            try {
                await fetch(API.ARTISTIC_SET, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ mode }) });
                document.querySelectorAll('.art-mode-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                showToast(`Filter: ${mode === 'off' ? 'Normal' : mode}`);
            } catch (e) { }
        });
    });
}

// Camera Switching
function initCameraSelector() {
    document.querySelectorAll('.cam-btn').forEach(btn => {
        btn.addEventListener('click', async function () {
            const idx = +this.dataset.cam;
            try {
                const r = await fetch(API.CAMERA_SWITCH, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ index: idx }) });
                const d = await r.json();
                if (d.success) {
                    document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
                    this.classList.add('active');
                    showToast(`Switched to Camera ${idx}`);
                } else showToast('Camera not available', 'error');
            } catch (e) { }
        });
    });
}

// Analytics
async function pollAnalytics() {
    try {
        const r = await fetch(API.ANALYTICS);
        const d = await r.json();
        const u = document.getElementById('analytics-uptime'), t = document.getElementById('analytics-total'), a = document.getElementById('analytics-active');
        if (u) u.textContent = d.uptime || '00:00:00';
        if (t) t.textContent = (d.total_detections || 0).toLocaleString();
        if (a) a.textContent = `${d.active_modules || 0} / ${modules.length}`;
        if (d.chart_data) { chartData = d.chart_data; drawChart(); }
    } catch (e) { }
}

// Chart
function drawChart() {
    const cnv = document.getElementById('detection-chart');
    if (!cnv || chartData.length < 2) return;
    const ctx = cnv.getContext('2d'), w = cnv.width, h = cnv.height;
    ctx.clearRect(0, 0, w, h);
    const mx = Math.max(...chartData, 1), step = w / (chartData.length - 1);
    ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1;
    for (let i = 0; i < 4; i++) { const y = (h / 4) * i; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0,255,136,.3)'); grad.addColorStop(1, 'rgba(0,255,136,0)');
    ctx.beginPath(); ctx.moveTo(0, h);
    chartData.forEach((v, i) => ctx.lineTo(i * step, h - (v / mx) * (h - 10)));
    ctx.lineTo(w, h); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath();
    chartData.forEach((v, i) => { const x = i * step, y = h - (v / mx) * (h - 10); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.strokeStyle = '#00ff88'; ctx.lineWidth = 1.5; ctx.stroke();
    if (chartData.length > 0) {
        const lx = (chartData.length - 1) * step, ly = h - (chartData[chartData.length - 1] / mx) * (h - 10);
        ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fillStyle = '#00ff88'; ctx.fill();
    }
}

// Init
document.addEventListener('DOMContentLoaded', () => {
    loadModules();
    initFilters();
    initArtistic();
    initCameraSelector();
    document.getElementById('screenshot-btn')?.addEventListener('click', takeScreenshot);
    document.getElementById('toggle-all-btn')?.addEventListener('click', toggleAllModules);
    document.getElementById('record-btn')?.addEventListener('click', toggleRecording);
    setInterval(pollStatus, 1000);
    setInterval(pollAnalytics, 2000);
    setTimeout(pollStatus, 500);
    setTimeout(pollAnalytics, 1000);
    const cnv = document.getElementById('detection-chart');
    if (cnv) { const r = cnv.getBoundingClientRect(); cnv.width = r.width; cnv.height = r.height; }
});

// Keyboard
document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key >= '1' && e.key <= '9' && !e.ctrlKey) { const i = +e.key - 1; if (modules[i]) toggleModule(modules[i].id); }
    if (e.key === '0' && !e.ctrlKey && modules[9]) toggleModule(modules[9].id);
    if (e.key === 's' && !e.ctrlKey) takeScreenshot();
    if (e.key === 'r' && !e.ctrlKey) toggleRecording();
    if (e.key === ' ') { e.preventDefault(); toggleAllModules(); }
    if (e.key === 'f' && !e.ctrlKey) fetch(API.ARTISTIC_CYCLE, { method: 'POST' });
});
