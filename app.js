/* ==========================================================================
   1. 상수 및 기본 상태
   ========================================================================== */
const SAMPLE_RATE = 44100;
const DURATION = 1.3;       // 전체 합성 길이 (초)
const NUM_SECTIONS = 44;    // 성도 관 분할 개수 (Kelly-Lochbaum 모델)
const MAX_RADIUS = 2.5;
const MIN_HEIGHT = 0.04;

const $ = (id) => document.getElementById(id);
const clamp = (val, min, max) => Math.max(min, Math.min(max, val));

// 모음 프리셋 (x: 성문 0 → 입술 1, h: 관 반지름 비율)
const VOWEL_PRESETS = {
  ah: [[0, 0.30], [0.20, 0.25], [0.40, 0.30], [0.55, 0.60], [0.75, 0.85], [1, 0.90]],
  eh: [[0, 0.35], [0.20, 0.55], [0.45, 0.45], [0.65, 0.40], [0.85, 0.55], [1, 0.65]],
  ee: [[0, 0.40], [0.25, 0.80], [0.50, 0.80], [0.65, 0.18], [0.85, 0.30], [1, 0.50]],
  oh: [[0, 0.30], [0.20, 0.35], [0.40, 0.60], [0.60, 0.55], [0.80, 0.35], [1, 0.30]],
  oo: [[0, 0.40], [0.25, 0.55], [0.50, 0.60], [0.65, 0.30], [0.85, 0.40], [0.93, 0.15], [1, 0.15]],
};

const createPointsFromPreset = (arr) =>
  arr.map(([x, h], idx) => ({
    x,
    h,
    fixed: idx === 0 || idx === arr.length - 1,
  }));

// 키프레임 및 신시사이저 파라미터 상태
let keyframes = [
  { t: 0, pts: createPointsFromPreset(VOWEL_PRESETS.ah) },
  { t: DURATION, pts: createPointsFromPreset(VOWEL_PRESETS.ee) },
];
let selectedKeyframe = keyframes[0];

const params = {
  f0: 110,    // 기본 주파수 (Hz)
  att: 0.15,  // 관 벽 손실 (Attenuation)
  A: 30,      // Attack (ms)
  D: 150,     // Decay (ms)
  S: 0.8,     // Sustain (0~1)
  R: 250,     // Release (ms)
};

let sourceMode = 'saw';
let uploadedBuffer = null;
let uploadedFileName = '';
let renderedAudio = null;
let spectrogramData = null;
let playheadTime = -1;

/* ==========================================================================
   2. 음향 물리 모델 및 DSP (Kelly-Lochbaum 도파관 + FFT)
   ========================================================================== */

/** 제어점 배열에서 특정 위치 x(0~1)의 관 반지름 비율 h를 코사인 보간으로 계산 */
function getHeightAt(pts, x) {
  if (x <= pts[0].x) return pts[0].h;

  for (let i = 1; i < pts.length; i++) {
    if (x <= pts[i].x) {
      const prev = pts[i - 1];
      const curr = pts[i];
      const u = (x - prev.x) / Math.max(curr.x - prev.x, 1e-6);
      const smooth = (1 - Math.cos(Math.PI * u)) / 2;
      return prev.h + (curr.h - prev.h) * smooth;
    }
  }
  return pts[pts.length - 1].h;
}

/** 제어점 배열을 NUM_SECTIONS 개의 이산 단면 반지름 배열로 샘플링 */
const sampleSections = (pts) =>
  Array.from({ length: NUM_SECTIONS }, (_, i) => getHeightAt(pts, (i + 0.5) / NUM_SECTIONS));

/** 시간 t에서의 단면 반지름 배열을 인접 키프레임 간 선형 보간하여 outArr에 기록 */
function interpolateSectionsAtTime(sortedKeyframes, sampledSections, t, outArr) {
  const count = sortedKeyframes.length;
  if (t <= sortedKeyframes[0].t) {
    outArr.set(sampledSections[0]);
    return;
  }
  if (t >= sortedKeyframes[count - 1].t) {
    outArr.set(sampledSections[count - 1]);
    return;
  }

  for (let i = 1; i < count; i++) {
    if (t < sortedKeyframes[i].t) {
      const prevKf = sortedKeyframes[i - 1];
      const currKf = sortedKeyframes[i];
      const u = (t - prevKf.t) / Math.max(currKf.t - prevKf.t, 1e-6);
      const secA = sampledSections[i - 1];
      const secB = sampledSections[i];

      for (let j = 0; j < NUM_SECTIONS; j++) {
        outArr[j] = secA[j] + (secB[j] - secA[j]) * u;
      }
      return;
    }
  }
}

/** 시간 t(초)에서의 ADSR 엔벨로프 게인 계산 */
function getEnvelope(t) {
  const attack = params.A / 1000;
  const decay = params.D / 1000;
  const release = params.R / 1000;
  const sustain = params.S;
  const releaseStart = Math.max(DURATION - release, 0);

  const adsGain = (time) => {
    if (time < attack) return attack ? time / attack : 1;
    if (time < attack + decay) return 1 - ((1 - sustain) * (time - attack)) / decay;
    return sustain;
  };

  if (t < releaseStart) return adsGain(t);
  return adsGain(releaseStart) * (1 - (t - releaseStart) / Math.max(release, 1e-6));
}

/** 성문 음원 신호 생성 (PolyBLEP 톱니파 또는 업로드된 오디오 파일) */
function generateSource(length) {
  const signal = new Float32Array(length);

  if (sourceMode === 'file' && uploadedBuffer) {
    let peak = 1e-9;
    const limit = Math.min(length, uploadedBuffer.length);
    for (let i = 0; i < limit; i++) {
      peak = Math.max(peak, Math.abs(uploadedBuffer[i]));
    }
    for (let i = 0; i < limit; i++) {
      signal[i] = uploadedBuffer[i] / peak;
    }
    return signal;
  }

  // Band-limited Sawtooth (PolyBLEP)
  const dt = params.f0 / SAMPLE_RATE;
  let phase = 0;

  for (let n = 0; n < length; n++) {
    phase += dt;
    if (phase >= 1) phase -= 1;

    let sample = 2 * phase - 1;
    if (phase < dt) {
      const t = phase / dt;
      sample -= t + t - t * t - 1;
    } else if (phase > 1 - dt) {
      const t = (phase - 1) / dt;
      sample -= t * t + t + t + 1;
    }
    signal[n] = sample;
  }
  return signal;
}

/** 성도 도파관 필터링 및 파형·스펙트로그램 렌더링 */
function renderAudio() {
  const totalSamples = Math.round(DURATION * SAMPLE_RATE);
  const source = generateSource(totalSamples);

  const sortedKeyframes = [...keyframes].sort((a, b) => a.t - b.t);
  const sampledSections = sortedKeyframes.map((kf) => sampleSections(kf.pts));

  const forwardWave = new Float64Array(NUM_SECTIONS);
  const backwardWave = new Float64Array(NUM_SECTIONS);
  const junctionF = new Float64Array(NUM_SECTIONS + 1);
  const junctionB = new Float64Array(NUM_SECTIONS + 1);
  const reflection = new Float64Array(NUM_SECTIONS);
  const heights = new Float64Array(NUM_SECTIONS);
  const areas = new Float64Array(NUM_SECTIONS);

  const damping = 1 - (0.0002 + params.att * 0.006);
  const output = new Float32Array(totalSamples);

  let prevLip = 0;
  let envGain = 1;

  for (let n = 0; n < totalSamples; n++) {
    // 32샘플마다 성도 단면적 및 반사 계수 갱신
    if (n % 32 === 0) {
      interpolateSectionsAtTime(sortedKeyframes, sampledSections, n / SAMPLE_RATE, heights);
      for (let i = 0; i < NUM_SECTIONS; i++) {
        areas[i] = Math.pow(MAX_RADIUS * heights[i], 2);
      }
      for (let i = 1; i < NUM_SECTIONS; i++) {
        reflection[i] = (areas[i - 1] - areas[i]) / (areas[i - 1] + areas[i]);
      }
      envGain = getEnvelope(n / SAMPLE_RATE);
    }

    // 성문 입력 및 입술 방사 반사
    forwardWave[0] = backwardWave[0] * 0.75 + source[n] * envGain;
    backwardWave[NUM_SECTIONS - 1] = forwardWave[NUM_SECTIONS - 1] * -0.85;

    // 인접 관 경계면 산란(Scattering) 계산
    for (let i = 1; i < NUM_SECTIONS; i++) {
      const r = reflection[i] * (forwardWave[i - 1] + backwardWave[i]);
      junctionF[i] = forwardWave[i - 1] - r;
      junctionB[i] = backwardWave[i] + r;
    }
    junctionF[0] = forwardWave[0];
    junctionB[NUM_SECTIONS] = backwardWave[NUM_SECTIONS - 1];

    for (let i = 0; i < NUM_SECTIONS; i++) {
      forwardWave[i] = junctionF[i] * damping;
      backwardWave[i] = junctionB[i + 1] * damping;
    }

    // 입술 끝단 고역 강조(1차 차분 방사 필터)
    const lip = forwardWave[NUM_SECTIONS - 1];
    output[n] = lip - prevLip;
    prevLip = lip;
  }

  // 출력 정규화 (-2dB 수준)
  let peak = 1e-9;
  for (let i = 0; i < totalSamples; i++) {
    peak = Math.max(peak, Math.abs(output[i]));
  }
  for (let i = 0; i < totalSamples; i++) {
    output[i] *= 0.8 / peak;
  }

  renderedAudio = output;
  computeSpectrogram();
}

/** 제자리 기수-2 고속 푸리에 변환 (In-place Radix-2 FFT) */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }

  for (let len = 2; len <= n; len <<= 1) {
    const angle = (-2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);
    const half = len >> 1;

    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < half; k++) {
        const u = i + k;
        const v = u + half;
        const tRe = re[v] * curRe - im[v] * curIm;
        const tIm = re[v] * curIm + im[v] * curRe;
        re[v] = re[u] - tRe;
        im[v] = im[u] - tIm;
        re[u] += tRe;
        im[u] += tIm;

        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/** STFT 기반 스펙트로그램(0~8 kHz) 데이터 계산 */
function computeSpectrogram() {
  const winSize = 1024;
  const hopSize = 256;
  const maxBin = Math.floor((8000 / SAMPLE_RATE) * winSize);
  const numFrames = Math.floor((renderedAudio.length - winSize) / hopSize) + 1;

  const re = new Float64Array(winSize);
  const im = new Float64Array(winSize);
  const frames = [];
  let maxMag = 1e-12;

  for (let f = 0; f < numFrames; f++) {
    for (let i = 0; i < winSize; i++) {
      const hann = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / winSize);
      re[i] = renderedAudio[f * hopSize + i] * hann;
      im[i] = 0;
    }
    fft(re, im);

    const bins = new Float32Array(maxBin);
    for (let b = 0; b < maxBin; b++) {
      bins[b] = Math.hypot(re[b], im[b]);
      if (bins[b] > maxMag) maxMag = bins[b];
    }
    frames.push(bins);
  }

  spectrogramData = { numFrames, maxBin, frames, maxMag };
}

/* ==========================================================================
   3. 캔버스 렌더링 (성도 편집기, 타임라인, 엔벨로프, 파형/스펙트로그램)
   ========================================================================== */

/** 고해상도(HiDPI) 디스플레이에 맞춰 캔버스 크기와 스케일 조정 */
function setupCanvas(canvas) {
  const dpr = devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const targetW = Math.round(width * dpr);
  const targetH = Math.round(height * dpr);

  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, width, height];
}

const getCssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

let editorLayout = { padX: 40, midY: 0, halfHeight: 0, width: 0 };

/** 성도 단면 편집기 그리기 */
function drawEditor() {
  const [ctx, width, height] = setupCanvas($('ed'));
  const padX = 40;
  const midY = height / 2;
  const halfHeight = height * 0.42;
  const toCanvasX = (x) => padX + x * (width - 2 * padX);
  const pts = keyframes.includes(selectedKeyframe) ? selectedKeyframe.pts : keyframes[0].pts;

  editorLayout = { padX, midY, halfHeight, width };
  ctx.clearRect(0, 0, width, height);

  const tubeColor = getCssVar('--tube');
  const pointColor = getCssVar('--pt');
  const mutedColor = getCssVar('--mut');

  // 관 내부 채우기
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = i / 200;
    const y = midY - getHeightAt(pts, x) * halfHeight;
    i === 0 ? ctx.moveTo(toCanvasX(x), y) : ctx.lineTo(toCanvasX(x), y);
  }
  for (let i = 200; i >= 0; i--) {
    const x = i / 200;
    const y = midY + getHeightAt(pts, x) * halfHeight;
    ctx.lineTo(toCanvasX(x), y);
  }
  ctx.closePath();
  ctx.globalAlpha = 0.16;
  ctx.fillStyle = tubeColor;
  ctx.fill();
  ctx.globalAlpha = 1;

  // 중심선 (점선)
  ctx.strokeStyle = mutedColor;
  ctx.setLineDash([4, 5]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, midY);
  ctx.lineTo(width - padX, midY);
  ctx.stroke();
  ctx.setLineDash([]);

  // 위·아래 경계 곡선
  const traceWall = (sign) => {
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = i / 200;
      const y = midY + sign * getHeightAt(pts, x) * halfHeight;
      i === 0 ? ctx.moveTo(toCanvasX(x), y) : ctx.lineTo(toCanvasX(x), y);
    }
  };
  ctx.strokeStyle = tubeColor;
  ctx.lineWidth = 2.5;
  traceWall(-1);
  ctx.stroke();
  traceWall(1);
  ctx.stroke();

  // 제어점 핸들
  for (const pt of pts) {
    for (const sign of [-1, 1]) {
      ctx.beginPath();
      ctx.arc(toCanvasX(pt.x), midY + sign * pt.h * halfHeight, 6, 0, Math.PI * 2);
      ctx.fillStyle = pointColor;
      ctx.fill();
      ctx.strokeStyle = getCssVar('--card');
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  // 라벨 표시
  ctx.fillStyle = mutedColor;
  ctx.font = '12px "Noto Sans KR", sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText('성문', 6, midY + 4);
  ctx.textAlign = 'right';
  ctx.fillText('입술', width - 6, midY + 4);
  ctx.textAlign = 'center';
  ctx.fillText(`편집 중: ${selectedKeyframe.t.toFixed(2)} s 키프레임`, width / 2, 14);
}

/** 키프레임 타임라인 그리기 */
function drawTimeline() {
  const [ctx, width, height] = setupCanvas($('tl'));
  const padX = 22;
  const timeToX = (t) => padX + (t / DURATION) * (width - 2 * padX);

  ctx.clearRect(0, 0, width, height);

  const mutedColor = getCssVar('--mut');
  const lineColor = getCssVar('--ln');
  const pointColor = getCssVar('--pt');
  const fgColor = getCssVar('--fg');

  // 시간 눈금선
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 1;
  ctx.fillStyle = mutedColor;
  ctx.font = '11px "Noto Sans KR", sans-serif';
  ctx.textAlign = 'center';

  for (let i = 0; i <= 13; i++) {
    const x = timeToX(i / 10);
    ctx.beginPath();
    ctx.moveTo(x, height - 18);
    ctx.lineTo(x, height - (i % 5 ? 13 : 8));
    ctx.stroke();
    if (i % 5 === 0) {
      ctx.fillText(`${(i / 10).toFixed(1)}s`, x, height);
    }
  }

  ctx.beginPath();
  ctx.moveTo(padX, height - 18);
  ctx.lineTo(width - padX, height - 18);
  ctx.stroke();

  // 키프레임 마커
  for (const kf of keyframes) {
    const x = timeToX(kf.t);
    const isSelected = kf === selectedKeyframe;

    ctx.strokeStyle = isSelected ? pointColor : fgColor;
    ctx.lineWidth = isSelected ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 8);
    ctx.lineTo(x, height - 18);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, 2);
    ctx.lineTo(x + 7, 10);
    ctx.lineTo(x, 18);
    ctx.lineTo(x - 7, 10);
    ctx.closePath();
    ctx.fillStyle = isSelected ? pointColor : fgColor;
    ctx.fill();
  }

  // 재생 헤드 표시
  if (playheadTime >= 0) {
    const px = timeToX(playheadTime);
    ctx.strokeStyle = pointColor;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height - 18);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  $('kfinfo').textContent = `키프레임 ${keyframes.length}개 · 빈 곳 클릭: 추가, 마커 드래그: 시간 이동`;
}

/** ADSR 엔벨로프 그래프 그리기 */
function drawEnvelope() {
  const [ctx, width, height] = setupCanvas($('env'));
  ctx.clearRect(0, 0, width, height);

  ctx.strokeStyle = getCssVar('--tube');
  ctx.lineWidth = 2;
  ctx.beginPath();

  for (let i = 0; i <= 300; i++) {
    const t = (DURATION * i) / 300;
    const x = 4 + (t / DURATION) * (width - 8);
    const y = height - 6 - getEnvelope(t) * (height - 14);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = getCssVar('--mut');
  ctx.font = '11px "Noto Sans KR", sans-serif';
  ctx.fillText('엔벨로프 (1.3초 전체, 입력 음원에 적용)', 6, 11);
}

/** 파형 및 스펙트로그램 그리기 */
function drawOutput() {
  // 1. 파형
  const [wfCtx, wfW, wfH] = setupCanvas($('wf'));
  wfCtx.clearRect(0, 0, wfW, wfH);
  wfCtx.strokeStyle = getCssVar('--ln');
  wfCtx.beginPath();
  wfCtx.moveTo(0, wfH / 2);
  wfCtx.lineTo(wfW, wfH / 2);
  wfCtx.stroke();

  if (renderedAudio) {
    wfCtx.fillStyle = getCssVar('--tube');
    const samplesPerPx = renderedAudio.length / wfW;

    for (let x = 0; x < wfW; x++) {
      let minVal = 1;
      let maxVal = -1;
      const start = Math.floor(x * samplesPerPx);
      const end = Math.floor((x + 1) * samplesPerPx);

      for (let i = start; i < end; i++) {
        const val = renderedAudio[i];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }

      const barTop = wfH / 2 - (maxVal * wfH) / 2;
      const barHeight = Math.max(1, ((maxVal - minVal) * wfH) / 2);
      wfCtx.fillRect(x, barTop, 1, barHeight);
    }
  }

  // 2. 스펙트로그램
  const [spCtx, spW, spH] = setupCanvas($('sp'));
  spCtx.clearRect(0, 0, spW, spH);
  if (!spectrogramData) return;

  const { numFrames, maxBin, frames, maxMag } = spectrogramData;
  const offscreen = document.createElement('canvas');
  offscreen.width = numFrames;
  offscreen.height = maxBin;

  const offCtx = offscreen.getContext('2d');
  const imgData = offCtx.createImageData(numFrames, maxBin);

  for (let f = 0; f < numFrames; f++) {
    for (let b = 0; b < maxBin; b++) {
      const dbNorm = clamp((20 * Math.log10(frames[f][b] / maxMag + 1e-9) + 70) / 70, 0, 1);
      const idx = ((maxBin - 1 - b) * numFrames + f) * 4;

      imgData.data[idx]     = 255 * clamp((dbNorm - 0.30) * 2.2, 0, 1);
      imgData.data[idx + 1] = 255 * clamp((dbNorm - 0.62) * 2.6, 0, 1);
      imgData.data[idx + 2] = 230 * clamp(dbNorm * 2.2, 0, 1) * (1 - 0.75 * clamp((dbNorm - 0.55) * 2.2, 0, 1));
      imgData.data[idx + 3] = 255;
    }
  }

  offCtx.putImageData(imgData, 0, 0);
  spCtx.imageSmoothingEnabled = true;
  spCtx.drawImage(offscreen, 0, 0, spW, spH);

  // 주파수 눈금 라벨
  spCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
  spCtx.font = '11px "Noto Sans KR", sans-serif';
  [0, 2, 4, 6, 8].forEach((khz) => {
    const y = khz === 0 ? spH - 3 : spH - (khz / 8) * spH + 11;
    spCtx.fillText(`${khz}k`, 3, y);
  });
}

const drawAll = () => {
  drawEditor();
  drawTimeline();
  drawEnvelope();
  drawOutput();
};

let renderTimer;
const scheduleRender = () => {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => {
    renderAudio();
    drawOutput();
  }, 90);
};

const refreshEditorAndRender = () => {
  drawEditor();
  drawTimeline();
  scheduleRender();
};

/* ==========================================================================
   4. 사용자 인터랙션 (단면 편집기, 타임라인, 슬라이더)
   ========================================================================== */

// --- 단면 편집기 포인터 이벤트 ---
let draggingPoint = null;
const editorCanvas = $('ed');

editorCanvas.addEventListener('pointerdown', (e) => {
  const rect = editorCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const { padX, midY, halfHeight, width } = editorLayout;
  const pts = selectedKeyframe.pts;
  const toCanvasX = (x) => padX + x * (width - 2 * padX);

  let hit = pts.find(
    (pt) => Math.hypot(px - toCanvasX(pt.x), Math.abs(py - midY) - pt.h * halfHeight) < 14
  );

  // 기존 점이 아니라 곡선 근처를 클릭했다면 새 제어점 추가
  if (!hit) {
    const normX = (px - padX) / (width - 2 * padX);
    const distFromWall = Math.abs(Math.abs(py - midY) - getHeightAt(pts, normX) * halfHeight);
    if (normX > 0.01 && normX < 0.99 && distFromWall < 12) {
      hit = { x: normX, h: getHeightAt(pts, normX), fixed: false };
      pts.push(hit);
      pts.sort((a, b) => a.x - b.x);
    }
  }

  if (hit) {
    draggingPoint = hit;
    editorCanvas.setPointerCapture(e.pointerId);
    refreshEditorAndRender();
  }
});

editorCanvas.addEventListener('pointermove', (e) => {
  if (!draggingPoint) return;
  const rect = editorCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const { padX, midY, halfHeight, width } = editorLayout;

  draggingPoint.h = clamp(Math.abs(py - midY) / halfHeight, MIN_HEIGHT, 1);
  if (!draggingPoint.fixed) {
    draggingPoint.x = clamp((px - padX) / (width - 2 * padX), 0.02, 0.98);
  }

  selectedKeyframe.pts.sort((a, b) => a.x - b.x);
  refreshEditorAndRender();
});

editorCanvas.addEventListener('pointerup', () => {
  draggingPoint = null;
});

editorCanvas.addEventListener('dblclick', (e) => {
  const rect = editorCanvas.getBoundingClientRect();
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  const { padX, midY, halfHeight, width } = editorLayout;
  const pts = selectedKeyframe.pts;

  const target = pts.find(
    (pt) =>
      !pt.fixed &&
      Math.hypot(px - (padX + pt.x * (width - 2 * padX)), Math.abs(py - midY) - pt.h * halfHeight) < 14
  );

  if (target) {
    pts.splice(pts.indexOf(target), 1);
    refreshEditorAndRender();
  }
});

// --- 타임라인 포인터 이벤트 ---
const timelineCanvas = $('tl');
let draggingKeyframe = null;

const pointerToTime = (e) => {
  const rect = timelineCanvas.getBoundingClientRect();
  return clamp((e.clientX - rect.left - 22) / (rect.width - 44), 0, 1) * DURATION;
};

timelineCanvas.addEventListener('pointerdown', (e) => {
  const rect = timelineCanvas.getBoundingClientRect();
  const width = rect.width;
  const t = pointerToTime(e);
  const px = e.clientX - rect.left;

  const hit = keyframes.find((kf) => Math.abs(22 + (kf.t / DURATION) * (width - 44) - px) < 10);

  if (hit) {
    selectedKeyframe = hit;
    draggingKeyframe = hit;
  } else {
    // 빈 공간 클릭 시 현재 보간된 모양으로 새 키프레임 생성
    const sorted = [...keyframes].sort((a, b) => a.t - b.t);
    const sampled = sorted.map((kf) => sampleSections(kf.pts));
    const interp = new Float64Array(NUM_SECTIONS);
    interpolateSectionsAtTime(sorted, sampled, t, interp);

    const pts = [];
    for (let j = 0; j <= 8; j++) {
      const idx = Math.min(NUM_SECTIONS - 1, Math.floor((j / 8) * NUM_SECTIONS));
      pts.push({ x: j / 8, h: interp[idx], fixed: j === 0 || j === 8 });
    }

    const newKf = { t, pts };
    keyframes.push(newKf);
    selectedKeyframe = newKf;
    draggingKeyframe = newKf;
  }

  timelineCanvas.setPointerCapture(e.pointerId);
  refreshEditorAndRender();
});

timelineCanvas.addEventListener('pointermove', (e) => {
  if (!draggingKeyframe) return;
  draggingKeyframe.t = pointerToTime(e);
  refreshEditorAndRender();
});

timelineCanvas.addEventListener('pointerup', () => {
  draggingKeyframe = null;
  keyframes.sort((a, b) => a.t - b.t);
});

$('del').onclick = () => {
  if (keyframes.length <= 2) return;
  keyframes.splice(keyframes.indexOf(selectedKeyframe), 1);
  selectedKeyframe = keyframes[0];
  refreshEditorAndRender();
};

document.querySelectorAll('[data-p]').forEach((btn) => {
  btn.onclick = () => {
    selectedKeyframe.pts = createPointsFromPreset(VOWEL_PRESETS[btn.dataset.p]);
    refreshEditorAndRender();
  };
});

// --- 파라미터 슬라이더 UI 동적 생성 ---
const sliderRefs = {};
const SLIDER_CONFIGS = [
  ['f0', '기본 주파수 f0', 'Hz', 60, 300, 1],
  ['att', 'Attenuation (관 벽 손실)', '', 0, 1, 0.01],
  ['A', 'Attack', 'ms', 0, 600, 1],
  ['D', 'Decay', 'ms', 0, 800, 1],
  ['S', 'Sustain', '', 0, 1, 0.01],
  ['R', 'Release', 'ms', 0, 800, 1],
];

for (const [key, label, unit, min, max, step] of SLIDER_CONFIGS) {
  const wrapper = document.createElement('div');
  wrapper.className = 'sl';
  wrapper.innerHTML = `
    <span>${label}</span>
    <input type="range" min="${min}" max="${max}" step="${step}" value="${params[key]}" aria-label="${label}">
    <b></b>
  `;

  const input = wrapper.querySelector('input');
  const readout = wrapper.querySelector('b');

  const updateReadout = () => {
    const formatted = step < 1 ? Number(input.value).toFixed(2) : input.value;
    readout.textContent = unit ? `${formatted} ${unit}` : formatted;
  };

  input.oninput = () => {
    params[key] = Number(input.value);
    updateReadout();
    drawEnvelope();
    scheduleRender();
  };

  updateReadout();
  sliderRefs[key] = { input, updateReadout };
  $('sl').appendChild(wrapper);
}

document.querySelectorAll('[name=m]').forEach((radio) => {
  radio.onchange = () => {
    sourceMode = radio.value;
    if (sourceMode === 'file' && !uploadedBuffer) {
      $('st').textContent = '오디오 파일을 선택하세요. 파일이 없으면 무음입니다.';
    } else {
      $('st').textContent = '';
    }
    scheduleRender();
  };
});

/* ==========================================================================
   5. Web Audio 재생 및 파일 입력
   ========================================================================== */
let audioCtx;
let activeSourceNode;
let playStartTime = 0;

const getAudioContext = () =>
  audioCtx || (audioCtx = new (window.AudioContext || window.webkitAudioContext)());

$('file').onchange = async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  try {
    const decoded = await getAudioContext().decodeAudioData(await file.arrayBuffer());
    const offline = new OfflineAudioContext(1, Math.round(DURATION * SAMPLE_RATE), SAMPLE_RATE);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();

    const rendered = await offline.startRendering();
    uploadedBuffer = rendered.getChannelData(0);
    uploadedFileName = file.name;
    sourceMode = 'file';

    document.querySelector('[name=m][value=file]').checked = true;
    $('st').textContent = `${file.name} (앞 1.3초 사용)`;
    scheduleRender();
  } catch {
    $('st').textContent = '파일을 읽지 못했습니다. 다른 오디오 파일을 선택하세요.';
  }
};

function updatePlayhead() {
  if (!activeSourceNode) return;
  playheadTime = getAudioContext().currentTime - playStartTime;
  const playheadEl = $('ph');

  if (playheadTime > DURATION) {
    playheadTime = -1;
    playheadEl.style.display = 'none';
    drawTimeline();
    return;
  }

  const spCanvas = $('sp');
  const leftPos = spCanvas.offsetLeft + (playheadTime / DURATION) * spCanvas.clientWidth;
  playheadEl.style.display = 'block';
  playheadEl.style.left = `${leftPos}px`;

  drawTimeline();
  requestAnimationFrame(updatePlayhead);
}

$('play').onclick = () => {
  if (!renderedAudio) renderAudio();
  const ctx = getAudioContext();
  ctx.resume();

  if (activeSourceNode) {
    try {
      activeSourceNode.stop();
    } catch {
      // 이미 정지된 노드 무시
    }
  }

  const buffer = ctx.createBuffer(1, renderedAudio.length, SAMPLE_RATE);
  buffer.copyToChannel(renderedAudio, 0);

  activeSourceNode = ctx.createBufferSource();
  activeSourceNode.buffer = buffer;
  activeSourceNode.connect(ctx.destination);

  playStartTime = ctx.currentTime;
  activeSourceNode.start();
  requestAnimationFrame(updatePlayhead);
};

/* ==========================================================================
   6. 오디오 내보내기 (WAV / MP3) 및 프로젝트 저장·불러오기
   ========================================================================== */
let toastTimer;

const showToast = (message) => {
  const toastEl = $('xs');
  toastEl.textContent = message;
  toastEl.style.display = 'block';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.style.display = 'none';
  }, 5000);
};

function downloadFile(filename, data, successMessage) {
  const blob = data instanceof Blob ? data : new Blob([data], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 3000);
  showToast(successMessage);
}

function floatToPcm16(floatArr) {
  const pcm = new Int16Array(floatArr.length);
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = Math.round(clamp(floatArr[i], -1, 1) * 32767);
  }
  return pcm;
}

function encodeWav(floatArr) {
  const pcm = floatToPcm16(floatArr);
  const view = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const writeStr = (offset, str) =>
    [...str].forEach((ch, idx) => view.setUint8(offset + idx, ch.charCodeAt(0)));

  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, pcm.length * 2, true);

  for (let i = 0; i < pcm.length; i++) {
    view.setInt16(44 + i * 2, pcm[i], true);
  }
  return new Uint8Array(view.buffer);
}

function encodeMp3(floatArr) {
  const encoder = new lamejs.Mp3Encoder(1, SAMPLE_RATE, 128);
  const pcm = floatToPcm16(floatArr);
  const chunks = [];

  for (let i = 0; i < pcm.length; i += 1152) {
    const encoded = encoder.encodeBuffer(pcm.subarray(i, i + 1152));
    if (encoded.length) chunks.push(new Uint8Array(encoded));
  }
  const flushed = encoder.flush();
  if (flushed.length) chunks.push(new Uint8Array(flushed));

  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result;
}

$('xwav').onclick = () => {
  renderAudio();
  downloadFile(
    'vowel.wav',
    new Blob([encodeWav(renderedAudio)], { type: 'audio/wav' }),
    'vowel.wav를 저장했습니다.'
  );
};

$('xmp3').onclick = () => {
  if (!window.lamejs) {
    showToast('MP3 인코더를 불러오지 못했습니다. 인터넷 연결을 확인하거나 WAV로 내보내 주세요.');
    return;
  }
  renderAudio();
  downloadFile(
    'vowel.mp3',
    new Blob([encodeMp3(renderedAudio)], { type: 'audio/mpeg' }),
    'vowel.mp3를 저장했습니다.'
  );
};

const uint8ToBase64 = (bytes) => {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + 8192));
  }
  return btoa(binary);
};

$('psave').onclick = () => {
  const encodedFile = uploadedBuffer
    ? uint8ToBase64(new Uint8Array(floatToPcm16(uploadedBuffer).buffer))
    : null;

  const projectData = {
    app: 'vowel-tract',
    v: 1,
    mode: sourceMode,
    V: params,
    fileName: uploadedFileName,
    file: encodedFile,
    kfs: keyframes.map((kf) => ({
      t: kf.t,
      pts: kf.pts.map((pt) => [pt.x, pt.h]),
    })),
  };

  downloadFile(
    'vowel-project.json',
    JSON.stringify(projectData, null, 2),
    'vowel-project.json을 저장했습니다.'
  );
};

$('pload').onclick = () => $('pfile').click();

$('pfile').onchange = async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file) return;

  try {
    const data = JSON.parse(await file.text());
    if (data.app !== 'vowel-tract' || !Array.isArray(data.kfs) || data.kfs.length < 2) {
      throw new Error('Invalid project format');
    }

    const loadedKeyframes = data.kfs
      .map((kf) => {
        if (!Array.isArray(kf.pts) || kf.pts.length < 2) {
          throw new Error('Invalid keyframe points');
        }
        const pts = kf.pts
          .map(([x, h]) => ({
            x: clamp(Number(x), 0, 1),
            h: clamp(Number(h), MIN_HEIGHT, 1),
            fixed: false,
          }))
          .sort((a, b) => a.x - b.x);

        pts[0].x = 0;
        pts[0].fixed = true;
        pts[pts.length - 1].x = 1;
        pts[pts.length - 1].fixed = true;

        return { t: clamp(Number(kf.t), 0, DURATION), pts };
      })
      .sort((a, b) => a.t - b.t);

    const hasNaN = loadedKeyframes.some(
      (kf) => Number.isNaN(kf.t) || kf.pts.some((pt) => Number.isNaN(pt.x) || Number.isNaN(pt.h))
    );
    if (hasNaN) throw new Error('Corrupted numeric values');

    keyframes = loadedKeyframes;
    selectedKeyframe = keyframes[0];

    for (const key in params) {
      if (data.V && Number.isFinite(data.V[key])) {
        const ref = sliderRefs[key];
        params[key] = clamp(Number(data.V[key]), Number(ref.input.min), Number(ref.input.max));
        ref.input.value = params[key];
        ref.updateReadout();
      }
    }

    uploadedFileName = data.fileName || '';
    uploadedBuffer = null;

    if (data.file) {
      const binary = atob(data.file);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
      }
      const int16 = new Int16Array(bytes.buffer, 0, bytes.length >> 1);
      uploadedBuffer = Float32Array.from(int16, (v) => v / 32767);
    }

    sourceMode = data.mode === 'file' && uploadedBuffer ? 'file' : 'saw';
    document.querySelector(`[name=m][value=${sourceMode}]`).checked = true;
    $('st').textContent = uploadedBuffer ? `${uploadedFileName} (프로젝트에 포함된 입력)` : '';

    renderAudio();
    drawAll();
    showToast('프로젝트를 불러왔습니다.');
  } catch {
    showToast('프로젝트 파일을 읽지 못했습니다. 이 앱에서 저장한 .json 파일인지 확인하세요.');
  }
};

/* ==========================================================================
   7. 초기화
   ========================================================================== */
window.addEventListener('resize', drawAll);
renderAudio();
drawAll();