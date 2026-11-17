/* Unit 5 — Neural Networks: T13 optimizers (+Dropout/BatchNorm), T14 back-propagation, T15 CNN face recognition, B3 perceptron */
(function () {
  'use strict';
  const { RNG, Net, Optimizer, softmaxCE } = ML;
  const OPTS = [['sgd', 'SGD', '#9b9384'], ['momentum', 'Momentum', '#d7b56d'], ['nesterov', 'Nesterov', '#b4c46a'], ['rmsprop', 'RMSProp', '#86b894'], ['adagrad', 'Adagrad', '#c97b9e'], ['adam', 'Adam', '#7aa2c8']];

  /* shared 2-D datasets in [-1.2, 1.2]^2 */
  function make2D(kind, n, noise, rng) {
    const X = [], y = []; const K = kind === 'spiral' ? 3 : 2;
    for (let i = 0; i < n; i++) {
      const c = i % K; let x, yy;
      if (kind === 'spiral') { const t = rng.rand() * 0.95 + 0.05, a = t * 3.2 + c * 2 * Math.PI / 3; x = Math.cos(a) * t * 1.05 + rng.normal() * noise * 0.15; yy = Math.sin(a) * t * 1.05 + rng.normal() * noise * 0.15; }
      else if (kind === 'moons') { const t = rng.rand() * Math.PI; if (c === 0) { x = Math.cos(t) - 0.5; yy = Math.sin(t) - 0.25; } else { x = 0.5 - Math.cos(t); yy = 0.25 - Math.sin(t); } x += rng.normal() * noise * 0.25; yy += rng.normal() * noise * 0.25; x *= 0.85; yy *= 0.85; }
      else if (kind === 'circles') { const t = rng.rand() * 6.283, r = (c === 0 ? 0.45 : 0.95) + rng.normal() * noise * 0.12; x = Math.cos(t) * r; yy = Math.sin(t) * r; }
      else { x = rng.uniform(-1, 1); yy = rng.uniform(-1, 1); const cc = x * yy > 0 ? 1 : 0; if (rng.rand() < noise * 0.3) { X.push(Float64Array.from([x, yy])); y.push(1 - cc); continue; } X.push(Float64Array.from([x, yy])); y.push(cc); continue; }
      X.push(Float64Array.from([x, yy])); y.push(c);
    }
    return { X, y, K };
  }
  const flat = X => { const o = new Float64Array(X.length * 2); X.forEach((x, i) => { o[i * 2] = x[0]; o[i * 2 + 1] = x[1]; }); return o; };
  function drawBoundary(g, net, K, W, H, data, res = 28, alpha = 0.5, pts = true, dom = 1.25) {
    const gx = new Float64Array(res * res * 2); for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) { gx[(j * res + i) * 2] = -dom + 2 * dom * (i + 0.5) / res; gx[(j * res + i) * 2 + 1] = dom - 2 * dom * (j + 0.5) / res; }
    const out = net.forward(gx, res * res, false), cw = W / res, ch = H / res;
    for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) { const o = (j * res + i); const p = ML.softmax(Array.from(out.subarray(o * K, o * K + K))); let b = 0; for (let k = 1; k < K; k++) if (p[k] > p[b]) b = k; g.fillStyle = Colors.rgba(Colors.classColor(b), 0.15 + alpha * (p[b] - 1 / K) / (1 - 1 / K)); g.fillRect(i * cw, j * ch, cw + 0.6, ch + 0.6); }
    if (pts && data) data.X.forEach((x, i) => { if (i % (res > 30 ? 1 : 2)) return; g.beginPath(); g.arc((x[0] + dom) / (2 * dom) * W, (dom - x[1]) / (2 * dom) * H, res > 30 ? 3.2 : 2, 0, 7); g.fillStyle = Colors.classColor(data.y[i]); g.fill(); g.lineWidth = 0.8; g.strokeStyle = '#100e0c'; g.stroke(); });
  }

  /* ============================================================ T13 ===== */
  const LAND = {
    rosen: { name: 'Rosenbrock banana valley', dom: [-2, 2, -1, 3], f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2, g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)], mins: [[1, 1]], fmin: 0, start: [-1.5, 2.2], lr: 0.002, hint: 'a narrow curved valley: gradient points across it, not along it' },
    himmel: { name: 'Himmelblau (4 minima)', dom: [-5, 5, -5, 5], f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2, g: (x, y) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)], mins: [[3, 2], [-2.805, 3.131], [-3.779, -3.283], [3.584, -1.848]], fmin: 0, start: [-0.3, 0.4], lr: 0.01, hint: 'four equally good minima — which one each optimizer picks depends on its dynamics' },
    ravine: { name: 'Ill-conditioned ravine', dom: [-4, 4, -2, 2], f: (x, y) => 0.05 * x * x + 2.5 * y * y, g: (x, y) => [0.1 * x, 5 * y], mins: [[0, 0]], fmin: 0, start: [-3.6, 1.6], lr: 0.1, hint: '50:1 curvature ratio — plain SGD zig-zags' },
    beale: { name: 'Beale', dom: [-4.5, 4.5, -4.5, 4.5], f: (x, y) => (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y ** 3) ** 2, g: (x, y) => [2 * (1.5 - x + x * y) * (y - 1) + 2 * (2.25 - x + x * y * y) * (y * y - 1) + 2 * (2.625 - x + x * y ** 3) * (y ** 3 - 1), 2 * (1.5 - x + x * y) * x + 2 * (2.25 - x + x * y * y) * 2 * x * y + 2 * (2.625 - x + x * y ** 3) * 3 * x * y * y], mins: [[3, 0.5]], fmin: 0, start: [-3, -3], lr: 0.01, hint: 'flat plateaus and steep walls: huge gradients far away' },
    saddle: { name: 'Saddle then valleys', dom: [-3, 3, -3, 3], f: (x, y) => x * x - y * y + 0.25 * y ** 4, g: (x, y) => [2 * x, -2 * y + y ** 3], mins: [[0, 1.414], [0, -1.414]], fmin: -1, start: [2.2, 0.03], lr: 0.05, hint: 'starts right next to a saddle point where the y-gradient is almost zero' },
  };
  const DSN = { spiral: 'Spiral (3 arms)', moons: 'Two moons', circles: 'Circles' };
  Lab.register({
    id: 't13', unit: 5, num: 'T13', title: 'Optimizer arena (+ Dropout & BatchNorm)',
    tag: 'Race SGD, Momentum, Nesterov, RMSProp, Adagrad and Adam down real loss landscapes, then train identical MLPs with Dropout / BatchNorm and compare decision boundaries.',
    star: 'Click anywhere to set the start point · six optimizers in lock-step · live decision boundaries per optimizer',
    idea: `<p>Every optimizer in the syllabus follows the same loop — compute a gradient, move — but they differ in <i>how they use gradient history</i>. <b>Momentum</b> accumulates a velocity, <b>RMSProp</b> and <b>Adagrad</b> scale each coordinate by the history of squared gradients, <b>Adam</b> does both with bias correction. The notebook compares four of them on a digits MLP via a loss curve; here you can <b>see the trajectories</b>.</p>
    <p><b>Tab 1</b> races the optimizers on classic 2-D loss surfaces (click to set the start, change the learning rate and watch who diverges, who zig-zags, who gets stuck on the saddle). <b>Tab 2</b> trains the <i>same</i> MLP (same init, same mini-batches) with each optimizer, with switches for <b>Dropout</b> and <b>Batch Normalisation</b>, and shows each model’s decision boundary.</p>`,
    tryIt: ['Landscape race → <b>Ill-conditioned ravine</b> (default): Momentum and Nesterov overshoot and swing back and forth across the narrow valley before settling; plain SGD glides but slowly; Adagrad’s steps shrink as its accumulator grows; RMSProp keeps jittering at a fixed step size.',
      '<b>Rosenbrock</b>: at the default step size Nesterov <b>diverges</b> (the card turns “diverged”) while SGD and Momentum crawl along the curved valley; the adaptive methods barely move at this small η — raise η and watch the order change.',
      '<b>Himmelblau</b>: SGD and Nesterov reach a minimum within ~50 steps; Adam, RMSProp and Adagrad need several hundred at this η. Click elsewhere to restart from a different point and see which of the four minima each one picks.',
      'MLP arena: train on <b>Spiral</b>. Then toggle <b>BatchNorm</b> — early progress usually speeds up. Raise <b>Dropout</b> to 0.5: training loss rises but test accuracy can improve (less overfitting).',
      'Compare the boundaries: a good optimizer finds smooth spirals; a poor one is still a blob after the same number of steps.'],
    hood: `<pre>SGD       : θ ← θ − η g
Momentum  : v ← μ v + g ;           θ ← θ − η v
Nesterov  : v ← μ v + g ;           θ ← θ − η (g + μ v)
Adagrad   : G ← G + g² ;            θ ← θ − η g / (√G + ε)
RMSProp   : s ← 0.9 s + 0.1 g² ;    θ ← θ − η g / (√s + ε)
Adam      : m ← β₁m + (1−β₁)g ; v ← β₂v + (1−β₂)g² ;  θ ← θ − η m̂ / (√v̂ + ε)    (m̂, v̂ bias-corrected)

BatchNorm (train): x̂ = (x − μ_B)/√(σ²_B + ε) ; y = γ x̂ + β   — running statistics are used at test time
Dropout  (train): h ← h ⊙ mask / (1−p)                          — identity at test time</pre>
    <p>All of this is the same <code>ML.Optimizer</code> / <code>ML.Net</code> code used by the DQN in T2 and gradient-checked against finite differences.</p>`,
    quiz: [
      { q: 'Which optimizer keeps a per-parameter running average of squared gradients AND a momentum term?', o: ['SGD', 'Adagrad', 'RMSProp', 'Adam'], a: 3, why: 'Adam combines momentum (first moment) with RMSProp-style scaling (second moment).' },
      { q: 'Adagrad’s effective learning rate over a long run…', o: ['grows', 'shrinks because the squared-gradient sum only grows', 'stays constant', 'oscillates'], a: 1, why: 'G accumulates g² forever, so η/√G keeps decreasing — good for sparse features, but it can stall.' },
      { q: 'At test time Dropout and BatchNorm…', o: ['behave exactly as in training', 'use no random masks / use running statistics instead of batch statistics', 'are removed from the model entirely', 'double the learning rate'], a: 1, why: 'Dropout is disabled (inverted-dropout scaling was done in training) and BN uses the running mean/variance.' }],
    mount(root, ctx) {
      const tabs = Ui.seg([{ v: 'land', label: '1 · Landscape race' }, { v: 'mlp', label: '2 · MLP arena (Dropout / BatchNorm)' }], 'land', v => { landBox.style.display = v === 'land' ? '' : 'none'; mlpBox.style.display = v === 'mlp' ? '' : 'none'; });
      const landBox = h('div'), mlpBox = h('div', { style: 'display:none' }); root.appendChild(h('div', { style: 'margin-bottom:14px' }, tabs)); root.appendChild(landBox); root.appendChild(mlpBox);

      /* ---------- landscape ---------- */
      const L = { key: 'ravine', speed: 3, lr: LAND.ravine.lr, mom: 0.9, run: false, show: new Set(OPTS.map(o => o[0])) }; let runners = [], hist = {}, start = LAND.ravine.start.slice(), bg = null, steps = 0;
      const cv = Ui.canvas(640, 520), cvC = Ui.canvas(560, 230), chart = new LineChart(cvC, { xlabel: 'step', title: 'log₁₀ ( f(θ) − f_min + 1e-6 )' });
      const land = () => LAND[L.key], toPx = (x, y) => { const d = land().dom; return [(x - d[0]) / (d[1] - d[0]) * cv.w, (d[3] - y) / (d[3] - d[2]) * cv.h]; }, toXY = (px, py) => { const d = land().dom; return [d[0] + px / cv.w * (d[1] - d[0]), d[3] - py / cv.h * (d[3] - d[2])]; };
      function paintBg() { const d = land().dom, off = document.createElement('canvas'), N = 130; off.width = off.height = N; const og = off.getContext('2d'), im = og.createImageData(N, N); let lo = Infinity, hi = -Infinity; const vals = new Float64Array(N * N); for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { const [x, y] = toXY((i + 0.5) / N * cv.w, (j + 0.5) / N * cv.h); const v = land().f(x, y) - land().fmin; vals[j * N + i] = v; lo = Math.min(lo, v); hi = Math.max(hi, v); } for (let k = 0; k < N * N; k++) { const t = Math.pow(Math.log1p(Math.max(0, vals[k])) / Math.log1p(hi), 0.55), c = Colors.viridisRGB(1 - t); im.data[k * 4] = c[0] * 0.55; im.data[k * 4 + 1] = c[1] * 0.55; im.data[k * 4 + 2] = c[2] * 0.6 + 20; im.data[k * 4 + 3] = 255; } og.putImageData(im, 0, 0); bg = off; }
      function reset() { runners = OPTS.map(([k, n, c]) => { const th = new Float64Array(start), g = new Float64Array(2); return { k, n, c, th, g, opt: new Optimizer(k, [{ p: th, g }], { lr: L.lr, momentum: L.mom }), trail: [[th[0], th[1]]], dead: false }; }); hist = {}; OPTS.forEach(o => { hist[o[0]] = []; }); steps = 0; }
      function stepAll() { steps++; for (const r of runners) { if (r.dead) continue; const [gx, gy] = land().g(r.th[0], r.th[1]); r.g[0] = gx; r.g[1] = gy; r.opt.lr = L.lr; r.opt.mom = L.mom; r.opt.step(); const d = land().dom; if (!Number.isFinite(r.th[0]) || !Number.isFinite(r.th[1]) || Math.abs(r.th[0]) > 60 || Math.abs(r.th[1]) > 60) { r.dead = true; continue; } r.trail.push([r.th[0], r.th[1]]); if (r.trail.length > 900) r.trail.shift(); const f = land().f(r.th[0], r.th[1]) - land().fmin; hist[r.k].push([steps, Math.log10(Math.max(1e-6, f) + 1e-6)]); } }
      function drawLand() {
        const { g, w, h: H } = cv; g.clearRect(0, 0, w, H); g.imageSmoothingEnabled = true; g.drawImage(bg, 0, 0, w, H);
        land().mins.forEach(([x, y]) => { const [px, py] = toPx(x, y); g.strokeStyle = '#fff'; g.lineWidth = 2; g.beginPath(); g.moveTo(px - 7, py - 7); g.lineTo(px + 7, py + 7); g.moveTo(px + 7, py - 7); g.lineTo(px - 7, py + 7); g.stroke(); });
        const [sx, sy] = toPx(start[0], start[1]); g.strokeStyle = '#d7b56d'; g.setLineDash([3, 3]); g.beginPath(); g.arc(sx, sy, 8, 0, 7); g.stroke(); g.setLineDash([]);
        for (const r of runners) { if (!L.show.has(r.k)) continue; g.strokeStyle = r.c; g.lineWidth = 2; g.globalAlpha = r.dead ? 0.4 : 0.95; g.beginPath(); r.trail.forEach(([x, y], i) => { const [px, py] = toPx(x, y); i ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke(); g.globalAlpha = 1; const [px, py] = toPx(r.th[0], r.th[1]); if (!r.dead) { g.beginPath(); g.arc(px, py, 6, 0, 7); g.fillStyle = r.c; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke(); } }
        g.fillStyle = '#ece6d8'; g.font = '12px "Instrument Sans"'; g.fillText(land().name + ' — ' + land().hint, 10, 18); g.fillStyle = '#d7b56d'; g.fillText('click anywhere to set the start point  ·  × = minimum', 10, H - 10);
      }
      cv.c.addEventListener('pointerdown', e => { const [px, py] = cv.xy(e); start = toXY(px, py); reset(); });
      const lrSl = Ui.slider({ label: 'learning rate η (log scale)', min: -4, max: 0.3, step: 0.05, value: Math.log10(L.lr), fmt: v => Math.pow(10, v).toPrecision(2), on: v => { L.lr = Math.pow(10, v); } });
      const landSel = Ui.select('Loss landscape', Object.keys(LAND).map(k => ({ v: k, label: LAND[k].name })), L.key, v => { L.key = v; start = land().start.slice(); L.lr = land().lr; lrSl.set(Math.log10(L.lr), true); paintBg(); reset(); });
      const runBtn = Ui.btn('Run', () => { L.run = !L.run; runBtn.textContent = L.run ? 'Pause' : 'Run'; }, 'pri');
      const chks = h('div', { class: 'row' }, OPTS.map(([k, n, c]) => { const el = Ui.chk('', true, v => { v ? L.show.add(k) : L.show.delete(k); }); el.appendChild(h('span', { style: 'color:' + c + ';font-weight:700' }, n)); return el; }));
      const status = h('div', { class: 'row', style: 'font-size:12.5px' });
      landBox.appendChild(Ui.grid('g2', Ui.card('Trajectories', cv.c), h('div', null, Ui.card('Controls', landSel, Ui.row(runBtn, Ui.btn('Step ×1', stepAll), Ui.btn('Reset', reset)), Ui.row(lrSl, Ui.slider({ label: 'momentum μ', min: 0, max: 0.99, step: 0.01, value: L.mom, on: v => { L.mom = v; } }), Ui.slider({ label: 'steps / frame', min: 1, max: 20, step: 1, value: L.speed, on: v => { L.speed = v; } })), h('div', { style: 'margin-top:8px' }, chks)), Ui.card('Convergence', cvC.c), Ui.card('Status', status))));
      paintBg(); reset(); let fr = 0;
      /* ---------- MLP arena ---------- */
      const M = { ds: 'spiral', bn: false, drop: 0, lr: 0.01, hid: 32, batch: 64, speed: 5, run: false, noise: 0.5 }; let data, trn, tst, nets = [], stats = [], step = 0, rngB = new RNG(3);
      const cvs = OPTS.map(() => Ui.canvas(190, 190)), cvLoss = Ui.canvas(520, 220), cvAcc = Ui.canvas(520, 220), lossCh = new LineChart(cvLoss, { xlabel: 'training step', title: 'training loss (smoothed)', yMin: 0 }), accCh = new LineChart(cvAcc, { xlabel: 'training step', title: 'test accuracy %', yMin: 0, yMax: 100 });
      function buildM() {
        const r = new RNG(11); data = make2D(M.ds, 300, M.noise, r); const idx = r.shuffle(data.X.map((_, i) => i)), nt = Math.round(data.X.length * 0.75); trn = { X: idx.slice(0, nt).map(i => data.X[i]), y: idx.slice(0, nt).map(i => data.y[i]), K: data.K }; tst = { X: idx.slice(nt).map(i => data.X[i]), y: idx.slice(nt).map(i => data.y[i]), K: data.K };
        nets = OPTS.map(([k]) => { const net = new Net([2, M.hid, M.hid, data.K], { act: 'relu', bn: M.bn, dropout: M.drop, rng: new RNG(5) }); return { k, net, opt: new Optimizer(k, net.params, { lr: M.lr, momentum: 0.9 }) }; }); stats = OPTS.map(() => ({ ema: null, loss: [], acc: [] })); step = 0; rngB = new RNG(3);
      }
      function trainStep() {
        const n = Math.min(M.batch, trn.X.length), X = new Float64Array(n * 2), y = []; for (let i = 0; i < n; i++) { const j = rngB.int(trn.X.length); X[i * 2] = trn.X[j][0]; X[i * 2 + 1] = trn.X[j][1]; y.push(trn.y[j]); } step++;
        const tx = flat(tst.X);
        nets.forEach((m, i) => { m.opt.lr = M.lr; m.net.zeroGrad(); const out = m.net.forward(X, n, true), r = softmaxCE(out, n, data.K, y); m.net.backward(r.d); m.opt.step(); const s = stats[i]; s.ema = s.ema === null ? r.loss : 0.95 * s.ema + 0.05 * r.loss; if (step % 4 === 0) s.loss.push([step, s.ema]); if (step % 8 === 0) { const o = m.net.forward(tx, tst.X.length, false), rr = softmaxCE(o, tst.X.length, data.K, tst.y); s.acc.push([step, rr.acc * 100]); s.last = rr.acc; } });
      }
      function drawArena() { nets.forEach((m, i) => { const { g, w, h: H } = cvs[i]; g.clearRect(0, 0, w, H); drawBoundary(g, m.net, data.K, w, H, tst, 26, 0.55, true); g.fillStyle = '#0a1128cc'; g.fillRect(0, 0, w, 20); g.fillStyle = OPTS[i][2]; g.font = '700 12px "Instrument Sans"'; g.fillText(OPTS[i][1] + '  ' + (stats[i].last !== undefined ? (stats[i].last * 100).toFixed(0) + '% test' : ''), 6, 14); }); }
      const mRun = Ui.btn('Train all', () => { M.run = !M.run; mRun.textContent = M.run ? 'Pause' : 'Train all'; }, 'pri');
      const bnChk = Ui.chk('BatchNorm', false, v => { M.bn = v; buildM(); }), dsSel = Ui.select('Dataset', Object.keys(DSN).map(k => ({ v: k, label: DSN[k] })), M.ds, v => { M.ds = v; buildM(); });
      const grid = h('div', { class: 'grid', style: 'grid-template-columns:repeat(3,1fr);gap:8px' }, cvs.map(c => c.c));
      mlpBox.appendChild(Ui.grid('g2', h('div', null, Ui.card('Decision boundaries on the held-out points (same init, same mini-batches)', grid)), h('div', null, Ui.card('Controls', Ui.row(mRun, Ui.btn('Reset', buildM)), Ui.row(dsSel, bnChk), Ui.row(Ui.slider({ label: 'Dropout p (hidden layers)', min: 0, max: 0.7, step: 0.05, value: 0, on: v => { M.drop = v; buildM(); } }), Ui.slider({ label: 'learning rate', min: -3.5, max: -0.5, step: 0.05, value: -2, fmt: v => Math.pow(10, v).toPrecision(2), on: v => { M.lr = Math.pow(10, v); } })), Ui.row(Ui.slider({ label: 'data noise', min: 0.1, max: 1.2, step: 0.1, value: M.noise, on: v => { M.noise = v; buildM(); } }), Ui.slider({ label: 'steps / frame', min: 1, max: 15, step: 1, value: M.speed, on: v => { M.speed = v; } }))), Ui.card(null, cvLoss.c), Ui.card(null, cvAcc.c, Ui.legend(OPTS.map(o => [o[2], o[1]]))))));
      buildM();
      ctx.raf(() => {
        if (landBox.style.display !== 'none') {
          if (L.run) for (let i = 0; i < L.speed; i++) stepAll(); drawLand();
          if (fr++ % 4 === 0) { chart.draw(runners.filter(r => L.show.has(r.k)).map(r => ({ name: r.n, color: r.c, pts: hist[r.k], width: 2 })), { yMin: -6.2 }); status.innerHTML = ''; runners.forEach(r => { const f = land().f(r.th[0], r.th[1]) - land().fmin; status.appendChild(h('span', { class: 'tag', style: 'border:1px solid ' + r.c }, r.n + ': ' + (r.dead ? 'diverged ×' : 'f−f* = ' + (f < 1e-4 ? f.toExponential(1) : f.toFixed(3))))); }); }
        } else { if (M.run) for (let i = 0; i < M.speed; i++) trainStep(); if (fr++ % 4 === 0) { drawArena(); lossCh.draw(OPTS.map((o, i) => ({ name: o[1], color: o[2], pts: stats[i].loss })), { yMin: 0 }); accCh.draw(OPTS.map((o, i) => ({ name: o[1], color: o[2], pts: stats[i].acc })), { yMin: 0, yMax: 100 }); } }
      });
    },
  });

  /* ============================================================ T14 ===== */
  Lab.register({
    id: 't14', unit: 5, num: 'T14', title: 'Back-propagation, step by step',
    tag: 'See the chain rule run on a tiny network with real numbers and verify every gradient numerically — then watch gradients vanish in a deep sigmoid net.',
    star: 'Stage-by-stage walkthrough · live gradient check · per-layer gradient meter (vanishing gradients)',
    idea: `<p>Back-propagation is the <b>chain rule</b> applied layer by layer from the loss back to every weight. The notebook codes it in NumPy and shows a decision boundary on two moons. Here the first tab walks through <b>one forward and one backward pass on a 2-3-1 network</b>, one stage at a time, with the actual numbers substituted into each formula and a <b>finite-difference check</b> confirming each gradient. The second tab trains a deeper MLP and shows the <b>gradient magnitude per layer</b>, which makes the vanishing-gradient problem (sigmoid/tanh) and its cure (ReLU) visible.</p>`,
    tryIt: ['Walkthrough: press <b>Next stage</b> and compare the formula with the numbers on the network. At the end the table shows analytic vs numerical gradients (differences ≈ 1e-10).',
      'Change <b>x₁, x₂, target y</b> and the activation: with sigmoid the hidden deltas δ¹ shrink because σ′ ≤ 0.25.',
      'Press <b>Update ×100</b>: the loss falls and the prediction approaches the target.',
      'Training tab: set depth to 6, activation <b>sigmoid</b>, press Train — the gradient meter shows layer 1 with gradients orders of magnitude below layer 6. Switch to ReLU: the profile flattens and the boundary learns.',
      'Press <b>Gradient check</b>: relative error between back-prop and finite differences on random weights (should be ≲ 1e-6).'],
    hood: `<pre>forward : z¹ = W¹ᵀx + b¹ ,  a¹ = φ(z¹) ,  z² = W²ᵀa¹ + b² ,  ŷ = σ(z²) ,  L = −[y ln ŷ + (1−y) ln(1−ŷ)]
backward: δ² = ∂L/∂z² = ŷ − y                        (sigmoid + cross-entropy cancel)
          ∂L/∂W² = a¹ δ² ,   ∂L/∂b² = δ²
          δ¹ = (W² δ²) ⊙ φ′(z¹)                       (chain rule through W² and the activation)
          ∂L/∂W¹ = x δ¹ᵀ ,   ∂L/∂b¹ = δ¹
update  : W ← W − η ∂L/∂W
check   : (L(w+ε) − L(w−ε)) / 2ε  ≈  ∂L/∂w</pre>
    <p>Vanishing gradients: δ at layer ℓ contains the product of all upstream W·φ′ factors; with σ′ ≤ ¼ the product shrinks geometrically with depth.</p>`,
    quiz: [
      { q: 'For a sigmoid output with cross-entropy loss, δ² = ∂L/∂z² equals…', o: ['ŷ·(1−ŷ)', 'ŷ − y', '−ln ŷ', 'y·ŷ'], a: 1, why: 'The σ′ term cancels against the 1/ŷ(1−ŷ) of the cross-entropy, leaving ŷ − y.' },
      { q: 'Why do deep sigmoid nets suffer vanishing gradients?', o: ['W is too large', 'each layer multiplies the gradient by σ′ ≤ 0.25 (and W)', 'ReLU is used', 'the loss is convex'], a: 1, why: 'The backward signal is repeatedly multiplied by small derivatives, so early layers receive tiny updates.' },
      { q: 'A numerical gradient check compares back-prop gradients with…', o: ['random numbers', 'central finite differences of the loss', 'the weights', 'the learning rate'], a: 1, why: '(L(w+ε) − L(w−ε))/2ε approximates ∂L/∂w and catches back-prop bugs.' }],
    mount(root, ctx) {
      const tabs = Ui.seg([{ v: 'walk', label: '1 · Walkthrough (2-3-1)' }, { v: 'train', label: '2 · Train & watch gradients' }], 'walk', v => { walkBox.style.display = v === 'walk' ? '' : 'none'; trainBox.style.display = v === 'train' ? '' : 'none'; });
      const walkBox = h('div'), trainBox = h('div', { style: 'display:none' }); root.appendChild(h('div', { style: 'margin-bottom:14px' }, tabs)); root.appendChild(walkBox); root.appendChild(trainBox);
      /* ---------- walkthrough ---------- */
      const ACTS = { sigmoid: [z => 1 / (1 + Math.exp(-z)), a => a * (1 - a)], tanh: [z => Math.tanh(z), a => 1 - a * a], relu: [z => Math.max(0, z), (a, z) => (z > 0 ? 1 : 0)] };
      const rnd = new RNG(4); const S = { x: [0.8, -0.5], y: 1, act: 'tanh', lr: 0.5, stage: 0, W1: [[0.4, -0.6, 0.3], [0.2, 0.5, -0.7]], b1: [0.1, -0.1, 0.05], W2: [0.6, -0.4, 0.5], b2: 0.1 }; let lossHist = [];
      const sg = z => 1 / (1 + Math.exp(-z));
      function fwd(P = S) { const [f, d] = ACTS[S.act]; const z1 = [0, 1, 2].map(j => P.b1[j] + P.x[0] * P.W1[0][j] + P.x[1] * P.W1[1][j]), a1 = z1.map(f), z2 = P.b2 + a1.reduce((s, a, j) => s + a * P.W2[j], 0), a2 = sg(z2), L = -(P.y * Math.log(a2 + 1e-12) + (1 - P.y) * Math.log(1 - a2 + 1e-12)); return { z1, a1, z2, a2, L }; }
      function bwd() { const F = fwd(), [, d] = ACTS[S.act], d2 = F.a2 - S.y, dW2 = F.a1.map(a => a * d2), d1 = F.a1.map((a, j) => S.W2[j] * d2 * d(a, F.z1[j])), dW1 = [0, 1].map(i => d1.map(v => v * S.x[i])); return { F, d2, dW2, db2: d2, d1, db1: d1.slice(), dW1 }; }
      function numGrad() { const out = {}; const e = 1e-6; const L = () => fwd().L; const num = (get, set) => { const o = get(); set(o + e); const a = L(); set(o - e); const b = L(); set(o); return (a - b) / (2 * e); }; out.W2 = [0, 1, 2].map(j => num(() => S.W2[j], v => { S.W2[j] = v; })); out.W1 = [0, 1].map(i => [0, 1, 2].map(j => num(() => S.W1[i][j], v => { S.W1[i][j] = v; }))); out.b2 = num(() => S.b2, v => { S.b2 = v; }); return out; }
      const cvW = Ui.canvas(660, 420), txt = h('div', { class: 'body', style: 'min-height:180px' }), gcBox = h('div');
      const f4 = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(3), n3 = v => v.toFixed(3);
      const stageNames = ['inputs', 'hidden pre-activations z¹', 'hidden activations a¹', 'output ŷ', 'loss L', 'output error δ² and ∂L/∂W²', 'hidden error δ¹ (chain rule)', '∂L/∂W¹', 'gradient step'];
      function drawWalk() {
        const { g, w, h: H } = cvW, B = bwd(), F = B.F, st = S.stage; g.clearRect(0, 0, w, H); const X = [90, 330, 570], inY = [140, 280], hY = [80, 210, 340], oY = 210;
        const edge = (x1, y1, x2, y2, wv, gv, tf) => { g.strokeStyle = wv >= 0 ? '#d9825b' : '#7aa2c8'; g.globalAlpha = 0.35 + 0.5 * Math.min(1, Math.abs(wv)); g.lineWidth = 1 + 3.5 * Math.min(1, Math.abs(wv)); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke(); g.globalAlpha = 1; const mx = x1 + (x2 - x1) * tf, my = y1 + (y2 - y1) * tf; g.font = '10.5px "JetBrains Mono",monospace'; g.textAlign = 'center'; g.fillStyle = '#cfc8b8'; g.fillText('w=' + wv.toFixed(2), mx, my - 3); if (gv !== null) { g.fillStyle = '#d7b56d'; g.fillText('∂=' + gv.toFixed(3), mx, my + 9); } };
        for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) edge(X[0], inY[i], X[1], hY[j], S.W1[i][j], st >= 7 ? B.dW1[i][j] : null, i === 0 ? 0.3 : 0.45);
        for (let j = 0; j < 3; j++) edge(X[1], hY[j], X[2], oY, S.W2[j], st >= 5 ? B.dW2[j] : null, 0.55);
        const node = (x, y, label, val, col, sub) => { g.beginPath(); g.arc(x, y, 30, 0, 7); g.fillStyle = col; g.fill(); g.lineWidth = 2; g.strokeStyle = '#100e0c'; g.stroke(); g.fillStyle = '#14110d'; g.font = '700 12px "Instrument Sans"'; g.textAlign = 'center'; g.fillText(label, x, y - 5); g.font = '12px "JetBrains Mono",monospace'; g.fillText(val, x, y + 10); if (sub) { g.fillStyle = '#d7b56d'; g.font = '11px "JetBrains Mono",monospace'; g.fillText(sub, x, y + 46); } };
        S.x.forEach((v, i) => node(X[0], inY[i], 'x' + (i + 1), n3(v), '#8a8171'));
        F.z1.forEach((z, j) => node(X[1], hY[j], 'h' + (j + 1), st < 1 ? '?' : st < 2 ? 'z=' + n3(z) : n3(F.a1[j]), st < 1 ? '#4a4338' : '#86b894', st >= 6 ? 'δ¹=' + f4(B.d1[j]) : (st >= 1 ? 'z=' + n3(z) : '')));
        node(X[2], oY, 'ŷ', st < 3 ? '?' : n3(F.a2), st < 3 ? '#4a4338' : '#d7b56d', st >= 5 ? 'δ²=' + f4(B.d2) : (st >= 4 ? 'L=' + n3(F.L) : ''));
        g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.textAlign = 'left'; g.fillText('target y = ' + S.y, 520, 275); g.fillText('stage ' + st + '/8: ' + stageNames[st], 10, 20); g.fillStyle = '#d9825b'; g.fillText('■ positive weight', 10, H - 22); g.fillStyle = '#7aa2c8'; g.fillText('■ negative weight', 130, H - 22); g.fillStyle = '#d7b56d'; g.fillText('∂ = ∂L/∂w (after the backward stages)', 250, H - 22);
      }
      function describe() {
        const B = bwd(), F = B.F, st = S.stage, act = S.act === 'sigmoid' ? 'σ' : S.act === 'tanh' ? 'tanh' : 'ReLU'; let html = '';
        if (st === 0) html = `<p><b>Inputs.</b> x = (${n3(S.x[0])}, ${n3(S.x[1])}), target y = ${S.y}. Press <b>Next stage</b> to run the forward pass.</p>`;
        if (st === 1) html = `<p><b>Forward 1: pre-activations.</b> <code>z¹ⱼ = b¹ⱼ + x₁W¹₁ⱼ + x₂W¹₂ⱼ</code></p><p>e.g. z¹₁ = ${n3(S.b1[0])} + ${n3(S.x[0])}·${n3(S.W1[0][0])} + ${n3(S.x[1])}·${n3(S.W1[1][0])} = <b>${n3(F.z1[0])}</b></p>`;
        if (st === 2) html = `<p><b>Forward 2: activations.</b> <code>a¹ = ${act}(z¹)</code></p><p>a¹ = (${F.a1.map(n3).join(', ')})</p>`;
        if (st === 3) html = `<p><b>Forward 3: output.</b> <code>z² = b² + Σ a¹ⱼ W²ⱼ</code>, <code>ŷ = σ(z²)</code></p><p>z² = ${n3(S.b2)} + ${F.a1.map((a, j) => n3(a) + '·' + n3(S.W2[j])).join(' + ')} = ${n3(F.z2)} → ŷ = <b>${n3(F.a2)}</b></p>`;
        if (st === 4) html = `<p><b>Loss.</b> <code>L = −[y ln ŷ + (1−y) ln(1−ŷ)]</code> = <b>${n3(F.L)}</b></p>`;
        if (st === 5) html = `<p><b>Backward 1: output layer.</b> Sigmoid + cross-entropy collapse to <code>δ² = ŷ − y</code> = ${n3(F.a2)} − ${S.y} = <b>${f4(B.d2)}</b>.</p><p><code>∂L/∂W²ⱼ = a¹ⱼ δ²</code> = (${B.dW2.map(n3).join(', ')}), <code>∂L/∂b² = δ²</code></p>`;
        if (st === 6) html = `<p><b>Backward 2: hidden layer (chain rule).</b> <code>δ¹ⱼ = W²ⱼ · δ² · ${act}′(z¹ⱼ)</code></p><p>e.g. δ¹₁ = ${n3(S.W2[0])} · ${n3(B.d2)} · ${n3(ACTS[S.act][1](F.a1[0], F.z1[0]))} = <b>${f4(B.d1[0])}</b>. Notice the factor ${act}′ — for σ it is ≤ 0.25.</p>`;
        if (st === 7) html = `<p><b>Backward 3: first-layer weights.</b> <code>∂L/∂W¹ᵢⱼ = xᵢ δ¹ⱼ</code></p><p>∂L/∂W¹₁ = (${B.dW1[0].map(n3).join(', ')}) · ∂L/∂W¹₂ = (${B.dW1[1].map(n3).join(', ')})</p>`;
        if (st === 8) html = `<p><b>Update.</b> <code>W ← W − η ∂L/∂W</code> with η = ${S.lr}. Press <b>Apply update</b> (or ×100) and the loss falls.</p>`;
        txt.innerHTML = html;
        const N = numGrad(); const rows = [...B.dW2.map((v, j) => ['∂L/∂W²' + (j + 1), v, N.W2[j]]), ...[0, 1].flatMap(i => [0, 1, 2].map(j => ['∂L/∂W¹' + (i + 1) + (j + 1), B.dW1[i][j], N.W1[i][j]])), ['∂L/∂b²', B.db2, N.b2]]; const mx = Math.max(...rows.map(r => Math.abs(r[1] - r[2])));
        gcBox.innerHTML = ''; gcBox.appendChild(h('div', { class: 'note', style: 'border-color:' + (mx < 1e-7 ? 'var(--good)' : 'var(--bad)') }, `Gradient check: max |back-prop − finite-difference| = ${mx.toExponential(1)} ${mx < 1e-7 ? '' : ''}`)); gcBox.appendChild(Ui.table(['gradient', 'back-prop', 'numerical'], rows.slice(0, 7).map(r => [r[0], r[1].toFixed(8), r[2].toFixed(8)]), { maxH: '170px' }));
      }
      const refreshW = () => { drawWalk(); describe(); };
      const applyUpdate = () => { const B = bwd(); for (let j = 0; j < 3; j++) { S.W2[j] -= S.lr * B.dW2[j]; S.b1[j] -= S.lr * B.db1[j]; for (let i = 0; i < 2; i++) S.W1[i][j] -= S.lr * B.dW1[i][j]; } S.b2 -= S.lr * B.db2; lossHist.push([lossHist.length, fwd().L]); };
      const sx1 = Ui.slider({ label: 'x₁', min: -1.5, max: 1.5, step: 0.1, value: S.x[0], on: v => { S.x[0] = v; refreshW(); } }), sx2 = Ui.slider({ label: 'x₂', min: -1.5, max: 1.5, step: 0.1, value: S.x[1], on: v => { S.x[1] = v; refreshW(); } });
      const cvH = Ui.canvas(480, 150), hch = new LineChart(cvH, { xlabel: 'update', title: 'loss during repeated updates', yMin: 0 });
      walkBox.appendChild(Ui.grid('g21', h('div', null, Ui.card('The network', cvW.c, h('div', { class: 'row', style: 'margin-top:10px' }, Ui.btn('Next stage', () => { S.stage = Math.min(8, S.stage + 1); refreshW(); }, 'pri'), Ui.btn('Back', () => { S.stage = Math.max(0, S.stage - 1); refreshW(); }), Ui.btn('Show all', () => { S.stage = 8; refreshW(); }))), Ui.card('What is happening', txt)),
        h('div', null, Ui.card('Inputs & settings', Ui.row(sx1, sx2), Ui.row(Ui.select('hidden activation', ['tanh', 'sigmoid', 'relu'], S.act, v => { S.act = v; refreshW(); }), Ui.select('target y', ['1', '0'], '1', v => { S.y = +v; refreshW(); })), Ui.row(Ui.slider({ label: 'learning rate η', min: 0.05, max: 2, step: 0.05, value: S.lr, on: v => { S.lr = v; } })), Ui.row(Ui.btn('Apply update', () => { applyUpdate(); refreshW(); }, 'pri'), Ui.btn('Update ×100', () => { for (let i = 0; i < 100; i++) applyUpdate(); refreshW(); }), Ui.btn('Randomise weights', () => { S.W1 = [0, 1].map(() => [0, 1, 2].map(() => rnd.normal() * 0.7)); S.W2 = [0, 1, 2].map(() => rnd.normal() * 0.7); S.b1 = [0, 1, 2].map(() => rnd.normal() * 0.2); S.b2 = rnd.normal() * 0.2; lossHist = []; refreshW(); }))), Ui.card('Numerical verification', gcBox), Ui.card(null, cvH.c))));
      refreshW();

      /* ---------- train & gradients ---------- */
      const T = { ds: 'moons', depth: 3, width: 16, act: 'relu', lr: 0.1, run: false, noise: 0.5, batchFull: true }; let net, opt, dat, step = 0, losses = [], accs = [], gnorm = [], gcMsg = '';
      const cvB = Ui.canvas(440, 440), cvG = Ui.canvas(440, 220), cvLc = Ui.canvas(440, 200), lch = new LineChart(cvLc, { xlabel: 'epoch', title: 'training loss', yMin: 0 });
      function buildT() { const r = new RNG(21); dat = make2D(T.ds, 300, T.noise, r); const sizes = [2, ...Array(T.depth).fill(T.width), dat.K]; net = new Net(sizes, { act: T.act, rng: new RNG(2) }); opt = new Optimizer('sgd', net.params, { lr: T.lr }); step = 0; losses = []; accs = []; gnorm = []; }
      function trainT() { const X = flat(dat.X), n = dat.X.length; net.zeroGrad(); const o = net.forward(X, n, true), r = softmaxCE(o, n, dat.K, dat.y); net.backward(r.d); gnorm = net.layers.map(ly => Math.sqrt(ly.gW.reduce((s, v) => s + v * v, 0) / ly.gW.length)); opt.lr = T.lr; opt.step(); step++; if (step % 3 === 0) { losses.push([step, r.loss]); accs.push(r.acc); } return r; }
      function gradCheck() { const r = new RNG(9), n = 20, X = flat(dat.X.slice(0, n)), y = dat.y.slice(0, n); const f = () => softmaxCE(net.forward(X, n, false), n, dat.K, y).loss; net.zeroGrad(); const o = net.forward(X, n, false), rr = softmaxCE(o, n, dat.K, y); net.backward(rr.d); let mx = 0, cnt = 0; for (const q of net.params) for (let k = 0; k < 5; k++) { const i = r.int(q.p.length), o0 = q.p[i]; q.p[i] = o0 + 1e-5; const a = f(); q.p[i] = o0 - 1e-5; const b = f(); q.p[i] = o0; const num = (a - b) / 2e-5, an = q.g[i]; mx = Math.max(mx, Math.abs(num - an) / (Math.abs(num) + Math.abs(an) + 1e-8)); cnt++; } gcMsg = `checked ${cnt} random weights: max relative error ${mx.toExponential(1)} ${mx < 1e-5 ? 'back-prop is correct' : ''}`; Ui.toast(gcMsg); }
      function drawT() {
        const { g, w, h: H } = cvB; g.clearRect(0, 0, w, H); drawBoundary(g, net, dat.K, w, H, dat, 40, 0.6, true);
        const c = cvG, g2 = c.g; g2.clearRect(0, 0, c.w, c.h); g2.fillStyle = '#9b9384'; g2.font = '11.5px "Instrument Sans"'; g2.fillText('RMS gradient per layer (log scale) — layer 1 is the input side', 10, 14); if (gnorm.length) { const lg = gnorm.map(v => Math.log10(v + 1e-12)), hi = Math.max(...lg, -1) + 0.3, lo = Math.min(...lg, hi - 4) - 0.3, bw = (c.w - 40) / gnorm.length; lg.forEach((v, i) => { const hh = (v - lo) / (hi - lo) * (c.h - 50); g2.fillStyle = Colors.viridis(i / Math.max(1, gnorm.length - 1)); g2.fillRect(24 + i * bw + 4, c.h - 22 - hh, bw - 8, hh); g2.fillStyle = '#ece6d8'; g2.font = '10px "JetBrains Mono",monospace'; g2.textAlign = 'center'; g2.fillText(gnorm[i].toExponential(1), 24 + i * bw + bw / 2, c.h - 26 - hh); g2.fillStyle = '#9b9384'; g2.fillText('L' + (i + 1), 24 + i * bw + bw / 2, c.h - 8); }); g2.textAlign = 'left'; }
      }
      const runB = Ui.btn('Train', () => { T.run = !T.run; runB.textContent = T.run ? 'Pause' : 'Train'; }, 'pri');
      trainBox.appendChild(Ui.grid('g2', h('div', null, Ui.card('Decision boundary (full-batch gradient descent, hand-coded back-prop)', cvB.c), Ui.card('Gradient meter', cvG.c)), h('div', null, Ui.card('Network', Ui.row(runB, Ui.btn('Step ×1', trainT), Ui.btn('Reset', buildT), Ui.btn('Gradient check', gradCheck)), Ui.row(Ui.select('dataset', Object.keys(DSN).filter(k => k !== 'spiral').concat('spiral').map(k => ({ v: k, label: DSN[k] })), T.ds, v => { T.ds = v; buildT(); }), Ui.select('activation', ['relu', 'tanh', 'sigmoid'], T.act, v => { T.act = v; buildT(); })), Ui.row(Ui.slider({ label: 'hidden layers (depth)', min: 1, max: 8, step: 1, value: T.depth, on: v => { T.depth = v; buildT(); } }), Ui.slider({ label: 'width', min: 4, max: 48, step: 4, value: T.width, on: v => { T.width = v; buildT(); } })), Ui.row(Ui.slider({ label: 'learning rate', min: -2.5, max: 0.5, step: 0.05, value: Math.log10(T.lr), fmt: v => Math.pow(10, v).toPrecision(2), on: v => { T.lr = Math.pow(10, v); } })), h('div', { class: 'row', id: 'gcLine', style: 'margin-top:8px' })), Ui.card(null, cvLc.c))));
      buildT(); let f = 0;
      ctx.raf(() => { if (trainBox.style.display === 'none') return; if (T.run) for (let i = 0; i < 6; i++) trainT(); drawT(); if (f++ % 4 === 0) { lch.draw([{ name: 'loss', color: '#d7b56d', pts: losses }], { yMin: 0 }); hch.draw([{ name: 'L', color: '#d9825b', pts: lossHist }], { yMin: 0 }); } });
      ctx.raf(() => { if (walkBox.style.display === 'none') return; if (f++ % 8 === 0) hch.draw([{ name: 'loss', color: '#d9825b', pts: lossHist }], { yMin: 0 }); });
    },
  });

  /* ============================================================ T15 ===== */
  class CNN {
    constructor(F1, F2, K, rng) {
      this.F1 = F1; this.F2 = F2; this.K = K; this.D = F2 * 36; const he = (n, f) => Float64Array.from({ length: n }, () => rng.normal() * Math.sqrt(2 / f));
      this.W1 = he(F1 * 9, 9); this.b1 = new Float64Array(F1); this.W2 = he(F2 * F1 * 9, F1 * 9); this.b2 = new Float64Array(F2); this.Wd = he(K * this.D, this.D); this.bd = new Float64Array(K);
      this.g = { W1: new Float64Array(this.W1.length), b1: new Float64Array(F1), W2: new Float64Array(this.W2.length), b2: new Float64Array(F2), Wd: new Float64Array(this.Wd.length), bd: new Float64Array(K) };
      this.rng = rng; this.drop = 0.3;
    }
    get params() { return [['W1', this.W1], ['b1', this.b1], ['W2', this.W2], ['b2', this.b2], ['Wd', this.Wd], ['bd', this.bd]].map(([k, p]) => ({ p, g: this.g[k] })); }
    zero() { for (const k in this.g) this.g[k].fill(0); }
    forward(x, train) {
      const F1 = this.F1, F2 = this.F2, K = this.K, S = 24, H = 12, Q = 6, c = {}; c.x = x;
      const a1 = new Float64Array(F1 * S * S), z1 = new Float64Array(F1 * S * S);
      for (let f = 0; f < F1; f++) for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) { let s = this.b1[f]; for (let u = -1; u <= 1; u++) { const ii = i + u; if (ii < 0 || ii >= S) continue; for (let v = -1; v <= 1; v++) { const jj = j + v; if (jj < 0 || jj >= S) continue; s += this.W1[f * 9 + (u + 1) * 3 + v + 1] * x[ii * S + jj]; } } const o = f * S * S + i * S + j; z1[o] = s; a1[o] = s > 0 ? s : 0; }
      const p1 = new Float64Array(F1 * H * H), i1 = new Int32Array(F1 * H * H);
      for (let f = 0; f < F1; f++) for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) { let b = -1, bi = 0; for (let u = 0; u < 2; u++) for (let v = 0; v < 2; v++) { const o = f * S * S + (2 * i + u) * S + 2 * j + v; if (b < 0 || a1[o] > b) { b = a1[o]; bi = o; } } p1[f * H * H + i * H + j] = b; i1[f * H * H + i * H + j] = bi; }
      const a2 = new Float64Array(F2 * H * H), z2 = new Float64Array(F2 * H * H);
      for (let g = 0; g < F2; g++) for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) { let s = this.b2[g]; for (let f = 0; f < F1; f++) { const wb = (g * F1 + f) * 9, pb = f * H * H; for (let u = -1; u <= 1; u++) { const ii = i + u; if (ii < 0 || ii >= H) continue; for (let v = -1; v <= 1; v++) { const jj = j + v; if (jj < 0 || jj >= H) continue; s += this.W2[wb + (u + 1) * 3 + v + 1] * p1[pb + ii * H + jj]; } } } const o = g * H * H + i * H + j; z2[o] = s; a2[o] = s > 0 ? s : 0; }
      const p2 = new Float64Array(F2 * Q * Q), i2 = new Int32Array(F2 * Q * Q);
      for (let g = 0; g < F2; g++) for (let i = 0; i < Q; i++) for (let j = 0; j < Q; j++) { let b = -1, bi = 0; for (let u = 0; u < 2; u++) for (let v = 0; v < 2; v++) { const o = g * H * H + (2 * i + u) * H + 2 * j + v; if (b < 0 || a2[o] > b) { b = a2[o]; bi = o; } } p2[g * Q * Q + i * Q + j] = b; i2[g * Q * Q + i * Q + j] = bi; }
      const mask = new Float64Array(this.D), pd = new Float64Array(this.D); for (let d = 0; d < this.D; d++) { mask[d] = train && this.drop > 0 ? (this.rng.rand() < this.drop ? 0 : 1 / (1 - this.drop)) : 1; pd[d] = p2[d] * mask[d]; }
      const logits = new Float64Array(K); for (let k = 0; k < K; k++) { let s = this.bd[k]; for (let d = 0; d < this.D; d++) s += this.Wd[k * this.D + d] * pd[d]; logits[k] = s; }
      Object.assign(c, { a1, z1, p1, i1, a2, z2, p2, i2, mask, pd, logits }); return c;
    }
    backward(c, dl) {
      const F1 = this.F1, F2 = this.F2, K = this.K, S = 24, H = 12, Q = 6, D = this.D, G = this.g;
      const dp2 = new Float64Array(D); for (let k = 0; k < K; k++) { G.bd[k] += dl[k]; for (let d = 0; d < D; d++) { G.Wd[k * D + d] += dl[k] * c.pd[d]; dp2[d] += dl[k] * this.Wd[k * D + d] * c.mask[d]; } }
      const dz2 = new Float64Array(F2 * H * H); for (let d = 0; d < D; d++) { const o = c.i2[d]; if (c.z2[o] > 0) dz2[o] += dp2[d]; }
      const dp1 = new Float64Array(F1 * H * H);
      for (let g = 0; g < F2; g++) for (let i = 0; i < H; i++) for (let j = 0; j < H; j++) { const d = dz2[g * H * H + i * H + j]; if (d === 0) continue; G.b2[g] += d; for (let f = 0; f < F1; f++) { const wb = (g * F1 + f) * 9, pb = f * H * H; for (let u = -1; u <= 1; u++) { const ii = i + u; if (ii < 0 || ii >= H) continue; for (let v = -1; v <= 1; v++) { const jj = j + v; if (jj < 0 || jj >= H) continue; const wi = wb + (u + 1) * 3 + v + 1; G.W2[wi] += d * c.p1[pb + ii * H + jj]; dp1[pb + ii * H + jj] += d * this.W2[wi]; } } } }
      const dz1 = new Float64Array(F1 * S * S); for (let d = 0; d < F1 * H * H; d++) { const o = c.i1[d]; if (c.z1[o] > 0) dz1[o] += dp1[d]; }
      for (let f = 0; f < F1; f++) for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) { const d = dz1[f * S * S + i * S + j]; if (d === 0) continue; G.b1[f] += d; for (let u = -1; u <= 1; u++) { const ii = i + u; if (ii < 0 || ii >= S) continue; for (let v = -1; v <= 1; v++) { const jj = j + v; if (jj < 0 || jj >= S) continue; G.W1[f * 9 + (u + 1) * 3 + v + 1] += d * c.x[ii * S + jj]; } } }
    }
    probs(x) { return ML.softmax(Array.from(this.forward(x, false).logits)); }
  }

  Lab.register({
    id: 't15', unit: 5, num: 'T15', title: 'Face recognition with a CNN',
    tag: 'Train a convolutional network from scratch on the T9 faces, watch the filters form, ask where it looks (occlusion maps) and compare with PCA + Random Forest.',
    star: 'Live learned filters & feature maps · occlusion-sensitivity heat-map · data augmentation · robustness test vs T9',
    idea: `<p>A <b>CNN</b> learns its own features: small shared <b>convolution kernels</b> slide over the image, a ReLU keeps positive responses, <b>max-pooling</b> adds tolerance to small shifts, and a dense layer turns the final feature map into identity scores — all trained end-to-end by back-propagation (T14). The notebook trains a PyTorch CNN on Olivetti faces; here the <b>same 12-person gallery as T9</b> is used and every layer — convolution, pooling, back-prop — is hand-written JavaScript.</p>
    <p>New here: you can <i>see the learned 3×3 filters</i>, inspect <b>feature maps</b> for any test face, compute an <b>occlusion-sensitivity map</b> (which pixels does the network rely on?), toggle <b>data augmentation</b>, and run the same <b>sunglasses/mask/noise</b> stress test as T9.</p>`,
    tryIt: ['Press <b>Train</b>. Within ~10–20 epochs test accuracy usually passes 90 %. Watch the 3×3 filters turn from noise into edge/blob detectors.',
      'Pick a test face and press <b>Occlusion map</b>: bright regions are where hiding a 6×6 patch hurts the correct-class probability most — typically eyes, brows, hair-line (identity cues).',
      'Run the <b>stress test</b>, then retrain with <b>augmentation</b> ticked (random shifts / brightness) and compare robustness; try the T9 baseline too.',
      'Use fewer filters (4) vs more (12): capacity vs speed vs overfitting.'],
    hood: `<pre>input 24×24 → conv3×3(1→F1, pad 1) → ReLU → maxpool 2   (12×12×F1)
             → conv3×3(F1→F2, pad 1) → ReLU → maxpool 2   (6×6×F2)
             → dropout(0.3) → dense(36·F2 → 12) → softmax,   loss = cross-entropy, Adam
backward: dense → un-pool (route gradient to the arg-max position) → ReLU mask
          → conv grads  ∂L/∂W[f,u,v] = Σᵢⱼ δ[i,j] · x[i+u, j+v]     (correlation of input with δ)
          → pass δ to the previous layer with the flipped-kernel sum</pre>
    <p>Occlusion sensitivity: for each patch position, replace the patch by mid-gray and record <code>p(true) before − after</code>.</p>`,
    quiz: [
      { q: 'What does weight sharing in a convolution layer give you?', o: ['more parameters', 'the same detector applied at every position, with far fewer parameters', 'a deeper network', 'dropout'], a: 1, why: 'One small kernel is reused across the image, so detecting an edge anywhere costs the same few weights.' },
      { q: 'Max-pooling mainly provides…', o: ['tolerance to small translations and a smaller feature map', 'more channels', 'normalisation', 'a loss function'], a: 0, why: 'Only the strongest response in each window survives, so tiny shifts of a feature do not change the output.' },
      { q: 'An occlusion-sensitivity map shows…', o: ['the weights', 'which image regions the prediction depends on most', 'the training loss', 'the learning rate'], a: 1, why: 'It measures the drop in the true-class probability when each region is hidden.' }],
    mount(root, ctx) {
      const NID = 12, PER = 14, NTR = 8, S = 24; const D = Faces.make(NID, PER, 4); const trI = [], teI = []; D.X.forEach((_, i) => (D.k[i] < NTR ? trI : teI).push(i));
      const P = { F1: 8, F2: 12, lr: 0.003, aug: false, run: false, batch: 8, budget: 14 }; let cnn, opt, hist = { loss: [], acc: [] }, epoch = 0, perm = [], pos = 0, lossAcc = 0, lossN = 0, selTest = 0, occ = null, base = null, stressRes = null;
      const rng0 = new RNG(5);
      function build() { cnn = new CNN(P.F1, P.F2, NID, new RNG(1)); opt = new Optimizer('adam', cnn.params, { lr: P.lr }); hist = { loss: [], acc: [] }; epoch = 0; perm = []; pos = 0; occ = null; testAcc(); }
      function augment(img) { const o = new Float64Array(S * S), dx = rng0.int(3) - 1, dy = rng0.int(3) - 1, br = rng0.uniform(0.8, 1.2); for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { const ii = Math.max(0, Math.min(S - 1, i - dx)), jj = Math.max(0, Math.min(S - 1, j - dy)); o[j * S + i] = Math.max(0, Math.min(1, img[jj * S + ii] * br + rng0.normal() * 0.02)); } return o; }
      function trainBatch() {
        if (pos >= perm.length) { if (perm.length) { epoch++; hist.loss.push([epoch, lossAcc / Math.max(1, lossN)]); hist.acc.push([epoch, testAcc() * 100]); lossAcc = lossN = 0; } perm = rng0.shuffle(trI.slice()); pos = 0; }
        const B = Math.min(P.batch, perm.length - pos); cnn.zero();
        for (let b = 0; b < B; b++) { const i = perm[pos + b], x = P.aug ? augment(D.X[i]) : D.X[i], c = cnn.forward(x, true), p = ML.softmax(Array.from(c.logits)); lossAcc -= Math.log(p[D.y[i]] + 1e-12); lossN++; const dl = p.map((v, k) => (v - (k === D.y[i] ? 1 : 0)) / B); cnn.backward(c, dl); }
        pos += B; opt.lr = P.lr; opt.step();
      }
      const testAcc = (fn) => { let ok = 0; teI.forEach(i => { const x = fn ? fn(D.X[i]) : D.X[i], p = cnn.probs(x); if (p.indexOf(Math.max(...p)) === D.y[i]) ok++; }); return ok / teI.length; };
      /* visualisation */
      const cvF = Ui.canvas(420, 110), cvM = Ui.canvas(560, 250), cvP = Ui.canvas(420, 190), cvO = Ui.canvas(420, 200), cvL = Ui.canvas(480, 200), lch = new LineChart(cvL, { xlabel: 'epoch', title: 'training loss ● and test accuracy %' });
      function drawFilters() { const { g, w, h: H } = cvF; g.clearRect(0, 0, w, H); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('learned conv-1 kernels (3×3)', 6, 12); for (let f = 0; f < P.F1; f++) { const k = cnn.W1.subarray(f * 9, f * 9 + 9); let m = 1e-9; k.forEach(v => { m = Math.max(m, Math.abs(v)); }); const x = 6 + f * 50, y = 20; for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) { g.fillStyle = Colors.diverge(k[j * 3 + i] / m); g.fillRect(x + i * 14, y + j * 14, 13, 13); } g.fillStyle = '#6f6859'; g.fillText('f' + (f + 1), x + 12, y + 56); } }
      function drawMaps() {
        const { g, w, h: H } = cvM; g.clearRect(0, 0, w, H); const i = teI[selTest], x = D.X[i], c = cnn.forward(x, false); drawGray(g, x, S, S, 8, 22, 3); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('input', 8, 14); g.fillText('conv-1 + ReLU feature maps (24×24)', 90, 14);
        for (let f = 0; f < P.F1; f++) { let mx = 1e-9; for (let k = 0; k < S * S; k++) mx = Math.max(mx, c.a1[f * S * S + k]); drawGray(g, c.a1.subarray(f * S * S, (f + 1) * S * S), S, S, 90 + (f % 4) * 62, 22 + Math.floor(f / 4) * 62, 2.5, 0, mx); }
        g.fillStyle = '#9b9384'; g.fillText('conv-2 pooled maps (6×6)', 350, 14); for (let f = 0; f < P.F2; f++) { let mx = 1e-9; for (let k = 0; k < 36; k++) mx = Math.max(mx, c.p2[f * 36 + k]); drawGray(g, c.p2.subarray(f * 36, (f + 1) * 36), 6, 6, 350 + (f % 4) * 50, 22 + Math.floor(f / 4) * 50, 7, 0, mx); }
      }
      function drawPred() {
        const { g, w, h: H } = cvP; g.clearRect(0, 0, w, H); const i = teI[selTest], p = cnn.probs(D.X[i]), top = p.map((v, k) => [v, k]).sort((a, b) => b[0] - a[0]).slice(0, 4); drawGray(g, D.X[i], S, S, 8, 8, 4); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('true: P' + (D.y[i] + 1), 10, 122); top.forEach(([v, k], j) => { g.fillStyle = k === D.y[i] ? '#86b894' : '#6f6859'; g.fillRect(130, 12 + j * 32, 250 * v, 22); g.fillStyle = '#ece6d8'; g.font = '12px "JetBrains Mono",monospace'; g.fillText('P' + (k + 1) + ' ' + (v * 100).toFixed(0) + '%', 136 + 250 * v, 28 + j * 32); });
      }
      function occlusion() {
        const i = teI[selTest], x = D.X[i], y = D.y[i], p0 = cnn.probs(x)[y], R = 6, st = 2, n = Math.floor((S - R) / st) + 1, map = new Float64Array(n * n);
        for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const o = Float64Array.from(x); for (let j = 0; j < R; j++) for (let k = 0; k < R; k++) o[(a * st + j) * S + b * st + k] = 0.5; map[a * n + b] = p0 - cnn.probs(o)[y]; } occ = { map, n, st, R, p0 }; drawOcc();
      }
      function drawOcc() { const { g, w, h: H } = cvO; g.clearRect(0, 0, w, H); const i = teI[selTest]; drawGray(g, D.X[i], S, S, 8, 8, 7); if (!occ) { g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.fillText('press “Occlusion map” →', 190, 100); return; } const { map, n, st, R } = occ; let mx = 1e-9; map.forEach(v => { mx = Math.max(mx, Math.abs(v)); }); drawGray(g, D.X[i], S, S, 200, 8, 7); for (let a = 0; a < n; a++) for (let b = 0; b < n; b++) { const v = map[a * n + b] / mx; if (v > 0.05) { g.fillStyle = `rgba(217,130,91,${0.6 * v})`; g.fillRect(200 + (b * st + R / 2 - st / 2) * 7, 8 + (a * st + R / 2 - st / 2) * 7, st * 7, st * 7); } } g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('face', 8, 190); g.fillText('sensitivity (orange = important)', 200, 190); }
      /* baseline: PCA + RF */
      function baseline() {
        baseBtn.textContent = 'training RF…'; setTimeout(() => {
          const pca = new ML.PCA(60).fit(trI.map(i => D.X[i])), k = 30, feat = x => { const z = pca.transform(x, k); for (let c = 0; c < z.length; c++) z[c] /= Math.sqrt(pca.vals[c] + 1e-9); return z; }, F = trI.map(i => feat(D.X[i])), y = trI.map(i => D.y[i]), rng = new RNG(1), trees = [];
          for (let t = 0; t < 60; t++) { const idx = Array.from({ length: F.length }, () => rng.int(F.length)); trees.push(new ML.ClsTree({ maxDepth: 12, K: NID, mtry: 6, rng }).fit(F, y, null, idx)); }
          const pred = x => { const f = feat(x), p = new Float64Array(NID); trees.forEach(t => { const q = t.proba(f); for (let c = 0; c < NID; c++) p[c] += q[c]; }); return p.indexOf(Math.max(...p)); };
          base = { pred, acc: teI.reduce((s, i) => s + (pred(D.X[i]) === D.y[i] ? 1 : 0), 0) / teI.length }; baseBtn.textContent = 'PCA + Random Forest baseline'; baseOut.textContent = `PCA+RF test accuracy: ${(base.acc * 100).toFixed(1)}%   ·   CNN: ${(testAcc() * 100).toFixed(1)}%`;
        }, 30);
      }
      function stress() {
        const conds = ['none', 'sunglasses', 'mask', 'noise', 'dark'], r = new RNG(8); const rows = [['CNN', ...conds.map(c => { let ok = 0; teI.forEach(i => { const x = Faces.occlude(D.X[i], c, r), p = cnn.probs(x); if (p.indexOf(Math.max(...p)) === D.y[i]) ok++; }); return ok / teI.length; })]];
        if (base) rows.push(['PCA + RF', ...conds.map(c => { let ok = 0; teI.forEach(i => { if (base.pred(Faces.occlude(D.X[i], c, r)) === D.y[i]) ok++; }); return ok / teI.length; })]);
        stressBox.innerHTML = ''; stressBox.appendChild(h('table', { class: 't' }, h('thead', null, h('tr', null, ['model', ...conds].map(x => h('th', null, x)))), h('tbody', null, rows.map(r2 => h('tr', null, h('td', null, r2[0]), r2.slice(1).map(v => h('td', { style: `background:${Colors.rgba('#7aa2c8', v * 0.55)};text-align:center` }, (v * 100).toFixed(0) + '%')))))));
        if (!base) stressBox.appendChild(h('div', { class: 'muted', style: 'font-size:12px;margin-top:6px' }, 'Tip: run the PCA + RF baseline first to add its row.'));
      }
      const stEp = Ui.stat('epoch'), stL = Ui.stat('train loss'), stA = Ui.stat('test acc'), stBox = h('div'), baseOut = h('div', { class: 'muted', style: 'font-size:13px;margin-top:6px' }), stressBox = h('div');
      const runB = Ui.btn('Train', () => { P.run = !P.run; runB.textContent = P.run ? 'Pause' : 'Train'; }, 'pri'), baseBtn = Ui.btn('PCA + Random Forest baseline', baseline);
      const sel = Ui.slider({ label: 'test face', min: 0, max: teI.length - 1, step: 1, value: 0, on: v => { selTest = v; occ = null; } });
      build();
      root.appendChild(Ui.grid('g2', h('div', null, Ui.card('Training', Ui.row(runB, Ui.btn('Reset', () => { build(); }), Ui.chk('data augmentation (shift ±1px, brightness)', false, v => { P.aug = v; })), Ui.row(Ui.slider({ label: 'conv-1 filters', min: 4, max: 12, step: 4, value: P.F1, on: v => { P.F1 = v; build(); } }), Ui.slider({ label: 'conv-2 filters', min: 4, max: 16, step: 4, value: P.F2, on: v => { P.F2 = v; build(); } }), Ui.slider({ label: 'learning rate', min: -3.5, max: -1.5, step: 0.1, value: Math.log10(P.lr), fmt: v => Math.pow(10, v).toPrecision(2), on: v => { P.lr = Math.pow(10, v); } })), Ui.row(stEp, stL, stA)), Ui.card(null, cvL.c), Ui.card('Learned filters', cvF.c), Ui.card('Feature maps for the selected test face', sel, cvM.c)),
        h('div', null, Ui.card('Prediction', cvP.c), Ui.card('Where does it look?', Ui.row(Ui.btn('Occlusion map', occlusion, 'pri')), cvO.c), Ui.card('vs T9 and robustness', Ui.row(baseBtn, Ui.btn('Run stress test', stress)), baseOut, h('div', { style: 'margin-top:8px' }, stressBox)))));
      let f = 0;
      ctx.raf(() => {
        if (P.run) { const t0 = performance.now(); while (performance.now() - t0 < P.budget) trainBatch(); }
        if (f++ % 3 === 0) { drawFilters(); drawMaps(); drawPred(); drawOcc(); lch.draw([{ name: 'loss', color: '#d7b56d', pts: hist.loss.map(p => [p[0], p[1]]) }, { name: 'test acc /100', color: '#86b894', pts: hist.acc.map(p => [p[0], p[1] / 100]) }], { yMin: 0 }); stEp.set(epoch); stL.set(hist.loss.length ? hist.loss[hist.loss.length - 1][1].toFixed(3) : '–'); stA.set(hist.acc.length ? hist.acc[hist.acc.length - 1][1].toFixed(0) + '%' : '–'); }
      });
    },
  });

  /* ============================================================ B3 ====== */
  Lab.register({
    id: 'b3', unit: 5, num: 'B3', bonus: true, title: 'Perceptron, McCulloch–Pitts & XOR',
    tag: 'Syllabus extras: threshold logic units, the perceptron learning rule — and why one neuron cannot do XOR but two layers can.',
    star: 'Click to place points · watch each perceptron update move the line · fix XOR with a hidden layer',
    idea: `<p>A <b>McCulloch–Pitts unit</b> fires (outputs 1) when <code>w·x ≥ θ</code>. The <b>perceptron</b> learns <code>w</code> and a bias from labelled points with the rule <code>w ← w + η(y − ŷ)x</code>, converging whenever the classes are linearly separable. Click the plane to place class-A (orange) and class-B (blue) points, press <b>Step</b> to apply one correction at a time, then load <b>XOR</b>: the perceptron cycles forever. Press <b>Add hidden layer</b> and a small MLP trained with back-prop (T14) solves it.</p>`,
    tryIt: ['Load <b>AND</b>, <b>OR</b>, then <b>XOR</b> from the presets; compare the number of epochs to converge.', 'On XOR watch the “mistakes per epoch” never reach 0, then switch to the MLP.', 'Place your own separable clusters and see the final line; add one outlier on the wrong side to see the perceptron fail to converge.'],
    hood: `<pre>ŷ = 1 if w·x + b ≥ 0 else 0
for each sample (x, y):  if ŷ ≠ y:   w ← w + η (y − ŷ) x ;   b ← b + η (y − ŷ)
Perceptron convergence theorem: if the data are linearly separable the loop makes finitely many mistakes.</pre>`,
    quiz: [{ q: 'Why can a single perceptron not learn XOR?', o: ['too little data', 'XOR is not linearly separable', 'wrong learning rate', 'needs ReLU'], a: 1, why: 'No single line separates {(0,0),(1,1)} from {(0,1),(1,0)}.' }, { q: 'What fixes it?', o: ['a hidden layer with a non-linearity', 'a smaller learning rate', 'more epochs', 'a bias'], a: 0, why: 'Two hidden units carve two half-planes whose combination is XOR.' }],
    mount(root, ctx) {
      let pts = [], cls = 1, w = [0.1, -0.1], b = 0, ep = 0, mis = [], idx = 0, lr = 0.2, mlp = null, useMLP = false, mlpStep = 0;
      const cv = Ui.canvas(520, 520), dom = 1.3; const toP = (x, y) => [(x + dom) / (2 * dom) * cv.w, (dom - y) / (2 * dom) * cv.h], toX = (px, py) => [px / cv.w * 2 * dom - dom, dom - py / cv.h * 2 * dom];
      const preset = k => { const P = { and: [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]], or: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], xor: [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]] }[k]; pts = []; P.forEach(([x, y, c]) => { for (let i = 0; i < 4; i++) pts.push([x - 0.5 + (Math.random() - 0.5) * 0.18, y - 0.5 + (Math.random() - 0.5) * 0.18, c]); }); resetP(); };
      const resetP = () => { w = [0.1, -0.1]; b = 0; ep = 0; mis = []; idx = 0; useMLP = false; mlp = null; };
      const pred = (x, y) => (w[0] * x + w[1] * y + b >= 0 ? 1 : 0);
      function stepP() { if (!pts.length) return; let tries = 0; while (tries++ <= pts.length) { const [x, y, c] = pts[idx % pts.length]; idx++; if (idx % pts.length === 0) { ep++; mis.push([ep, mis.cur || 0]); mis.cur = 0; } const e = c - pred(x, y); if (e !== 0) { w[0] += lr * e * x; w[1] += lr * e * y; b += lr * e; mis.cur = (mis.cur || 0) + 1; return; } } }
      function trainMLP() { if (!mlp) { mlp = { net: new Net([2, 4, 2], { act: 'tanh', rng: new RNG(3) }) }; mlp.opt = new Optimizer('adam', mlp.net.params, { lr: 0.05 }); } const n = pts.length, X = new Float64Array(n * 2), y = pts.map(p => p[2]); pts.forEach((p, i) => { X[i * 2] = p[0]; X[i * 2 + 1] = p[1]; }); for (let k = 0; k < 20; k++) { mlp.net.zeroGrad(); const o = mlp.net.forward(X, n, true), r = softmaxCE(o, n, 2, y); mlp.net.backward(r.d); mlp.opt.step(); mlp.acc = r.acc; mlp.loss = r.loss; } mlpStep += 20; }
      function draw() {
        const { g, w: W, h: H } = cv; g.clearRect(0, 0, W, H);
        if (useMLP && mlp) drawBoundary(g, mlp.net, 2, W, H, null, 40, 0.5, false, dom); else { for (let j = 0; j < 26; j++) for (let i = 0; i < 26; i++) { const [x, y] = toX((i + 0.5) * W / 26, (j + 0.5) * H / 26); g.fillStyle = Colors.rgba(pred(x, y) ? '#d9825b' : '#7aa2c8', 0.16); g.fillRect(i * W / 26, j * H / 26, W / 26 + 0.5, H / 26 + 0.5); } if (Math.abs(w[0]) + Math.abs(w[1]) > 1e-9) { const pts2 = []; for (const x of [-dom, dom]) { const y = -(w[0] * x + b) / (w[1] || 1e-9); pts2.push(toP(x, y)); } g.strokeStyle = '#d7b56d'; g.lineWidth = 3; g.beginPath(); g.moveTo(...pts2[0]); g.lineTo(...pts2[1]); g.stroke(); } }
        g.strokeStyle = '#2a2620'; g.beginPath(); g.moveTo(W / 2, 0); g.lineTo(W / 2, H); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
        pts.forEach(([x, y, c]) => { const [px, py] = toP(x, y); const wrong = !useMLP && pred(x, y) !== c; g.beginPath(); g.arc(px, py, 7, 0, 7); g.fillStyle = c ? '#d9825b' : '#7aa2c8'; g.fill(); g.lineWidth = wrong ? 3 : 1; g.strokeStyle = wrong ? '#fff' : '#100e0c'; g.stroke(); });
        g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.fillText(useMLP ? 'MLP 2-4-2 (tanh) decision regions' : `perceptron: ${w[0].toFixed(2)}·x + ${w[1].toFixed(2)}·y + ${b.toFixed(2)} ≥ 0   (white ring = currently misclassified)`, 8, H - 8);
      }
      cv.c.addEventListener('pointerdown', e => { const [px, py] = cv.xy(e), [x, y] = toX(px, py); pts.push([x, y, cls]); });
      const segC = Ui.seg([{ v: 1, label: '● class A (orange)' }, { v: 0, label: '● class B (blue)' }], 1, v => { cls = v; });
      const cvE = Ui.canvas(420, 160), ech = new LineChart(cvE, { xlabel: 'epoch', title: 'mistakes per epoch', yMin: 0 }); const stat = Ui.stat('epochs', 0), stM = Ui.stat('mistakes (now)', 0); let run = false;
      const runB = Ui.btn('Run', () => { run = !run; runB.textContent = run ? 'Pause' : 'Run'; }, 'pri');
      root.appendChild(Ui.grid('g2', Ui.card('Click to add points', segC, cv.c), h('div', null, Ui.card('Perceptron', Ui.row(Ui.btn('Step (one update)', stepP, 'pri'), runB, Ui.btn('Reset weights', resetP), Ui.btn('Clear points', () => { pts = []; resetP(); })), Ui.row(Ui.btn('AND', () => preset('and')), Ui.btn('OR', () => preset('or')), Ui.btn('XOR', () => preset('xor'))), Ui.slider({ label: 'learning rate η', min: 0.05, max: 1, step: 0.05, value: lr, on: v => { lr = v; } }), Ui.row(stat, stM)), Ui.card('Fix XOR with a hidden layer', Ui.row(Ui.btn('Add hidden layer & train (back-prop)', () => { useMLP = true; trainMLP(); }, 'pri')), h('div', { class: 'muted', id: 'mlpinfo', style: 'font-size:13px;margin-top:6px' }, '')), Ui.card(null, cvE.c))));
      preset('and'); const info = root.querySelector('#mlpinfo'); let f = 0;
      ctx.raf(() => { if (run) { for (let i = 0; i < 2; i++) stepP(); } if (useMLP && mlp && f % 2 === 0) trainMLP(); draw(); if (f++ % 6 === 0) { ech.draw([{ name: 'mistakes', color: '#d9825b', pts: mis.slice(-80) }], { yMin: 0 }); stat.set(ep); stM.set(pts.filter(p => pred(p[0], p[1]) !== p[2]).length); if (useMLP && mlp) info.textContent = `MLP step ${mlpStep} · loss ${mlp.loss.toFixed(3)} · accuracy ${(mlp.acc * 100).toFixed(0)}%`; } });
    },
  });
})();
