/* Unit 3 — Ensemble Learning: T7 boosting, T8 bagging & bias–variance, T9 ensemble face recognition (+ shared procedural Faces) */
(function () {
  'use strict';
  const { RNG, ClsTree, RegTree } = ML;

  /* =================================================== procedural faces ===== */
  const Faces = (function () {
    const S = 24, BIG = 96, K = BIG / S;
    let cvs = null, g = null;
    function ctx2d() { if (!cvs) { cvs = document.createElement('canvas'); cvs.width = cvs.height = BIG; g = cvs.getContext('2d', { willReadFrequently: true }); } return g; }
    const gray = v => { v = Math.max(0, Math.min(1, v)) * 255 | 0; return `rgb(${v},${v},${v})`; };
    function identity(rng) {
      return { fw: rng.uniform(0.86, 1.14), fh: rng.uniform(0.9, 1.1), skin: rng.uniform(0.5, 0.92), hair: rng.uniform(0.05, 0.42), style: rng.int(3), eyeGap: rng.uniform(3.3, 4.9), eyeY: rng.uniform(-4.8, -2.8), eyeR: rng.uniform(0.95, 1.6), brow: rng.uniform(-0.35, 0.35), browY: rng.uniform(-2.5, -1.7), browW: rng.uniform(2.4, 4), noseL: rng.uniform(0.8, 3.4), mouthY: rng.uniform(5.2, 7.6), mouthW: rng.uniform(4.5, 8), glasses: rng.rand() < 0.3, beard: rng.rand() < 0.25, hl: rng.uniform(0.5, 0.75) };
    }
    function vary(rng, strength = 1) { return { dx: rng.uniform(-1.3, 1.3) * strength, dy: rng.uniform(-1.3, 1.3) * strength, scale: 1 + rng.uniform(-0.06, 0.06) * strength, light: rng.uniform(-0.55, 0.55) * strength, smile: rng.uniform(-0.4, 1) * strength, eye: 1 + rng.uniform(-0.35, 0.25) * strength, noise: 0.03 * strength }; }
    function render(p, v, rng) {
      const g = ctx2d(), U = K; g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, BIG, BIG);
      const bg = 0.16 + 0.06 * (v.light || 0); g.fillStyle = gray(bg); g.fillRect(0, 0, BIG, BIG);
      const cx = BIG / 2 + (v.dx || 0) * U, cy = BIG / 2 + 0.5 * U + (v.dy || 0) * U, sc = v.scale || 1, rx = 8.6 * U * p.fw * sc, ry = 10 * U * p.fh * sc;
      const el = (x, y, a, b, col) => { g.beginPath(); g.ellipse(x, y, a, b, 0, 0, 7); g.fillStyle = col; g.fill(); };
      // hair (behind / top)
      if (p.style !== 2) { el(cx, cy - 1.5 * U * sc, rx * 1.12, ry * 1.1, gray(p.hair)); if (p.style === 1) { g.fillStyle = gray(p.hair); g.fillRect(cx - rx * 1.12, cy - 2 * U, rx * 0.34, ry * 1.4); g.fillRect(cx + rx * 0.78, cy - 2 * U, rx * 0.34, ry * 1.4); } }
      // face with lighting gradient
      const grd = g.createLinearGradient(cx - rx, 0, cx + rx, 0), L = (v.light || 0) * 0.28; grd.addColorStop(0, gray(p.skin - L)); grd.addColorStop(1, gray(p.skin + L)); g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, 7); g.fillStyle = grd; g.fill();
      if (p.style !== 2) { g.save(); g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, 7); g.clip(); g.fillStyle = gray(p.hair); g.fillRect(0, 0, BIG, cy - ry * p.hl); g.restore(); } else { g.save(); g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, 7); g.clip(); g.fillStyle = gray(Math.min(1, p.skin + 0.05)); g.fillRect(0, 0, BIG, cy - ry * 0.75); g.restore(); }
      if (p.beard) { g.save(); g.beginPath(); g.ellipse(cx, cy, rx, ry, 0, 0, 7); g.clip(); g.fillStyle = gray(p.hair + 0.04); g.fillRect(0, cy + ry * 0.35, BIG, BIG); g.restore(); }
      // eyes & brows
      const ey = cy + p.eyeY * U * sc, eg = p.eyeGap * U * sc;
      for (const s of [-1, 1]) {
        el(cx + s * eg, ey, p.eyeR * U * 1.25 * sc, p.eyeR * U * 0.8 * (v.eye || 1) * sc, gray(0.93)); el(cx + s * eg, ey, p.eyeR * U * 0.6 * sc, p.eyeR * U * 0.6 * Math.min(1, v.eye || 1) * sc, gray(0.06));
        g.strokeStyle = gray(p.hair * 0.8); g.lineWidth = 1.6 * U * 0.8; g.lineCap = 'round'; g.beginPath(); g.moveTo(cx + s * (eg - p.browW * U / 2), ey + p.browY * U * sc + s * p.brow * U * 1.5); g.lineTo(cx + s * (eg + p.browW * U / 2), ey + p.browY * U * sc - s * p.brow * U * 1.5); g.stroke();
      }
      // nose
      g.strokeStyle = gray(p.skin - 0.22); g.lineWidth = 1.5 * U * 0.8; g.beginPath(); g.moveTo(cx, ey + 1 * U); g.lineTo(cx - 0.5 * U, cy + p.noseL * U * sc); g.lineTo(cx + 1.4 * U, cy + p.noseL * U * sc + 0.3 * U); g.stroke();
      // mouth
      const my = cy + p.mouthY * U * sc, mw = p.mouthW * U * sc * 0.5, sm = (0.4 + (v.smile ?? 0.3)) * 1.6 * U; g.strokeStyle = gray(Math.max(0.05, p.skin - 0.42)); g.lineWidth = 1.9 * U * 0.8; g.beginPath(); g.moveTo(cx - mw, my); g.quadraticCurveTo(cx, my + sm * 1.4, cx + mw, my); g.stroke();
      if (p.glasses) { g.strokeStyle = gray(0.08); g.lineWidth = 1.3 * U * 0.8; for (const s of [-1, 1]) { g.beginPath(); g.ellipse(cx + s * eg, ey, p.eyeR * U * 1.9 * sc, p.eyeR * U * 1.5 * sc, 0, 0, 7); g.stroke(); } g.beginPath(); g.moveTo(cx - eg + p.eyeR * U * 1.9, ey); g.lineTo(cx + eg - p.eyeR * U * 1.9, ey); g.stroke(); }
      // downsample 96 -> 24 (box filter)
      const d = g.getImageData(0, 0, BIG, BIG).data, out = new Float64Array(S * S);
      for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) { let s = 0; for (let b = 0; b < K; b++) for (let a = 0; a < K; a++) s += d[((j * K + b) * BIG + i * K + a) * 4]; out[j * S + i] = s / (K * K * 255); }
      if (v.noise && rng) for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(1, out[i] + rng.normal() * v.noise));
      return out;
    }
    function make(nId = 12, per = 10, seed = 4, strength = 0.75) {
      const rng = new RNG(seed), ids = Array.from({ length: nId }, () => identity(rng)), X = [], y = [], k = [];
      ids.forEach((p, id) => { for (let j = 0; j < per; j++) { X.push(render(p, vary(rng, strength), rng)); y.push(id); k.push(j); } });
      return { X, y, k, ids, S };
    }
    function occlude(img, kind, rng) {
      const o = Float64Array.from(img);
      if (kind === 'sunglasses') { for (let j = 8; j < 12; j++) for (let i = 3; i < 21; i++) o[j * S + i] = 0.04; }
      else if (kind === 'mask') { for (let j = 15; j < 24; j++) for (let i = 4; i < 20; i++) o[j * S + i] = 0.8; }
      else if (kind === 'noise') { for (let i = 0; i < o.length; i++) o[i] = Math.max(0, Math.min(1, o[i] + (rng || new RNG(3)).normal() * 0.18)); }
      else if (kind === 'dark') { for (let i = 0; i < o.length; i++) o[i] *= 0.45; }
      return o;
    }
    return { S, identity, vary, render, make, occlude };
  })();
  window.Faces = Faces;

  /* ======================================================== shared bits ===== */
  function dataset(kind, n, noise, rng) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) {
      let x, yy, c = i % 2;
      if (kind === 'moons') { const t = rng.rand() * Math.PI; if (c === 0) { x = Math.cos(t) * 0.8 - 0.4; yy = Math.sin(t) * 0.8 - 0.2; } else { x = 0.4 - Math.cos(t) * 0.8; yy = 0.2 - Math.sin(t) * 0.8; } x += rng.normal() * noise * 0.4; yy += rng.normal() * noise * 0.4; }
      else if (kind === 'circles') { const t = rng.rand() * 6.283, r = (c === 0 ? 0.42 : 0.85) + rng.normal() * noise * 0.22; x = Math.cos(t) * r; yy = Math.sin(t) * r; }
      else if (kind === 'xor') { x = rng.uniform(-1, 1); yy = rng.uniform(-1, 1); c = x * yy > 0 ? 1 : 0; if (rng.rand() < noise * 0.5) c = 1 - c; }
      else { const m = c === 0 ? -0.32 : 0.32; x = m + rng.normal() * (0.2 + noise * 0.4); yy = m + rng.normal() * (0.2 + noise * 0.4); }
      X.push(Float64Array.from([x, yy])); y.push(c === 1 ? 1 : -1);
    }
    return { X, y };
  }
  const sig = z => 1 / (1 + Math.exp(-z));
  const lerpCol = (t) => Colors.diverge(t);

  /* ============================================================ T7 ====== */
  Lab.register({
    id: 't7', unit: 3, num: 'T7', title: 'AdaBoost & Gradient Boosting',
    tag: 'Watch sample weights shift toward the hard points round by round, then see gradient boosting fit residuals one tree at a time.',
    star: 'Point size = live sample weight · L1 vs L2 loss with outliers · boosting-head-to-head',
    idea: `<p><b>Boosting</b> builds a strong learner from many weak ones <i>sequentially</i>. <b>AdaBoost</b> re-weights the training points: those the last stump got wrong grow heavier, so the next learner is forced to focus on them, and each learner’s vote is weighted by <code>α = ½ ln((1−ε)/ε)</code>. <b>Gradient boosting</b> instead fits each new tree to the <i>negative gradient</i> of the loss (the residuals) and adds it with a small learning rate.</p>
    <p>The notebook only reports accuracy and AUC on a breast-cancer table. Here you <b>step through the rounds</b> on 2-D data: point radius is the live weight, the decision region updates per round, and the gradient-boosting tab lets you see the residuals each tree is fitting — and what happens when the loss is robust (L1) and the data has outliers.</p>`,
    tryIt: ['AdaBoost tab: start with <b>circles</b> and press <b>Step</b> repeatedly. Note the big dots after each step — they are the points the previous stump misclassified, so the next stump is forced to deal with them.',
      'Now pick <b>XOR</b> with stumps (depth 1): the error never leaves 50 %. Boosted stumps form an <i>additive</i> model f(x)+g(y), and XOR needs an interaction. Set <b>learner depth = 2</b> and test accuracy climbs to about 85 % within 40 rounds — boosting adds up weak learners, it cannot add capacity they lack.',
      'Raise <b>label noise</b> and train for 100+ rounds: AdaBoost’s exponential loss chases the noisy points (train error → 0, test error rises). Lower the <b>learning rate</b> to tame it.',
      'Press <b>Compare with gradient boosting</b> to see test accuracy of stump / AdaBoost / GB at the same number of rounds.',
      'Gradient boosting tab: step and watch the lower plot — the tree (teal) fits the residuals (dots). Add <b>outliers</b>: L2 gets dragged toward them, L1 stays robust.'],
    hood: `<pre>AdaBoost:   w_i = 1/n
  for t = 1..T:  h_t = fit_stump(X, y, w);   ε = Σ w_i [h_t(x_i) ≠ y_i]
                 α_t = lr · ½ ln((1−ε)/ε);   w_i ← w_i · exp(−α_t y_i h_t(x_i)); normalise
  H(x) = sign( Σ α_t h_t(x) )

Gradient boosting (squared loss):  F_0 = mean(y);   r_i = y_i − F_{m−1}(x_i)
  h_m = fit_tree(X, r);   F_m = F_{m−1} + ν · h_m          (L1 loss: fit sign(r), leaf = median(r))
Binary log-loss GB: r = y − σ(F); leaf value = Σr / Σ p(1−p)  (Newton step)</pre>`,
    quiz: [
      { q: 'After a round of AdaBoost, which points get a larger weight?', o: ['those the new stump classified correctly', 'those it misclassified', 'the points nearest the boundary only', 'all equally'], a: 1, why: 'w_i ← w_i·exp(−α y_i h(x_i)) grows when y_i h(x_i) = −1.' },
      { q: 'In gradient boosting with squared loss, each new tree is fit to…', o: ['the original labels', 'the current residuals', 'the sample weights', 'random noise'], a: 1, why: 'The negative gradient of ½(y−F)² is the residual y−F.' },
      { q: 'Why use a small learning rate ν together with many trees?', o: ['it is cheaper', 'it shrinks each tree’s contribution, acting as regularisation', 'it removes residuals', 'to avoid stumps'], a: 1, why: 'Shrinkage trades more rounds for better generalisation.' }],
    mount(root, ctx) {
      const tabs = Ui.seg([{ v: 'ada', label: 'AdaBoost (weights)' }, { v: 'gb', label: 'Gradient boosting (residuals)' }], 'ada', v => { adaBox.style.display = v === 'ada' ? '' : 'none'; gbBox.style.display = v === 'gb' ? '' : 'none'; });
      const adaBox = h('div'), gbBox = h('div', { style: 'display:none' }); root.appendChild(h('div', { style: 'margin-bottom:14px' }, tabs)); root.appendChild(adaBox); root.appendChild(gbBox);

      /* ---------- AdaBoost ---------- */
      const P = { kind: 'circles', noise: 0.15, n: 160, depth: 1, lr: 1.0, speed: 1 };
      let tr, te, w, F, Ft, learners, alphas, errs, grid, GF, lastStump, running = false, rngA = new RNG(5);
      const GN = 56, gx = i => -1.15 + 2.3 * (i + 0.5) / GN;
      const cv = Ui.canvas(560, 560), cvE = Ui.canvas(480, 200), cvA = Ui.canvas(480, 140), chart = new LineChart(cvE, { xlabel: 'boosting rounds', title: 'error %', yMin: 0 });
      function resetA() {
        rngA = new RNG(5); tr = dataset(P.kind, P.n, P.noise, rngA); te = dataset(P.kind, 300, P.noise, new RNG(77)); const n = tr.X.length;
        w = new Float64Array(n).fill(1 / n); F = new Float64Array(n); Ft = new Float64Array(te.X.length); GF = new Float64Array(GN * GN); learners = []; alphas = []; errs = { tr: [[0, 50]], te: [[0, 50]] }; lastStump = null;
        grid = []; for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) grid.push(Float64Array.from([gx(i), gx(j)]));
      }
      resetA();
      const lab01 = tr => tr.y.map(v => (v + 1) / 2);
      function stepA() {
        const y01 = lab01(tr), tree = new ClsTree({ maxDepth: P.depth, K: 2 }).fit(tr.X, y01, w), n = tr.X.length;
        const hp = x => (tree.proba(x)[1] > 0.5 ? 1 : -1); let err = 0; const pred = tr.X.map(hp); for (let i = 0; i < n; i++) if (pred[i] !== tr.y[i]) err += w[i];
        err = Math.min(Math.max(err, 1e-10), 1 - 1e-10); const alpha = P.lr * 0.5 * Math.log((1 - err) / err);
        let s = 0; for (let i = 0; i < n; i++) { w[i] *= Math.exp(-alpha * tr.y[i] * pred[i]); s += w[i]; } for (let i = 0; i < n; i++) w[i] /= s;
        for (let i = 0; i < n; i++) F[i] += alpha * pred[i]; te.X.forEach((x, i) => { Ft[i] += alpha * hp(x); }); grid.forEach((x, i) => { GF[i] += alpha * hp(x); });
        learners.push(tree); alphas.push(alpha); let e1 = 0, e2 = 0; for (let i = 0; i < n; i++) if ((F[i] >= 0 ? 1 : -1) !== tr.y[i]) e1++; for (let i = 0; i < te.X.length; i++) if ((Ft[i] >= 0 ? 1 : -1) !== te.y[i]) e2++;
        const t = learners.length; errs.tr.push([t, e1 / n * 100]); errs.te.push([t, e2 / te.X.length * 100]);
        lastStump = P.depth === 1 && tree.root.f !== undefined ? { f: tree.root.f, t: tree.root.t, pos: hp(Float64Array.from(tree.root.f === 0 ? [tree.root.t + 0.01, 0] : [0, tree.root.t + 0.01])) } : null; lastStump && (lastStump.err = err);
      }
      function drawA() {
        const { g, w: W, h: H } = cv, m = 10, sx = x => m + (x + 1.15) / 2.3 * (W - 2 * m), sy = y => H - m - (y + 1.15) / 2.3 * (H - 2 * m), cs = (W - 2 * m) / GN; g.clearRect(0, 0, W, H);
        if (learners.length) for (let j = 0; j < GN; j++) for (let i = 0; i < GN; i++) { const v = GF[j * GN + i]; g.fillStyle = Colors.rgba(v >= 0 ? '#d9825b' : '#7aa2c8', 0.1 + 0.4 * Math.tanh(Math.abs(v) * 0.6)); g.fillRect(sx(gx(i)) - cs / 2, sy(gx(j)) - cs / 2, cs + 0.6, cs + 0.6); }
        if (lastStump) { g.strokeStyle = '#d7b56d'; g.lineWidth = 2.5; g.setLineDash([7, 5]); g.beginPath(); if (lastStump.f === 0) { g.moveTo(sx(lastStump.t), m); g.lineTo(sx(lastStump.t), H - m); } else { g.moveTo(m, sy(lastStump.t)); g.lineTo(W - m, sy(lastStump.t)); } g.stroke(); g.setLineDash([]); }
        const n = tr.X.length; let mw = 0; for (let i = 0; i < n; i++) mw = Math.max(mw, w[i]);
        for (let i = 0; i < n; i++) { const x = sx(tr.X[i][0]), y = sy(tr.X[i][1]), wrong = learners.length && (F[i] >= 0 ? 1 : -1) !== tr.y[i], r = 2.5 + 11 * Math.sqrt(w[i] / (mw + 1e-12)) * Math.min(1, w[i] * n / 3 + 0.2); g.beginPath(); g.arc(x, y, r, 0, 7); g.fillStyle = tr.y[i] > 0 ? '#d9825b' : '#7aa2c8'; g.globalAlpha = 0.9; g.fill(); g.globalAlpha = 1; g.lineWidth = wrong ? 2.5 : 1; g.strokeStyle = wrong ? '#fff' : '#100e0c'; g.stroke(); }
        g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; g.fillText('dot size = sample weight · white ring = misclassified by the ensemble · dashed = latest ' + (P.depth === 1 ? 'stump' : 'tree'), 12, H - 14);
      }
      function drawAlpha() { const { g, w: W, h: H } = cvA; g.clearRect(0, 0, W, H); g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; g.fillText('learner weights α_t (last 60 rounds)', 10, 14); const a = alphas.slice(-60), mx = Math.max(0.1, ...a.map(Math.abs)), bw = (W - 20) / 60; a.forEach((v, i) => { const hh = (H - 34) * Math.abs(v) / mx; g.fillStyle = v >= 0 ? '#d7b56d' : '#cf6a76'; g.fillRect(10 + i * bw, H - 10 - hh, bw - 1.5, hh); }); }
      const stT = Ui.stat('rounds'), stE = Ui.stat('train err'), stV = Ui.stat('test err'), stM = Ui.stat('max weight ×n');
      const runBtn = Ui.btn('Run', () => { running = !running; runBtn.textContent = running ? 'Pause' : 'Run'; }, 'pri');
      const dsSeg = Ui.seg(['circles', 'moons', 'xor', 'blobs'], 'circles', v => { P.kind = v; resetA(); });
      const cmp = h('div');
      function compareGB() {
        const T = Math.max(1, learners.length), y01 = lab01(tr), n = tr.X.length, p0 = y01.reduce((a, b) => a + b, 0) / n; let Fg = new Float64Array(n).fill(Math.log(p0 / (1 - p0))), Fgt = new Float64Array(te.X.length).fill(Fg[0]);
        for (let m = 0; m < T; m++) { const p = Fg.map(sig), r = y01.map((v, i) => v - p[i]); const tree = new RegTree({ maxDepth: 3, minLeaf: 2, leafFn: idx => { let a = 0, b = 0; for (const i of idx) { a += r[i]; b += p[i] * (1 - p[i]); } return a / (b + 1e-9); } }).fit(tr.X, r); tr.X.forEach((x, i) => { Fg[i] += 0.1 * tree.predict(x); }); te.X.forEach((x, i) => { Fgt[i] += 0.1 * tree.predict(x); }); }
        const acc = (Fv, ys) => Fv.reduce((s, v, i) => s + ((v >= 0 ? 1 : -1) === ys[i] ? 1 : 0), 0) / ys.length * 100; const stump = new ClsTree({ maxDepth: 1, K: 2 }).fit(tr.X, y01); const sacc = te.X.reduce((s, x, i) => s + ((stump.proba(x)[1] > 0.5 ? 1 : -1) === te.y[i] ? 1 : 0), 0) / te.y.length * 100;
        cmp.innerHTML = ''; cmp.appendChild(Ui.table(['model', 'train acc %', 'test acc %'], [['single stump', '–', sacc.toFixed(1)], ['AdaBoost ×' + T + ' (this tab)', (100 - errs.tr[errs.tr.length - 1][1]).toFixed(1), (100 - errs.te[errs.te.length - 1][1]).toFixed(1)], ['Gradient boosting ×' + T + ' (depth 3, ν=0.1, log-loss)', acc(Fg, tr.y).toFixed(1), acc(Fgt, te.y).toFixed(1)]]));
      }
      adaBox.appendChild(Ui.grid('g2', Ui.card(null, h('div', { class: 'row', style: 'margin-bottom:8px' }, dsSeg), cv.c),
        h('div', null, Ui.card('Boosting controls', Ui.row(Ui.btn('Step', () => stepA(), 'pri'), runBtn, Ui.btn('Reset', () => { resetA(); })), Ui.row(Ui.slider({ label: 'label noise', min: 0, max: 0.6, step: 0.05, value: P.noise, on: v => { P.noise = v; resetA(); } }), Ui.slider({ label: 'learner depth (1 = stump)', min: 1, max: 4, step: 1, value: 1, on: v => { P.depth = v; resetA(); } })), Ui.row(Ui.slider({ label: 'learning rate', min: 0.1, max: 1, step: 0.05, value: 1, on: v => { P.lr = v; resetA(); } }), Ui.slider({ label: 'rounds per frame', min: 1, max: 5, step: 1, value: 1, on: v => { P.speed = v; } })), Ui.row(stT, stE, stV, stM)),
          Ui.card(null, cvE.c), Ui.card(null, cvA.c), Ui.card('Head-to-head', Ui.btn('Compare with gradient boosting', compareGB), cmp))));

      /* ---------- Gradient boosting ---------- */
      const Q = { fn: 'sine', loss: 'l2', depth: 2, nu: 0.3, sub: 1, outliers: false };
      let gx_, gy_, gF, gG, gtrees = [], hist2 = { tr: [], te: [] }, lastH = null, lastRes = null, grun = false, rg = new RNG(9);
      const f1 = x => (Q.fn === 'sine' ? Math.sin(2 * Math.PI * x) : Q.fn === 'steps' ? (x < 0.3 ? -1 : x < 0.65 ? 1 : 0) : Math.sin(5 * Math.PI * x) * (1 - x) + 0.5 * x);
      const XG = Array.from({ length: 120 }, (_, i) => Float64Array.from([i / 119]));
      function resetG() {
        rg = new RNG(9); gx_ = Array.from({ length: 60 }, () => Float64Array.from([rg.rand()])); gy_ = gx_.map(x => f1(x[0]) + rg.normal() * 0.2);
        if (Q.outliers) [3, 17, 31, 44, 52].forEach(i => { gy_[i] += 2.6 * (i % 2 ? 1 : -1) + 1; });
        const init = Q.loss === 'l2' ? gy_.reduce((a, b) => a + b, 0) / gy_.length : [...gy_].sort((a, b) => a - b)[gy_.length >> 1]; gF = new Float64Array(gy_.length).fill(init); gG = new Float64Array(XG.length).fill(init); gtrees = []; hist2 = { tr: [[0, mse()]], te: [[0, tmse()]] }; lastH = null; lastRes = gy_.map((y, i) => y - gF[i]);
      }
      const mse = () => gy_.reduce((s, y, i) => s + (y - gF[i]) ** 2, 0) / gy_.length, tmse = () => XG.reduce((s, x, i) => s + (f1(x[0]) - gG[i]) ** 2, 0) / XG.length;
      function stepG() {
        const n = gy_.length, r = gy_.map((y, i) => y - gF[i]); let idx = Array.from({ length: n }, (_, i) => i); if (Q.sub < 1) { rg.shuffle(idx); idx = idx.slice(0, Math.max(8, Math.round(n * Q.sub))); }
        const target = Q.loss === 'l2' ? r : r.map(Math.sign), med = ids => { const a = ids.map(i => r[i]).sort((p, q) => p - q); return a[a.length >> 1]; };
        const tree = new RegTree({ maxDepth: Q.depth, minLeaf: 2, leafFn: Q.loss === 'l1' ? med : null }).fit(gx_, target, idx);
        for (let i = 0; i < n; i++) gF[i] += Q.nu * tree.predict(gx_[i]); XG.forEach((x, i) => { gG[i] += Q.nu * tree.predict(x); });
        gtrees.push(tree); lastH = XG.map(x => tree.predict(x)); lastRes = r; hist2.tr.push([gtrees.length, mse()]); hist2.te.push([gtrees.length, tmse()]);
      }
      resetG();
      const cvG = Ui.canvas(560, 300), cvR = Ui.canvas(560, 230), cvL = Ui.canvas(480, 200), lchart = new LineChart(cvL, { xlabel: 'trees', title: 'squared error vs the TRUE function (test) and vs noisy data (train)', yMin: 0 });
      function drawG() {
        const { g, w: W, h: H } = cvG, m = 28, X = x => m + x * (W - 2 * m), Y = y => H - m - (y + 2.2) / 5.6 * (H - 2 * m); g.clearRect(0, 0, W, H);
        g.strokeStyle = '#2a2620'; g.beginPath(); g.moveTo(m, Y(0)); g.lineTo(W - m, Y(0)); g.stroke();
        g.strokeStyle = '#9b9384'; g.lineWidth = 1.5; g.setLineDash([4, 4]); g.beginPath(); XG.forEach((x, i) => { i ? g.lineTo(X(x[0]), Y(f1(x[0]))) : g.moveTo(X(x[0]), Y(f1(x[0]))); }); g.stroke(); g.setLineDash([]);
        g.strokeStyle = '#ff5d7a55'; g.lineWidth = 1; gx_.forEach((x, i) => { g.beginPath(); g.moveTo(X(x[0]), Y(gy_[i])); g.lineTo(X(x[0]), Y(gF[i])); g.stroke(); });
        g.strokeStyle = '#d7b56d'; g.lineWidth = 3; g.beginPath(); XG.forEach((x, i) => { i ? g.lineTo(X(x[0]), Y(gG[i])) : g.moveTo(X(x[0]), Y(gG[i])); }); g.stroke();
        gx_.forEach((x, i) => { g.beginPath(); g.arc(X(x[0]), Y(gy_[i]), 3.6, 0, 7); g.fillStyle = '#7aa2c8'; g.fill(); }); g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; g.fillText('blue = data · gold = ensemble F_m · dashed = truth · red sticks = residuals', 10, 16);
        const c = cvR, g2 = c.g; g2.clearRect(0, 0, c.w, c.h); const Y2 = y => c.h - 24 - (y + 3.4) / 6.8 * (c.h - 44); g2.strokeStyle = '#2a2620'; g2.beginPath(); g2.moveTo(m, Y2(0)); g2.lineTo(c.w - m, Y2(0)); g2.stroke();
        if (lastRes) { gx_.forEach((x, i) => { g2.beginPath(); g2.arc(X(x[0]), Y2(lastRes[i]), 3.2, 0, 7); g2.fillStyle = '#d9825b'; g2.fill(); }); }
        if (lastH) { g2.strokeStyle = '#6bb3aa'; g2.lineWidth = 3; g2.beginPath(); XG.forEach((x, i) => { const v = lastH[i] * (Q.loss === 'l2' ? 1 : 1); i ? g2.lineTo(X(x[0]), Y2(v)) : g2.moveTo(X(x[0]), Y2(v)); }); g2.stroke(); }
        g2.fillStyle = '#9b9384'; g2.font = '11.5px "Instrument Sans"'; g2.fillText(gtrees.length ? `orange = residuals before tree ${gtrees.length} · teal = tree ${gtrees.length}${Q.loss === 'l1' ? ' (fit to sign of residuals, median leaves)' : ''}` : 'press Step: the first tree will be fit to the residuals y − F₀', 10, 16);
      }
      const gRun = Ui.btn('Run', () => { grun = !grun; gRun.textContent = grun ? 'Pause' : 'Run'; }, 'pri');
      const fnSeg = Ui.seg([{ v: 'sine', label: 'sine' }, { v: 'steps', label: 'steps' }, { v: 'wig', label: 'wiggle' }], 'sine', v => { Q.fn = v; resetG(); });
      const lossSeg = Ui.seg([{ v: 'l2', label: 'L2 (squared)' }, { v: 'l1', label: 'L1 (absolute)' }], 'l2', v => { Q.loss = v; resetG(); });
      gbBox.appendChild(Ui.grid('g2', h('div', null, Ui.card('Ensemble F_m', cvG.c), Ui.card('What tree m learns — the residuals', cvR.c)),
        h('div', null, Ui.card('Controls', Ui.row(Ui.btn('Step', stepG, 'pri'), gRun, Ui.btn('Reset', resetG)), h('div', { class: 'row', style: 'margin:8px 0' }, fnSeg, lossSeg, Ui.chk('add outliers', false, v => { Q.outliers = v; resetG(); })),
          Ui.row(Ui.slider({ label: 'tree depth', min: 1, max: 5, step: 1, value: 2, on: v => { Q.depth = v; } }), Ui.slider({ label: 'learning rate ν', min: 0.05, max: 1, step: 0.05, value: 0.3, on: v => { Q.nu = v; } })), Ui.row(Ui.slider({ label: 'subsample (stochastic GB)', min: 0.3, max: 1, step: 0.1, value: 1, on: v => { Q.sub = v; } }))), Ui.card(null, cvL.c))));
      let f = 0;
      ctx.raf(() => {
        if (adaBox.style.display !== 'none') {
          if (running) for (let i = 0; i < P.speed && learners.length < 400; i++) stepA(); drawA(); if (f++ % 3 === 0) { chart.draw([{ name: 'train', color: '#d7b56d', pts: errs.tr }, { name: 'test', color: '#7aa2c8', pts: errs.te }], { yMin: 0 }); drawAlpha(); stT.set(learners.length); stE.set(errs.tr[errs.tr.length - 1][1].toFixed(1) + '%'); stV.set(errs.te[errs.te.length - 1][1].toFixed(1) + '%'); let mw = 0; w.forEach(v => { mw = Math.max(mw, v); }); stM.set((mw * w.length).toFixed(1)); }
        } else { if (grun && gtrees.length < 300 && f % 4 === 0) stepG(); drawG(); if (f++ % 3 === 0) lchart.draw([{ name: 'train (noisy)', color: '#d7b56d', pts: hist2.tr }, { name: 'test (truth)', color: '#7aa2c8', pts: hist2.te }], { yMin: 0 }); }
      });
    },
  });

  /* ============================================================ T8 ====== */
  Lab.register({
    id: 't8', unit: 3, num: 'T8', title: 'Bagging & the bias–variance trade-off',
    tag: 'Re-draw the training set dozens of times and watch a deep tree thrash while its bagged version stays calm — with bias² and variance measured, not asserted.',
    star: 'Live bias²/variance decomposition over 40 fresh datasets · bootstrap lab · depth & ensemble-size sweeps',
    idea: `<p>Expected test error decomposes as <code>E[(y − ĥ)²] = bias² + variance + σ²</code>. A fully grown tree has low bias but high variance: re-draw the data and the fit changes wildly. <b>Bagging</b> trains many trees on <b>bootstrap resamples</b> and averages them — the average of high-variance, low-bias predictors keeps the bias and slashes the variance.</p>
    <p>The notebook uses mlxtend’s <code>bias_variance_decomp</code> on classification with 0-1 loss. Here the decomposition is for regression (squared loss), computed live by <b>actually fitting the model on 40 independent training sets</b> and measuring how far the average prediction is from the truth (bias²) and how much predictions scatter around their own mean (variance).</p>`,
    tryIt: ['Select <b>single tree</b> and raise depth: curves fan out (variance ↑) while the mean hugs the truth (bias² ↓).',
      'Switch to <b>bagged</b> with 30+ trees at the same depth: the fan collapses; bias² stays similar; variance falls by roughly 1/B-ish.',
      'Raise <b>noise</b>: σ² (the irreducible floor) grows and overfitting hurts more.',
      'Press <b>Sweep depth</b>: the U-shaped total-error curve of a single tree versus the flatter bagged one.',
      'The <b>bootstrap lab</b> shows each bootstrap resample keeps ≈63.2 % of the distinct points (the rest are out-of-bag).'],
    hood: `<pre>for r in 1..R (=40):   draw training set D_r;  fit model;  predict on a grid  → ĥ_r(x)
mean(x)  = (1/R) Σ_r ĥ_r(x)
bias²    = mean_x ( mean(x) − f(x) )²          variance = mean_x (1/R) Σ_r ( ĥ_r(x) − mean(x) )²
expected error ≈ bias² + variance + σ²
bagging: ĥ(x) = (1/B) Σ_b tree_b(x),  tree_b trained on a bootstrap sample (n draws with replacement)</pre>`,
    quiz: [
      { q: 'Bagging mainly reduces…', o: ['bias', 'variance', 'noise σ²', 'the number of features'], a: 1, why: 'Averaging many de-correlated high-variance predictors reduces variance while leaving bias about the same.' },
      { q: 'A bootstrap sample of size n contains about what fraction of the distinct original points?', o: ['50%', '63.2%', '90%', '100%'], a: 1, why: '1 − (1−1/n)ⁿ → 1 − 1/e ≈ 0.632.' },
      { q: 'A depth-1 tree (stump) on a sine curve mostly suffers from…', o: ['high variance', 'high bias (underfitting)', 'noise', 'nothing'], a: 1, why: 'It cannot represent the sine, so its average prediction is far from the truth; the curves barely change between datasets.' }],
    mount(root, ctx) {
      const P = { model: 'single', depth: 6, B: 25, noise: 0.3, n: 40 }; const GX = Array.from({ length: 100 }, (_, i) => Float64Array.from([i / 99])); const f = x => Math.sin(2 * Math.PI * x);
      const sample = (rng, n, noise) => { const X = [], y = []; for (let i = 0; i < n; i++) { const x = rng.rand(); X.push(Float64Array.from([x])); y.push(f(x) + rng.normal() * noise); } return { X, y }; };
      const bagFit = (d, B, rng, depth) => { const n = d.X.length, trees = []; for (let b = 0; b < B; b++) { const idx = Array.from({ length: n }, () => rng.int(n)); trees.push(new RegTree({ maxDepth: depth, minLeaf: 1 }).fit(d.X, d.y, idx)); } return trees; };
      const predTrees = (trees, x) => { let s = 0; for (const t of trees) s += t.predict(x); return s / trees.length; };
      function experiment(model, depth, B, noise, n, R = 40) {
        const rng = new RNG(1234), curves = []; for (let r = 0; r < R; r++) { const d = sample(rng, n, noise); const trees = model === 'single' ? [new RegTree({ maxDepth: depth, minLeaf: 1 }).fit(d.X, d.y)] : bagFit(d, B, rng, depth); curves.push(Float64Array.from(GX, x => predTrees(trees, x))); }
        const mean = Float64Array.from(GX, (_, i) => curves.reduce((s, c) => s + c[i], 0) / R); let bias2 = 0, vr = 0; GX.forEach((x, i) => { bias2 += (mean[i] - f(x[0])) ** 2; vr += curves.reduce((s, c) => s + (c[i] - mean[i]) ** 2, 0) / R; }); bias2 /= GX.length; vr /= GX.length; return { curves, mean, bias2, vr, total: bias2 + vr + noise * noise };
      }
      const cv = Ui.canvas(560, 320), cvB = Ui.canvas(560, 280), cvS = Ui.canvas(480, 230), sweepChart = new LineChart(cvS, { xlabel: 'tree depth', title: 'error decomposition vs depth', yMin: 0 }), cvBar = Ui.canvas(480, 150);
      let res = null, boot = null;
      function plotCurves(c, curves, mean, title, col, data) {
        const { g, w: W, h: H } = c, m = 26, X = x => m + x * (W - 2 * m), Y = y => H - m - (y + 2.2) / 4.4 * (H - 2 * m); g.clearRect(0, 0, W, H); g.strokeStyle = '#2a2620'; g.beginPath(); g.moveTo(m, Y(0)); g.lineTo(W - m, Y(0)); g.stroke();
        curves.forEach(cu => { g.strokeStyle = Colors.rgba(col, 0.18); g.lineWidth = 1.2; g.beginPath(); GX.forEach((x, i) => { i ? g.lineTo(X(x[0]), Y(cu[i])) : g.moveTo(X(x[0]), Y(cu[i])); }); g.stroke(); });
        g.strokeStyle = '#d7b56d'; g.lineWidth = 3.2; g.beginPath(); GX.forEach((x, i) => { i ? g.lineTo(X(x[0]), Y(mean[i])) : g.moveTo(X(x[0]), Y(mean[i])); }); g.stroke(); g.strokeStyle = '#ece6d8'; g.lineWidth = 1.8; g.setLineDash([5, 4]); g.beginPath(); GX.forEach((x, i) => { i ? g.lineTo(X(x[0]), Y(f(x[0]))) : g.moveTo(X(x[0]), Y(f(x[0]))); }); g.stroke(); g.setLineDash([]);
        if (data) data.X.forEach((x, i) => { g.beginPath(); g.arc(X(x[0]), Y(data.y[i]), 3, 0, 7); g.fillStyle = data.in ? (data.in[i] ? '#7aa2c8' : '#6f6859') : '#7aa2c8'; g.fill(); }); g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; g.fillText(title, 10, 16);
      }
      function drawBars() {
        const { g, w: W, h: H } = cvBar; g.clearRect(0, 0, W, H); const mx = Math.max(0.2, res.total), items = [['bias²', res.bias2, '#7aa2c8'], ['variance', res.vr, '#d9825b'], ['noise σ²', P.noise ** 2, '#6f6859'], ['total', res.total, '#d7b56d']];
        items.forEach(([n, v, c], i) => { const y = 14 + i * 33; g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.textAlign = 'left'; g.fillText(n, 8, y + 14); g.fillStyle = c; g.fillRect(80, y, (W - 150) * v / mx, 20); g.fillStyle = '#ece6d8'; g.font = '12px "JetBrains Mono",monospace'; g.fillText(v.toFixed(3), 86 + (W - 150) * v / mx, y + 15); });
      }
      function update() {
        res = experiment(P.model, P.depth, P.B, P.noise, P.n); plotCurves(cv, res.curves, res.mean, `${P.model === 'single' ? 'single tree' : 'bagged ×' + P.B} · depth ${P.depth} · 40 fits on 40 different training sets (thin) · mean (gold) · truth (dashed)`, P.model === 'single' ? '#d9825b' : '#86b894'); drawBars();
        const rng = new RNG(31), d = sample(rng, P.n, P.noise), nB = 20, trees = []; const inSet = new Array(d.X.length).fill(false); let uniq = new Set();
        for (let b = 0; b < nB; b++) { const idx = Array.from({ length: d.X.length }, () => rng.int(d.X.length)); if (b === 0) idx.forEach(i => uniq.add(i)); trees.push(new RegTree({ maxDepth: P.depth, minLeaf: 1 }).fit(d.X, d.y, idx)); } uniq.forEach(i => { inSet[i] = true; }); d.in = inSet;
        plotCurves(cvB, trees.map(t => Float64Array.from(GX, x => t.predict(x))), Float64Array.from(GX, x => predTrees(trees, x)), `bootstrap lab: 20 trees on bootstrap resamples (thin) → their average (gold). Resample #1 keeps ${(uniq.size / d.X.length * 100).toFixed(0)}% distinct points (grey = out-of-bag)`, '#86b894', d);
      }
      let tm = 0; const sched = () => { clearTimeout(tm); tm = setTimeout(update, 120); }; ctx.cleanup(() => clearTimeout(tm));
      const seg = Ui.seg([{ v: 'single', label: 'Single tree' }, { v: 'bag', label: 'Bagged trees' }], 'single', v => { P.model = v; sched(); });
      const sweepBtn = Ui.btn('Sweep depth 1…12 (single vs bagged)', () => { sweepBtn.textContent = 'computing…'; setTimeout(() => { const A = [], Bv = [], Cb = [], Dv = [], T1 = [], T2 = []; for (let d = 1; d <= 12; d++) { const a = experiment('single', d, 1, P.noise, P.n, 30), b = experiment('bag', d, 15, P.noise, P.n, 30); A.push([d, a.bias2]); Bv.push([d, a.vr]); T1.push([d, a.total]); Cb.push([d, b.bias2]); Dv.push([d, b.vr]); T2.push([d, b.total]); } sweepChart.draw([{ name: 'single: bias²', color: '#7aa2c8', pts: A, width: 1.8 }, { name: 'single: variance', color: '#d9825b', pts: Bv, width: 1.8 }, { name: 'single: total', color: '#d7b56d', pts: T1, width: 3 }, { name: 'bagged: variance', color: '#d9825b', pts: Dv, dash: [5, 4], width: 1.8 }, { name: 'bagged: total', color: '#86b894', pts: T2, width: 3 }], { yMin: 0 }); sweepBtn.textContent = 'Sweep depth 1…12 (single vs bagged)'; }, 30); }, 'pri');
      root.appendChild(Ui.grid('g2', h('div', null, Ui.card('Variance made visible', cv.c), Ui.card('Bootstrap resampling', cvB.c)),
        h('div', null, Ui.card('Setup', h('div', { style: 'margin-bottom:8px' }, seg), Ui.row(Ui.slider({ label: 'tree depth', min: 1, max: 12, step: 1, value: P.depth, on: v => { P.depth = v; sched(); } }), Ui.slider({ label: 'bagged trees B', min: 2, max: 100, step: 1, value: P.B, on: v => { P.B = v; sched(); } })), Ui.row(Ui.slider({ label: 'noise σ', min: 0.05, max: 0.8, step: 0.05, value: P.noise, on: v => { P.noise = v; sched(); } }), Ui.slider({ label: 'training size n', min: 15, max: 120, step: 5, value: P.n, on: v => { P.n = v; sched(); } }))), Ui.card('Decomposition (measured)', cvBar.c), Ui.card('Sweep', sweepBtn, cvS.c, Ui.legend([['#7aa2c8', 'bias²'], ['#d9825b', 'variance'], ['#d7b56d', 'single total'], ['#86b894', 'bagged total']])))));
      update();
    },
  });

  /* ============================================================ T9 ====== */
  Lab.register({
    id: 't9', unit: 3, num: 'T9', title: 'Face recognition with ensembles',
    tag: 'PCA eigenfaces + Random Forest / Bagging / AdaBoost on a procedurally generated face gallery — then stress-test them with sunglasses, masks and noise.',
    star: 'Eigenface gallery · reconstruction slider · live probe with votes · occlusion stress test',
    idea: `<p>The notebook feeds PCA “eigenfaces” of the Olivetti dataset to three ensembles. The Olivetti photos need a download, so this page <b>generates a gallery of 12 synthetic people</b> (14 images each, with changes in lighting, pose shift, expression and noise) and runs the same pipeline: <b>PCA (whitened) → ensemble classifier</b>, all implemented from scratch.</p>
    <p>You can see the eigenfaces, drag a slider to see how many components are needed for a recognisable <b>reconstruction</b>, build a <b>probe</b> (any person, lighting, expression, occlusion) and read each model’s votes, and run a <b>stress test</b> that reports accuracy under sunglasses / mask / noise / darkness.</p>`,
    tryIt: ['Press <b>Train all models</b>. Random Forest and Bagging usually do well; AdaBoost with weak trees is typically the worst — the same pecking order as the notebook.',
      'Slide <b>components</b> down to 3–5: the reconstruction looks like a generic face — the identity lives in the later, finer eigenfaces.',
      'In the <b>probe</b>, choose “sunglasses” and watch the votes spread; whitened PCA features that depend on the eye region get corrupted.',
      'Run the <b>stress test</b>: which ensemble degrades most gracefully? (Random subspaces in the bagged 1-NN ensemble typically help.)'],
    hood: `<pre>faces: 24×24 grayscale, 12 identities × 14 images (8 train / 6 test each)
PCA(k, whiten): z = Uᵀ(x − μ) / √λ          (Gram-matrix trick: eigen-decompose the N×N matrix, N=96 ≪ D=576)
Random Forest : T trees, bootstrap rows, √k features per split, Gini
Bagging       : B × 1-NN on a bootstrap sample and a random half of the PCA features (random-subspace bagging)
AdaBoost.SAMME: α_t = ln((1−ε)/ε) + ln(K−1),  w_i ← w_i·exp(α_t·[h_t(x_i)≠y_i]) ,  depth-2 trees</pre>`,
    quiz: [
      { q: 'Why run PCA before the ensembles?', o: ['to increase dimensionality', 'to compress 576 pixels into a few decorrelated features that capture most variance', 'to label the data', 'to add noise'], a: 1, why: 'Eigenfaces give a compact, decorrelated representation that makes tree splits and distances meaningful.' },
      { q: 'Which trick makes PCA cheap when N ≪ D?', o: ['dropout', 'eigen-decompose the N×N Gram matrix instead of the D×D covariance', 'gradient boosting', 'resizing images'], a: 1, why: 'XXᵀ and XᵀX share nonzero eigenvalues; the eigenvectors of one give those of the other.' },
      { q: 'Random Forest differs from plain bagging of trees by…', o: ['using one tree', 'also sampling a random subset of features at each split', 'removing bootstraps', 'boosting weights'], a: 1, why: 'Feature subsampling de-correlates the trees, so averaging reduces variance more.' }],
    mount(root, ctx) {
      const NID = 12, PER = 14, NTR = 8, S = Faces.S; const names = Array.from({ length: NID }, (_, i) => 'P' + (i + 1));
      const D = Faces.make(NID, PER, 4); const trI = [], teI = []; D.X.forEach((_, i) => { (D.k[i] < NTR ? trI : teI).push(i); });
      let pca = null, kComp = 30, whiten = true, models = null, scores = null, trained = false;
      const toFeat = (x, k = kComp) => { const z = pca.transform(x, k); if (whiten) for (let c = 0; c < z.length; c++) z[c] /= Math.sqrt(pca.vals[c] + 1e-9); return z; };
      /* ----- ensemble implementations ----- */
      const Ens = {
        rf: { name: 'Random Forest', build(F, y, o) { const rng = new RNG(1), trees = [], d = F[0].length; for (let t = 0; t < o.T; t++) { const idx = Array.from({ length: F.length }, () => rng.int(F.length)); trees.push(new ClsTree({ maxDepth: 12, K: NID, mtry: Math.max(2, Math.round(Math.sqrt(d))), rng }).fit(F, y, null, idx)); } return { proba: f => { const p = new Float64Array(NID); trees.forEach(t => { const q = t.proba(f); for (let k = 0; k < NID; k++) p[k] += q[k] / trees.length; }); return p; } }; } },
        bag: { name: 'Bagging (1-NN, random subspaces)', build(F, y, o) { const rng = new RNG(2), d = F[0].length, ens = []; for (let b = 0; b < o.T; b++) { const idx = Array.from({ length: F.length }, () => rng.int(F.length)), feats = rng.shuffle(Array.from({ length: d }, (_, i) => i)).slice(0, Math.max(3, d >> 1)); ens.push({ idx, feats }); } return { proba: f => { const p = new Float64Array(NID); ens.forEach(({ idx, feats }) => { let bd = Infinity, bl = 0; for (const i of idx) { let s = 0; for (const c of feats) { const dd = F[i][c] - f[c]; s += dd * dd; } if (s < bd) { bd = s; bl = y[i]; } } p[bl] += 1 / ens.length; }); return p; } }; } },
        ada: { name: 'AdaBoost (SAMME, depth-2)', build(F, y, o) { const n = F.length, w = new Float64Array(n).fill(1 / n), hs = [], al = []; for (let t = 0; t < o.T; t++) { const tree = new ClsTree({ maxDepth: 2, K: NID }).fit(F, y, w); let err = 0; const pr = F.map(f => tree.predict(f)); for (let i = 0; i < n; i++) if (pr[i] !== y[i]) err += w[i]; if (err >= 1 - 1 / NID) break; err = Math.max(err, 1e-10); const a = Math.log((1 - err) / err) + Math.log(NID - 1); let s = 0; for (let i = 0; i < n; i++) { if (pr[i] !== y[i]) w[i] *= Math.exp(a); s += w[i]; } for (let i = 0; i < n; i++) w[i] /= s; hs.push(tree); al.push(a); if (err < 1e-9) break; } return { proba: f => { const p = new Float64Array(NID); let tot = 0; hs.forEach((t, i) => { p[t.predict(f)] += al[i]; tot += al[i]; }); for (let k = 0; k < NID; k++) p[k] /= tot || 1; return p; } }; } },
        tree: { name: 'Single decision tree (baseline)', build(F, y) { const t = new ClsTree({ maxDepth: 12, K: NID }).fit(F, y); return { proba: f => Float64Array.from(t.proba(f)) }; } },
      };
      const am = p => { let b = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[b]) b = k; return b; };
      /* ----- UI pieces ----- */
      const galleryCv = Ui.canvas(1160, 196), eigCv = Ui.canvas(600, 150), recCv = Ui.canvas(420, 130), probeCv = Ui.canvas(420, 200);
      function drawGallery() { const { g } = galleryCv; g.clearRect(0, 0, 1160, 196); for (let id = 0; id < NID; id++) for (let r = 0; r < 2; r++) { const i = id * PER + (r ? PER - 1 : 0); drawGray(g, D.X[i], S, S, 8 + id * 96, 4 + r * 90, 3.5); } g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; for (let id = 0; id < NID; id++) g.fillText(names[id], 40 + id * 96, 192); }
      function drawEig() { const { g } = eigCv; g.clearRect(0, 0, 600, 150); const mean = pca.mean; drawGray(g, mean, S, S, 6, 8, 3, 0, 1); g.fillStyle = '#9b9384'; g.font = '10px "Instrument Sans"'; g.fillText('mean', 22, 96); for (let c = 0; c < 11; c++) { const u = pca.comps[c]; let lo = Infinity, hi = -Infinity; u.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); }); const x = 86 + (c % 6) * 85, y = 4 + Math.floor(c / 6) * 72; drawGray(g, u, S, S, x, y, 3, lo, hi); g.fillText('λ' + (c + 1) + ' ' + (pca.vals[c] / pca.vals.reduce((a, b) => a + b, 0) * 100).toFixed(0) + '%', x + 12, y + 70); } }
      const probe = { id: 3, light: 0, smile: 0.3, dx: 0, occ: 'none', shot: 0 }; let probeImg = null;
      const stylesRng = new RNG(99);
      function makeProbe() { const v = { dx: probe.dx, dy: 0, scale: 1, light: probe.light, smile: probe.smile, eye: 1, noise: 0.03 }; probeImg = Faces.occlude(Faces.render(D.ids[probe.id], v, new RNG(5 + probe.shot)), probe.occ, new RNG(7)); }
      function drawProbe() {
        const { g, w, h: H } = probeCv; g.clearRect(0, 0, w, H); makeProbe(); drawGray(g, probeImg, S, S, 8, 8, 7); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('probe (true: ' + names[probe.id] + ')', 8, 196);
        if (!trained) { g.fillText('train the models to see their votes →', 190, 100); return; }
        const f = toFeat(probeImg); let y0 = 14; Object.keys(models).forEach(k => { const p = models[k].proba(f), top = [...p].map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3); g.fillStyle = '#ece6d8'; g.font = '600 11px "Instrument Sans"'; g.fillText(Ens[k].name.split(' (')[0], 190, y0); top.forEach(([v, i], j) => { g.fillStyle = i === probe.id ? '#86b894' : '#6f6859'; g.fillRect(190, y0 + 4 + j * 11, 150 * v, 8); g.fillStyle = '#cfc8b8'; g.font = '9.5px "Instrument Sans"'; g.fillText(names[i] + ' ' + (v * 100).toFixed(0) + '%', 195 + 150 * v, y0 + 11 + j * 11); }); y0 += 48; });
      }
      function drawRec() { const { g } = recCv; g.clearRect(0, 0, 420, 130); const i = teI[sel.get()], x = D.X[i], z = pca.transform(x, recK.get()), r = pca.inverse(z); drawGray(g, x, S, S, 10, 10, 4); drawGray(g, r, S, S, 130, 10, 4); const err = Math.sqrt(x.reduce((s, v, j) => s + (v - r[j]) ** 2, 0) / x.length); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText('original', 38, 120); g.fillText('k = ' + recK.get() + ' components', 140, 120); g.fillStyle = '#ece6d8'; g.font = '12px "JetBrains Mono",monospace'; g.fillText('RMSE ' + err.toFixed(3), 260, 60); g.fillText('variance kept ' + (pca.vals.slice(0, recK.get()).reduce((a, b) => a + b, 0) / pca.vals.reduce((a, b) => a + b, 0) * 100).toFixed(0) + '%', 260, 80); }
      const sel = Ui.slider({ label: 'test face', min: 0, max: teI.length - 1, step: 1, value: 4, on: () => pca && drawRec() }), recK = Ui.slider({ label: 'components k', min: 1, max: 80, step: 1, value: 20, on: () => pca && drawRec() });
      const accBox = h('div'), stressBox = h('div'), status = h('span', { class: 'pill' }, 'PCA not fitted');
      const nT = Ui.slider({ label: 'trees / estimators', min: 10, max: 150, step: 10, value: 60, on: () => { } }), kS = Ui.slider({ label: 'PCA components used', min: 5, max: 80, step: 5, value: 30, on: v => { kComp = v; } }), wChk = Ui.chk('whiten PCA features (as in the notebook)', true, v => { whiten = v; });
      function fitPCA() { pca = new ML.PCA(80).fit(trI.map(i => D.X[i])); status.textContent = 'PCA fitted on ' + trI.length + ' faces'; drawEig(); drawRec(); }
      function trainAll() {
        trainBtn.disabled = true; trainBtn.textContent = 'training…'; setTimeout(() => {
          const F = trI.map(i => toFeat(D.X[i])), y = trI.map(i => D.y[i]), o = { T: nT.get() }; models = {}; scores = {}; const t0 = performance.now();
          for (const k of ['tree', 'rf', 'bag', 'ada']) { models[k] = Ens[k].build(F, y, o); scores[k] = teI.reduce((s, i) => s + (am(models[k].proba(toFeat(D.X[i]))) === D.y[i] ? 1 : 0), 0) / teI.length; }
          trained = true; trainBtn.disabled = false; trainBtn.textContent = 'Retrain all models'; const mx = Math.max(...Object.values(scores));
          accBox.innerHTML = ''; accBox.appendChild(Ui.table(['model', 'test accuracy', ''], Object.keys(scores).map(k => [Ens[k].name, (scores[k] * 100).toFixed(1) + '%', h('div', { style: `height:10px;width:${scores[k] * 160}px;background:${scores[k] === mx ? '#86b894' : '#7aa2c8'};border-radius:4px` })]))); accBox.appendChild(h('div', { class: 'muted', style: 'font-size:12px;margin-top:6px' }, `trained in ${((performance.now() - t0) / 1000).toFixed(1)} s · ${teI.length} unseen test faces · chance = ${(100 / NID).toFixed(1)}%`)); stressBox.innerHTML = '';
        }, 30);
      }
      const trainBtn = Ui.btn('Train all models', trainAll, 'pri');
      function stress() {
        if (!trained) return Ui.toast('Train the models first'); stressBtn.textContent = 'testing…'; setTimeout(() => {
          const conds = ['none', 'sunglasses', 'mask', 'noise', 'dark'], rng = new RNG(8), rows = Object.keys(models).map(k => [Ens[k].name.split(' (')[0], ...conds.map(c => { let ok = 0; teI.forEach(i => { const x = Faces.occlude(D.X[i], c, rng); if (am(models[k].proba(toFeat(x))) === D.y[i]) ok++; }); return ok / teI.length; })]);
          stressBox.innerHTML = ''; const tb = h('table', { class: 't' }, h('thead', null, h('tr', null, ['model', ...conds].map(x => h('th', null, x)))), h('tbody', null, rows.map(r => h('tr', null, h('td', null, r[0]), r.slice(1).map(v => h('td', { style: `background:${Colors.rgba('#86b894', v * 0.55)};text-align:center` }, (v * 100).toFixed(0) + '%')))))); stressBox.appendChild(tb); stressBtn.textContent = 'Run stress test'; }, 30);
      }
      const stressBtn = Ui.btn('Run stress test', stress);
      const occSeg = Ui.seg(['none', 'sunglasses', 'mask', 'noise', 'dark'], 'none', v => { probe.occ = v; drawProbe(); });
      const idSel = Ui.select('probe identity', names.map((n, i) => ({ v: i, label: n })), 3, v => { probe.id = +v; drawProbe(); });
      root.appendChild(Ui.card('The gallery: 12 people × 14 images each (top row: image 1, bottom row: image 14 — same person, different light / pose / expression)', galleryCv.c)); root.appendChild(Ui.grid('g2', h('div', null, Ui.card('Eigenfaces  (PCA of the 96 training faces)', Ui.row(status), eigCv.c), Ui.card('Reconstruction from k components', Ui.row(sel, recK), recCv.c)),
        h('div', null, Ui.card('Ensembles', Ui.row(trainBtn), Ui.row(nT, kS), Ui.row(wChk), h('div', { style: 'margin-top:8px' }, accBox)), Ui.card('Probe lab', Ui.row(idSel, Ui.slider({ label: 'lighting', min: -1, max: 1, step: 0.1, value: 0, on: v => { probe.light = v; drawProbe(); } })), Ui.row(Ui.slider({ label: 'smile', min: -0.4, max: 1, step: 0.1, value: 0.3, on: v => { probe.smile = v; drawProbe(); } }), Ui.slider({ label: 'shift x', min: -2, max: 2, step: 0.5, value: 0, on: v => { probe.dx = v; drawProbe(); } })), h('div', { style: 'margin:6px 0' }, occSeg), probeCv.c), Ui.card('Occlusion stress test (accuracy on the 72 test faces)', stressBtn, stressBox))));
      drawGallery(); fitPCA(); drawProbe();
      ctx.cleanup(() => { });
    },
  });
})();
