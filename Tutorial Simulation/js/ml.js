/* AML Lab — pure-JS machine-learning engine (no DOM, no dependencies).
   Everything the simulations need is implemented from scratch here so the
   algorithms stay inspectable: nets + optimizers, CartPole, DQN, REINFORCE,
   GCN, CART trees, PCA, logistic regression. Works in browser and Node. */
(function (root) {
  'use strict';

  /* ---------------------------------------------------------------- RNG */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  class RNG {
    constructor(seed = 1) { this.f = mulberry32(seed); this.spare = null; }
    rand() { return this.f(); }
    uniform(a, b) { return a + (b - a) * this.f(); }
    int(n) { return Math.floor(this.f() * n); }
    normal() {
      if (this.spare !== null) { const s = this.spare; this.spare = null; return s; }
      let u = 0; while (u === 0) u = this.f();
      const v = this.f(), r = Math.sqrt(-2 * Math.log(u));
      this.spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    }
    shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = this.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; }
    choice(a) { return a[this.int(a.length)]; }
  }

  /* ------------------------------------------------- Optimizers (shared) */
  // params: [{p: Float64Array, g: Float64Array}]
  class Optimizer {
    constructor(kind, params, o = {}) {
      this.kind = kind; this.params = params;
      this.lr = o.lr ?? 0.01; this.b1 = o.beta1 ?? 0.9; this.b2 = o.beta2 ?? 0.999;
      this.mom = o.momentum ?? 0.9; this.eps = o.eps ?? 1e-8; this.wd = o.wd || 0; this.t = 0;
      this.m = params.map(q => new Float64Array(q.p.length));
      this.v = params.map(q => new Float64Array(q.p.length));
    }
    step() {
      this.t++;
      const { kind, lr, b1, b2, mom, eps, wd } = this;
      for (let k = 0; k < this.params.length; k++) {
        const p = this.params[k].p, g = this.params[k].g, m = this.m[k], v = this.v[k];
        for (let i = 0; i < p.length; i++) {
          let gi = g[i]; if (wd) gi += wd * p[i];
          switch (kind) {
            case 'sgd': p[i] -= lr * gi; break;
            case 'momentum': m[i] = mom * m[i] + gi; p[i] -= lr * m[i]; break;
            case 'nesterov': m[i] = mom * m[i] + gi; p[i] -= lr * (gi + mom * m[i]); break;
            case 'rmsprop': v[i] = 0.9 * v[i] + 0.1 * gi * gi; p[i] -= lr * gi / (Math.sqrt(v[i]) + eps); break;
            case 'adagrad': v[i] += gi * gi; p[i] -= lr * gi / (Math.sqrt(v[i]) + eps); break;
            case 'adam':
              m[i] = b1 * m[i] + (1 - b1) * gi; v[i] = b2 * v[i] + (1 - b2) * gi * gi;
              p[i] -= lr * (m[i] / (1 - Math.pow(b1, this.t))) / (Math.sqrt(v[i] / (1 - Math.pow(b2, this.t))) + eps); break;
            default: throw new Error('unknown optimizer ' + kind);
          }
        }
      }
    }
  }

  /* ------------------------------------------- Dense net (batched, BN, drop) */
  const ACT = {
    relu: { f: z => (z > 0 ? z : 0), d: (z, a) => (z > 0 ? 1 : 0) },
    tanh: { f: z => Math.tanh(z), d: (z, a) => 1 - a * a },
    sigmoid: { f: z => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a) },
  };
  class Net {
    constructor(sizes, o = {}) {
      this.sizes = sizes; this.act = o.act || 'relu'; this.bn = !!o.bn; this.p = o.dropout || 0;
      this.rng = o.rng || new RNG(1); this.L = sizes.length - 1; this.layers = [];
      for (let l = 0; l < this.L; l++) {
        const I = sizes[l], O = sizes[l + 1], hidden = l < this.L - 1;
        const W = new Float64Array(I * O);
        const sc = this.act === 'relu' ? Math.sqrt(2 / I) : Math.sqrt(2 / (I + O));
        for (let i = 0; i < W.length; i++) W[i] = this.rng.normal() * sc;
        const ly = { I, O, hidden, W, b: new Float64Array(O), gW: new Float64Array(I * O), gb: new Float64Array(O) };
        if (hidden && this.bn) {
          ly.gamma = new Float64Array(O).fill(1); ly.beta = new Float64Array(O);
          ly.gG = new Float64Array(O); ly.gB = new Float64Array(O);
          ly.rm = new Float64Array(O); ly.rv = new Float64Array(O).fill(1);
        }
        this.layers.push(ly);
      }
    }
    get params() {
      const r = [];
      for (const ly of this.layers) {
        r.push({ p: ly.W, g: ly.gW }, { p: ly.b, g: ly.gb });
        if (ly.gamma) r.push({ p: ly.gamma, g: ly.gG }, { p: ly.beta, g: ly.gB });
      }
      return r;
    }
    zeroGrad() { for (const q of this.params) q.g.fill(0); }
    copyFrom(o) {
      for (let l = 0; l < this.L; l++) {
        const a = this.layers[l], b = o.layers[l];
        a.W.set(b.W); a.b.set(b.b);
        if (a.gamma) { a.gamma.set(b.gamma); a.beta.set(b.beta); a.rm.set(b.rm); a.rv.set(b.rv); }
      }
    }
    softCopyFrom(o, tau) {
      for (let l = 0; l < this.L; l++) {
        const a = this.layers[l], b = o.layers[l];
        for (let i = 0; i < a.W.length; i++) a.W[i] += tau * (b.W[i] - a.W[i]);
        for (let i = 0; i < a.b.length; i++) a.b[i] += tau * (b.b[i] - a.b[i]);
      }
    }
    forward(X, n, train = false) {
      this.n = n; let a = X; const af = ACT[this.act].f;
      for (let l = 0; l < this.L; l++) {
        const ly = this.layers[l], { I, O, W, b } = ly;
        ly.x = a;
        const z = new Float64Array(n * O);
        for (let r = 0; r < n; r++) {
          const zo = r * O;
          for (let o = 0; o < O; o++) z[zo + o] = b[o];
          for (let i = 0; i < I; i++) {
            const xv = a[r * I + i]; if (xv === 0) continue;
            const wo = i * O;
            for (let o = 0; o < O; o++) z[zo + o] += xv * W[wo + o];
          }
        }
        if (!ly.hidden) { a = z; continue; }
        let y = z;
        if (ly.gamma) {
          const xhat = new Float64Array(n * O); y = new Float64Array(n * O);
          const mean = new Float64Array(O), vr = new Float64Array(O);
          if (train && n > 1) {
            for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) mean[o] += z[r * O + o] / n;
            for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) { const d = z[r * O + o] - mean[o]; vr[o] += d * d / n; }
            for (let o = 0; o < O; o++) { ly.rm[o] = 0.9 * ly.rm[o] + 0.1 * mean[o]; ly.rv[o] = 0.9 * ly.rv[o] + 0.1 * vr[o]; }
          } else { mean.set(ly.rm); vr.set(ly.rv); }
          const inv = new Float64Array(O);
          for (let o = 0; o < O; o++) inv[o] = 1 / Math.sqrt(vr[o] + 1e-5);
          for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) {
            const k = r * O + o; xhat[k] = (z[k] - mean[o]) * inv[o]; y[k] = ly.gamma[o] * xhat[k] + ly.beta[o];
          }
          ly.xhat = xhat; ly.inv = inv; ly.gammaTrain = train && n > 1;
        }
        ly.z = y;
        const out = new Float64Array(n * O);
        for (let k = 0; k < out.length; k++) out[k] = af(y[k]);
        ly.a = out;
        if (train && this.p > 0) {
          const mask = new Float64Array(out.length), keep = 1 - this.p;
          const o2 = new Float64Array(out.length);
          for (let k = 0; k < out.length; k++) { mask[k] = this.rng.rand() < this.p ? 0 : 1 / keep; o2[k] = out[k] * mask[k]; }
          ly.mask = mask; a = o2;
        } else { ly.mask = null; a = out; }
      }
      this.out = a; return a;
    }
    backward(dOut) {
      let d = dOut; const n = this.n, ad = ACT[this.act].d;
      for (let l = this.L - 1; l >= 0; l--) {
        const ly = this.layers[l], { I, O, W } = ly;
        if (ly.hidden) {
          const dd = new Float64Array(d.length);
          for (let k = 0; k < dd.length; k++) {
            let v = d[k]; if (ly.mask) v *= ly.mask[k];
            dd[k] = v * ad(ly.z[k], ly.a[k]);
          }
          d = dd;
          if (ly.gamma && ly.gammaTrain) {
            const sdx = new Float64Array(O), sdxx = new Float64Array(O);
            const dxh = new Float64Array(n * O);
            for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) {
              const k = r * O + o; ly.gG[o] += d[k] * ly.xhat[k]; ly.gB[o] += d[k];
              dxh[k] = d[k] * ly.gamma[o]; sdx[o] += dxh[k]; sdxx[o] += dxh[k] * ly.xhat[k];
            }
            const nd = new Float64Array(n * O);
            for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) {
              const k = r * O + o; nd[k] = (ly.inv[o] / n) * (n * dxh[k] - sdx[o] - ly.xhat[k] * sdxx[o]);
            }
            d = nd;
          } else if (ly.gamma) { // eval-mode BN: affine
            const nd = new Float64Array(n * O);
            for (let r = 0; r < n; r++) for (let o = 0; o < O; o++) { const k = r * O + o; ly.gG[o] += d[k] * ly.xhat[k]; ly.gB[o] += d[k]; nd[k] = d[k] * ly.gamma[o] * ly.inv[o]; }
            d = nd;
          }
        }
        const dIn = new Float64Array(n * I);
        for (let r = 0; r < n; r++) {
          for (let o = 0; o < O; o++) ly.gb[o] += d[r * O + o];
          for (let i = 0; i < I; i++) {
            const xv = ly.x[r * I + i]; let s = 0; const wo = i * O;
            for (let o = 0; o < O; o++) { const dv = d[r * O + o]; ly.gW[wo + o] += xv * dv; s += dv * W[wo + o]; }
            dIn[r * I + i] = s;
          }
        }
        d = dIn;
      }
      return d;
    }
  }

  /* ----------------------------------------------------------- Losses */
  // logits n×K, y Int array. Returns {loss, acc, d} with d = dL/dlogits (mean over n)
  function softmaxCE(logits, n, K, y) {
    const d = new Float64Array(n * K); let loss = 0, correct = 0;
    for (let r = 0; r < n; r++) {
      let mx = -Infinity, am = 0;
      for (let k = 0; k < K; k++) if (logits[r * K + k] > mx) { mx = logits[r * K + k]; am = k; }
      let s = 0;
      for (let k = 0; k < K; k++) { const e = Math.exp(logits[r * K + k] - mx); d[r * K + k] = e; s += e; }
      for (let k = 0; k < K; k++) d[r * K + k] /= s;
      loss -= Math.log(d[r * K + y[r]] + 1e-12); if (am === y[r]) correct++;
      d[r * K + y[r]] -= 1;
    }
    for (let i = 0; i < d.length; i++) d[i] /= n;
    return { loss: loss / n, acc: correct / n, d };
  }
  function softmax(v) {
    let mx = -Infinity; for (const x of v) mx = Math.max(mx, x);
    const e = Array.from(v, x => Math.exp(x - mx)); const s = e.reduce((a, b) => a + b, 0);
    return e.map(x => x / s);
  }
  // single logit per sample, labels 0/1
  function bceLogit(logits, n, y) {
    const d = new Float64Array(n); let loss = 0, correct = 0;
    for (let r = 0; r < n; r++) {
      const p = 1 / (1 + Math.exp(-logits[r]));
      loss -= y[r] * Math.log(p + 1e-12) + (1 - y[r]) * Math.log(1 - p + 1e-12);
      d[r] = (p - y[r]) / n; if ((p > 0.5 ? 1 : 0) === y[r]) correct++;
    }
    return { loss: loss / n, acc: correct / n, d };
  }

  /* ------------------------------------------------------------ CartPole */
  class CartPole {
    constructor(rng, maxSteps = 500) { this.rng = rng || new RNG(1); this.maxSteps = maxSteps; this.reset(); }
    reset() { this.s = [0, 0, 0, 0].map(() => this.rng.uniform(-0.05, 0.05)); this.t = 0; return this.obs(); }
    obs() { return Float64Array.from(this.s); }
    step(a, wind = 0) {
      const g = 9.8, mc = 1.0, mp = 0.1, tm = 1.1, l = 0.5, pml = mp * l, tau = 0.02;
      const F = (a === 1 ? 10 : -10) + wind;
      let [x, xd, th, thd] = this.s;
      const c = Math.cos(th), s = Math.sin(th);
      const temp = (F + pml * thd * thd * s) / tm;
      const thacc = (g * s - c * temp) / (l * (4 / 3 - mp * c * c / tm));
      const xacc = temp - pml * thacc * c / tm;
      x += tau * xd; xd += tau * xacc; th += tau * thd; thd += tau * thacc;
      this.s = [x, xd, th, thd]; this.t++;
      const terminated = Math.abs(x) > 2.4 || Math.abs(th) > 12 * Math.PI / 180;
      return { obs: this.obs(), reward: 1, terminated, truncated: this.t >= this.maxSteps };
    }
  }

  /* ------------------------------------------------------------------ DQN */
  class DQN {
    constructor(o = {}) {
      this.rng = o.rng || new RNG(1);
      this.hidden = o.hidden || 32; this.gamma = o.gamma ?? 0.99; this.batch = o.batch || 32;
      this.cap = o.capacity || 20000; this.useTarget = o.useTarget !== false; this.useReplay = o.useReplay !== false;
      this.tau = o.tau ?? 0.01; this.lr = o.lr ?? 1e-3; this.hardEvery = o.hardEvery ?? 200;
      this.net = new Net([4, this.hidden, this.hidden, 2], { rng: this.rng });
      this.target = new Net([4, this.hidden, this.hidden, 2], { rng: this.rng }); this.target.copyFrom(this.net);
      this.opt = new Optimizer('adam', this.net.params, { lr: this.lr });
      this.buf = []; this.pos = 0; this.steps = 0; this.last = null;
      this.eps = 1; this.epsEnd = o.epsEnd ?? 0.02; this.epsDecay = o.epsDecay ?? 0.98;
    }
    q(obs) { return this.net.forward(obs, 1, false).slice(); }
    act(obs, greedy = false) {
      if (!greedy && this.rng.rand() < this.eps) return this.rng.int(2);
      const q = this.net.forward(obs, 1, false); return q[1] > q[0] ? 1 : 0;
    }
    remember(s, a, r, s2, d) {
      const t = { s, a, r, s2, d };
      if (this.buf.length < this.cap) this.buf.push(t); else { this.buf[this.pos] = t; this.pos = (this.pos + 1) % this.cap; }
      this.last = t;
    }
    learn() {
      const B = this.useReplay ? this.batch : 1;
      if (this.useReplay && this.buf.length < this.batch) return 0;
      const S = new Float64Array(B * 4), S2 = new Float64Array(B * 4), A = [], R = [], D = [];
      for (let i = 0; i < B; i++) {
        const t = this.useReplay ? this.buf[this.rng.int(this.buf.length)] : this.last;
        S.set(t.s, i * 4); S2.set(t.s2, i * 4); A.push(t.a); R.push(t.r); D.push(t.d ? 1 : 0);
      }
      const tn = this.useTarget ? this.target : this.net;
      const qn = tn.forward(S2, B, false).slice();
      const best = []; for (let i = 0; i < B; i++) best.push(Math.max(qn[i * 2], qn[i * 2 + 1]));
      this.net.zeroGrad();
      const q = this.net.forward(S, B, true);
      const dq = new Float64Array(B * 2); let loss = 0;
      for (let i = 0; i < B; i++) {
        const tgt = R[i] + this.gamma * best[i] * (1 - D[i]);
        const e = q[i * 2 + A[i]] - tgt;
        dq[i * 2 + A[i]] = Math.max(-1, Math.min(1, e)) / B; loss += Math.abs(e) / B;
      }
      this.net.backward(dq); this.opt.step();
      if (this.useTarget) { if (this.hardEvery) { if (++this.steps % this.hardEvery === 0) this.target.copyFrom(this.net); } else this.target.softCopyFrom(this.net, this.tau); }
      return loss;
    }
    endEpisode() { this.eps = Math.max(this.epsEnd, this.eps * this.epsDecay); }
  }

  /* ------------------------------------------------------------ REINFORCE */
  class Reinforce {
    constructor(o = {}) {
      this.rng = o.rng || new RNG(2);
      this.gamma = o.gamma ?? 0.99; this.standardize = o.standardize !== false; this.entropy = o.entropy ?? 0.0;
      this.net = new Net([4, o.hidden || 32, 2], { act: 'tanh', rng: this.rng });
      this.opt = new Optimizer('adam', this.net.params, { lr: o.lr ?? 0.01 });
      this.S = []; this.A = []; this.R = [];
    }
    probs(obs) { return softmax(this.net.forward(obs, 1, false)); }
    act(obs, greedy = false) {
      const p = this.probs(obs);
      if (greedy) return p[1] > p[0] ? 1 : 0;
      return this.rng.rand() < p[1] ? 1 : 0;
    }
    record(s, a, r) { this.S.push(s); this.A.push(a); this.R.push(r); }
    endEpisode() {
      const T = this.R.length; const G = new Float64Array(T); let g = 0;
      for (let t = T - 1; t >= 0; t--) { g = this.R[t] + this.gamma * g; G[t] = g; }
      let mean = 0; for (const v of G) mean += v / T;
      let sd = 0; for (const v of G) sd += (v - mean) ** 2 / T; sd = Math.sqrt(sd);
      if (this.standardize) for (let t = 0; t < T; t++) G[t] = (G[t] - mean) / (sd + 1e-8);
      else for (let t = 0; t < T; t++) G[t] = G[t] / 100; // keep scale sane without standardisation
      const X = new Float64Array(T * 4); for (let t = 0; t < T; t++) X.set(this.S[t], t * 4);
      this.net.zeroGrad();
      const logits = this.net.forward(X, T, true);
      const d = new Float64Array(T * 2);
      for (let t = 0; t < T; t++) {
        const p = softmax([logits[t * 2], logits[t * 2 + 1]]);
        for (let k = 0; k < 2; k++) d[t * 2 + k] = (p[k] - (this.A[t] === k ? 1 : 0)) * G[t] / T;
      }
      this.net.backward(d); this.opt.step();
      this.S = []; this.A = []; this.R = [];
    }
  }

  /* ------------------------------------------------------------------ GCN */
  function matmul(A, n, m, B, k) { // (n×m)(m×k)
    const C = new Float64Array(n * k);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const a = A[i * m + j]; if (a === 0) continue;
      for (let c = 0; c < k; c++) C[i * k + c] += a * B[j * k + c];
    }
    return C;
  }
  function normAdj(n, edges, selfLoops = true) {
    const A = new Float64Array(n * n);
    for (const [u, v] of edges) { A[u * n + v] = 1; A[v * n + u] = 1; }
    if (selfLoops) for (let i = 0; i < n; i++) A[i * n + i] = 1;
    const deg = new Float64Array(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += A[i * n + j];
    const H = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (A[i * n + j]) H[i * n + j] = A[i * n + j] / Math.sqrt(deg[i] * deg[j]);
    return H;
  }
  class GCN {
    constructor(n, nFeat, hidden, C, layers, o = {}) {
      this.n = n; this.C = C; this.L = layers; this.rng = o.rng || new RNG(0); this.dropout = o.dropout ?? 0.3;
      this.sizes = [nFeat]; for (let i = 0; i < layers - 1; i++) this.sizes.push(hidden); this.sizes.push(C);
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      for (let l = 0; l < layers; l++) {
        const I = this.sizes[l], O = this.sizes[l + 1], W = new Float64Array(I * O);
        const sc = Math.sqrt(6 / (I + O)); for (let i = 0; i < W.length; i++) W[i] = this.rng.uniform(-sc, sc);
        this.W.push(W); this.b.push(new Float64Array(O)); this.gW.push(new Float64Array(I * O)); this.gb.push(new Float64Array(O));
      }
      const ps = []; for (let l = 0; l < layers; l++) ps.push({ p: this.W[l], g: this.gW[l] }, { p: this.b[l], g: this.gb[l] });
      this.opt = new Optimizer('adam', ps, { lr: o.lr ?? 0.01, wd: o.wd ?? 5e-4 });
    }
    setGraph(Ahat, X) { this.A = Ahat; this.X = X; }
    forward(train) {
      const n = this.n; this.cache = []; let H = this.X;
      for (let l = 0; l < this.L; l++) {
        const I = this.sizes[l], O = this.sizes[l + 1];
        const P = matmul(this.A, n, n, H, I);
        const Z = matmul(P, n, I, this.W[l], O);
        for (let i = 0; i < n; i++) for (let o = 0; o < O; o++) Z[i * O + o] += this.b[l][o];
        let Hn = Z, mask = null;
        if (l < this.L - 1) {
          Hn = new Float64Array(Z.length); for (let k = 0; k < Z.length; k++) Hn[k] = Z[k] > 0 ? Z[k] : 0;
          if (train && this.dropout > 0) {
            mask = new Float64Array(Z.length); const keep = 1 - this.dropout;
            for (let k = 0; k < Z.length; k++) { mask[k] = this.rng.rand() < this.dropout ? 0 : 1 / keep; Hn[k] *= mask[k]; }
          }
        }
        this.cache.push({ P, Z, mask, I, O }); H = Hn;
      }
      this.logits = H; return H;
    }
    hidden() { // last hidden representation (post ReLU) for embedding plots
      if (this.L < 2) return null; const c = this.cache[this.L - 2]; const out = new Float64Array(c.Z.length);
      for (let k = 0; k < out.length; k++) out[k] = Math.max(0, c.Z[k]); return { H: out, dim: c.O };
    }
    trainStep(y, mask) { // mask: boolean array of labelled nodes
      const n = this.n, C = this.C; const logits = this.forward(true);
      const m = mask.reduce((s, v) => s + (v ? 1 : 0), 0); if (!m) return { loss: 0 };
      let dZ = new Float64Array(n * C), loss = 0;
      for (let i = 0; i < n; i++) if (mask[i]) {
        const p = softmax(logits.subarray(i * C, (i + 1) * C)); loss -= Math.log(p[y[i]] + 1e-12);
        for (let c = 0; c < C; c++) dZ[i * C + c] = (p[c] - (c === y[i] ? 1 : 0)) / m;
      }
      for (let l = this.L - 1; l >= 0; l--) {
        const { P, I, O } = this.cache[l]; this.gW[l].fill(0); this.gb[l].fill(0);
        for (let i = 0; i < n; i++) for (let o = 0; o < O; o++) {
          const dz = dZ[i * O + o]; if (dz === 0) continue; this.gb[l][o] += dz;
          for (let k = 0; k < I; k++) this.gW[l][k * O + o] += P[i * I + k] * dz;
        }
        if (l > 0) {
          const dP = new Float64Array(n * I);
          for (let i = 0; i < n; i++) for (let o = 0; o < O; o++) { const dz = dZ[i * O + o]; if (dz === 0) continue; for (let k = 0; k < I; k++) dP[i * I + k] += dz * this.W[l][k * O + o]; }
          const dH = matmul(this.A, n, n, dP, I); const pc = this.cache[l - 1];
          const nd = new Float64Array(n * I);
          for (let k = 0; k < nd.length; k++) nd[k] = (pc.Z[k] > 0 ? 1 : 0) * dH[k] * (pc.mask ? pc.mask[k] : 1);
          dZ = nd;
        }
      }
      this.opt.step(); return { loss: loss / m };
    }
    predict() {
      const lg = this.forward(false), n = this.n, C = this.C, probs = [], pred = [];
      for (let i = 0; i < n; i++) { const p = softmax(lg.subarray(i * C, (i + 1) * C)); probs.push(p); pred.push(p.indexOf(Math.max(...p))); }
      return { probs, pred };
    }
  }

  /* --------------------------------------------------------------- Trees */
  class ClsTree {
    constructor(o = {}) { this.maxDepth = o.maxDepth ?? 5; this.minLeaf = o.minLeaf ?? 1; this.mtry = o.mtry || 0; this.K = o.K; this.rng = o.rng || new RNG(1); }
    fit(X, y, w, idx) {
      const n = X.length; this.d = X[0].length;
      w = w || new Float64Array(n).fill(1); idx = idx || Array.from({ length: n }, (_, i) => i);
      this.root = this.build(X, y, w, idx, 0); return this;
    }
    build(X, y, w, idx, depth) {
      const K = this.K, cnt = new Float64Array(K); let W = 0;
      for (const i of idx) { cnt[y[i]] += w[i]; W += w[i]; }
      const leaf = () => ({ p: Array.from(cnt, c => (W > 0 ? c / W : 1 / K)) });
      let pure = 0; for (let k = 0; k < K; k++) if (cnt[k] > 0) pure++;
      if (depth >= this.maxDepth || idx.length < 2 * this.minLeaf || pure <= 1) return leaf();
      let feats = Array.from({ length: this.d }, (_, i) => i);
      if (this.mtry && this.mtry < this.d) { this.rng.shuffle(feats); feats = feats.slice(0, this.mtry); }
      let bestScore = Infinity, bf = -1, bt = 0;
      const parent = (() => { let s = 0; for (let k = 0; k < K; k++) s += cnt[k] * cnt[k]; return W - s / W; })();
      for (const f of feats) {
        const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
        const cl = new Float64Array(K), cr = Float64Array.from(cnt); let Wl = 0, Wr = W;
        for (let j = 0; j < s.length - 1; j++) {
          const i = s[j]; cl[y[i]] += w[i]; cr[y[i]] -= w[i]; Wl += w[i]; Wr -= w[i];
          if (X[s[j + 1]][f] === X[i][f]) continue;
          if (j + 1 < this.minLeaf || s.length - j - 1 < this.minLeaf) continue;
          let sl = 0, sr = 0; for (let k = 0; k < K; k++) { sl += cl[k] * cl[k]; sr += cr[k] * cr[k]; }
          const score = (Wl - (Wl > 0 ? sl / Wl : 0)) + (Wr - (Wr > 0 ? sr / Wr : 0));
          if (score < bestScore - 1e-12) { bestScore = score; bf = f; bt = (X[i][f] + X[s[j + 1]][f]) / 2; }
        }
      }
      if (bf < 0 || bestScore >= parent - 1e-12) return leaf();
      const L = [], R = []; for (const i of idx) (X[i][bf] <= bt ? L : R).push(i);
      return { f: bf, t: bt, l: this.build(X, y, w, L, depth + 1), r: this.build(X, y, w, R, depth + 1) };
    }
    proba(x) { let n = this.root; while (n.p === undefined) n = x[n.f] <= n.t ? n.l : n.r; return n.p; }
    predict(x) { const p = this.proba(x); let b = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[b]) b = k; return b; }
  }

  class RegTree {
    constructor(o = {}) { this.maxDepth = o.maxDepth ?? 3; this.minLeaf = o.minLeaf ?? 1; this.leafFn = o.leafFn || null; }
    fit(X, y, idx) {
      idx = idx || Array.from({ length: X.length }, (_, i) => i);
      this.d = X[0].length; this.root = this.build(X, y, idx, 0); return this;
    }
    build(X, y, idx, depth) {
      const n = idx.length; let sum = 0, sq = 0; for (const i of idx) { sum += y[i]; sq += y[i] * y[i]; }
      const mk = () => ({ v: this.leafFn ? this.leafFn(idx) : sum / n });
      if (depth >= this.maxDepth || n < 2 * this.minLeaf) return mk();
      let best = sq - sum * sum / n, bf = -1, bt = 0; const base = best;
      for (let f = 0; f < this.d; f++) {
        const s = idx.slice().sort((a, b) => X[a][f] - X[b][f]); let ls = 0, lq = 0;
        for (let j = 0; j < n - 1; j++) {
          const i = s[j]; ls += y[i]; lq += y[i] * y[i];
          if (X[s[j + 1]][f] === X[i][f]) continue;
          const nl = j + 1, nr = n - nl; if (nl < this.minLeaf || nr < this.minLeaf) continue;
          const rs = sum - ls, rq = sq - lq;
          const sc = (lq - ls * ls / nl) + (rq - rs * rs / nr);
          if (sc < best - 1e-12) { best = sc; bf = f; bt = (X[i][f] + X[s[j + 1]][f]) / 2; }
        }
      }
      if (bf < 0 || best >= base - 1e-12) return mk();
      const L = [], R = []; for (const i of idx) (X[i][bf] <= bt ? L : R).push(i);
      return { f: bf, t: bt, l: this.build(X, y, L, depth + 1), r: this.build(X, y, R, depth + 1) };
    }
    predict(x) { let n = this.root; while (n.v === undefined) n = x[n.f] <= n.t ? n.l : n.r; return n.v; }
  }

  /* ----------------------------------------------- Small linear-algebra bits */
  function jacobiEig(A, n) { // symmetric eigen-decomposition; returns {vals, vecs(columns)}
    const a = Float64Array.from(A), v = new Float64Array(n * n); for (let i = 0; i < n; i++) v[i * n + i] = 1;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0; for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i * n + j] ** 2;
      if (off < 1e-18) break;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q]; if (Math.abs(apq) < 1e-15) continue;
        const th = (a[q * n + q] - a[p * n + p]) / (2 * apq);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1)), c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k * n + p], akq = a[k * n + q]; a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq; }
        for (let k = 0; k < n; k++) { const apk = a[p * n + k], aqk = a[q * n + k]; a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk; }
        for (let k = 0; k < n; k++) { const vkp = v[k * n + p], vkq = v[k * n + q]; v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq; }
      }
    }
    const vals = Array.from({ length: n }, (_, i) => a[i * n + i]);
    const order = vals.map((x, i) => i).sort((i, j) => vals[j] - vals[i]);
    return { vals: order.map(i => vals[i]), vecs: order.map(i => Float64Array.from({ length: n }, (_, k) => v[k * n + i])) };
  }
  // PCA via the Gram-matrix trick (N << D). X: array of Float64Array rows.
  class PCA {
    constructor(k) { this.k = k; }
    fit(X) {
      const N = X.length, D = X[0].length; this.D = D;
      this.mean = new Float64Array(D); for (const x of X) for (let j = 0; j < D; j++) this.mean[j] += x[j] / N;
      const C = X.map(x => x.map((v, j) => v - this.mean[j]));
      const G = new Float64Array(N * N);
      for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { let s = 0; for (let k = 0; k < D; k++) s += C[i][k] * C[j][k]; G[i * N + j] = G[j * N + i] = s / N; }
      const { vals, vecs } = jacobiEig(G, N); this.vals = vals; this.comps = [];
      for (let c = 0; c < Math.min(this.k, N - 1); c++) {
        const u = new Float64Array(D); for (let i = 0; i < N; i++) { const w = vecs[c][i]; for (let j = 0; j < D; j++) u[j] += w * C[i][j]; }
        let nm = 0; for (let j = 0; j < D; j++) nm += u[j] * u[j]; nm = Math.sqrt(nm) || 1; for (let j = 0; j < D; j++) u[j] /= nm;
        this.comps.push(u);
      }
      return this;
    }
    transform(x, k) {
      k = Math.min(k ?? this.comps.length, this.comps.length); const z = new Float64Array(k);
      for (let c = 0; c < k; c++) { let s = 0; const u = this.comps[c]; for (let j = 0; j < this.D; j++) s += (x[j] - this.mean[j]) * u[j]; z[c] = s; }
      return z;
    }
    inverse(z) { const x = Float64Array.from(this.mean); for (let c = 0; c < z.length; c++) { const u = this.comps[c]; for (let j = 0; j < this.D; j++) x[j] += z[c] * u[j]; } return x; }
  }

  // multinomial logistic regression (full-batch GD, L2)
  class Logistic {
    constructor(d, K, o = {}) { this.d = d; this.K = K; this.W = new Float64Array(d * K); this.b = new Float64Array(K); this.lr = o.lr ?? 0.3; this.l2 = o.l2 ?? 1e-3; }
    proba(x) { const z = new Array(this.K).fill(0); for (let k = 0; k < this.K; k++) { let s = this.b[k]; for (let j = 0; j < this.d; j++) s += x[j] * this.W[j * this.K + k]; z[k] = s; } return softmax(z); }
    fit(X, y, epochs = 200) {
      const n = X.length;
      for (let e = 0; e < epochs; e++) {
        const gW = new Float64Array(this.d * this.K), gb = new Float64Array(this.K);
        for (let i = 0; i < n; i++) {
          const p = this.proba(X[i]);
          for (let k = 0; k < this.K; k++) { const g = p[k] - (y[i] === k ? 1 : 0); gb[k] += g / n; for (let j = 0; j < this.d; j++) if (X[i][j]) gW[j * this.K + k] += g * X[i][j] / n; }
        }
        for (let i = 0; i < gW.length; i++) this.W[i] -= this.lr * (gW[i] + this.l2 * this.W[i]);
        for (let k = 0; k < this.K; k++) this.b[k] -= this.lr * gb[k];
      }
      return this;
    }
    predict(x) { const p = this.proba(x); let b = 0; for (let k = 1; k < this.K; k++) if (p[k] > p[b]) b = k; return b; }
  }

  root.ML = { RNG, Optimizer, Net, softmaxCE, softmax, bceLogit, CartPole, DQN, Reinforce, GCN, normAdj, matmul, ClsTree, RegTree, jacobiEig, PCA, Logistic };
  if (typeof module !== 'undefined') module.exports = root.ML;
})(typeof window !== 'undefined' ? window : globalThis);
