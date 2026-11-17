/* AML Lab — small UI kit: DOM builder, controls, canvas helpers, charts, force layout */
(function (root) {
  'use strict';

  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const fmt = (v, d = 3) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(1) : (+v.toFixed(d)).toString()) : String(v));

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    if (props) for (const k in props) {
      const v = props[k]; if (v === undefined || v === null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'style') { if (typeof v === 'string') el.style.cssText = v; else Object.assign(el.style, v); }
      else if (k === 'html') el.innerHTML = v;
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else if (k === 'value') el.value = v;
      else el.setAttribute(k, v === true ? '' : v);
    }
    const add = k => {
      if (k === null || k === undefined || k === false) return;
      if (Array.isArray(k)) k.forEach(add);
      else el.appendChild(k.nodeType ? k : document.createTextNode(String(k)));
    };
    kids.forEach(add); return el;
  }

  /* ------------------------------------------------------------ colours */
  const PAL = ['#7aa2c8', '#d9825b', '#86b894', '#d7b56d', '#a78bc4', '#c97b9e', '#8fc3d4', '#b4c46a'];
  const classColor = i => PAL[((i % PAL.length) + PAL.length) % PAL.length];
  function hex2rgb(hx) { const n = parseInt(hx.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function rgba(hx, a) { const [r, g, b] = hex2rgb(hx); return `rgba(${r},${g},${b},${a})`; }
  function mix(c1, c2, t) { const a = hex2rgb(c1), b = hex2rgb(c2); return a.map((v, i) => Math.round(lerp(v, b[i], t))); }
  const VIR = ['#2a1f2b', '#6b3f5c', '#b8634a', '#dca25f', '#f3e0a8'];
  function viridis(t) { t = clamp(t, 0, 1) * 4; const i = Math.min(3, Math.floor(t)); const m = mix(VIR[i], VIR[i + 1], t - i); return `rgb(${m[0]},${m[1]},${m[2]})`; }
  function viridisRGB(t) { t = clamp(t, 0, 1) * 4; const i = Math.min(3, Math.floor(t)); return mix(VIR[i], VIR[i + 1], t - i); }
  // diverging blue(−) – dark – orange(+) on dark background, t in [-1,1]
  function diverge(t) {
    t = clamp(t, -1, 1);
    const m = t < 0 ? mix('#26221d', '#628bb3', -t) : mix('#26221d', '#cf7a52', t); return `rgb(${m[0]},${m[1]},${m[2]})`;
  }
  const Colors = { PAL, classColor, rgba, viridis, viridisRGB, diverge, hex2rgb, mix };

  /* -------------------------------------------------------------- canvas */
  function canvas(w, h2, o = {}) {
    const c = document.createElement('canvas'), dpr = Math.min(2, root.devicePixelRatio || 1);
    c.width = Math.round(w * dpr); c.height = Math.round(h2 * dpr);
    c.style.width = '100%'; c.style.maxWidth = w + 'px'; c.style.aspectRatio = w + '/' + h2;
    if (o.free) c.className = 'free';
    const g = c.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cv = {
      c, g, w, h: h2,
      xy(e) { const r = c.getBoundingClientRect(), t = e.touches && e.touches[0] ? e.touches[0] : e; return [(t.clientX - r.left) * w / r.width, (t.clientY - r.top) * h2 / r.height]; },
      clear(col) { g.clearRect(0, 0, w, h2); if (col) { g.fillStyle = col; g.fillRect(0, 0, w, h2); } },
    };
    return cv;
  }
  // draw a w×h grid of values (Float array 0..1) into a canvas context at (x,y), cell size s
  function drawGray(g, data, W, H, x, y, s, lo = 0, hi = 1) {
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
      const v = clamp((data[j * W + i] - lo) / (hi - lo || 1), 0, 1) * 255 | 0;
      g.fillStyle = `rgb(${v},${v},${v})`; g.fillRect(x + i * s, y + j * s, s + 0.5, s + 0.5);
    }
  }

  /* ----------------------------------------------------------- controls */
  const Ui = {
    esc, clamp, fmt, lerp,
    btn(label, onClick, cls = '') { return h('button', { class: 'btn ' + cls, onclick: onClick }, label); },
    slider(o) {
      const fm = o.fmt || (v => fmt(v)); const val = h('span', null, fm(o.value));
      const inp = h('input', { type: 'range', min: o.min, max: o.max, step: o.step ?? 1, value: o.value });
      const fill = () => { const p = (+inp.value - +inp.min) / ((+inp.max - +inp.min) || 1) * 100; inp.style.setProperty('--p', p + '%'); };
      const el = h('div', { class: 'ctl', style: o.style || '' }, h('label', null, h('em', null, o.label), val), inp); fill();
      inp.addEventListener('input', () => { val.textContent = fm(+inp.value); fill(); o.on && o.on(+inp.value); });
      el.get = () => +inp.value; el.set = (v, silent) => { inp.value = v; val.textContent = fm(+inp.value); fill(); if (!silent && o.on) o.on(+inp.value); };
      el.disable = d => { inp.disabled = d; el.style.opacity = d ? .45 : 1; };
      return el;
    },
    seg(options, value, on) {
      const opts = options.map(o => (typeof o === 'object' ? o : { v: o, label: o }));
      let cur = value; const btns = opts.map(o => h('button', { class: o.v === cur ? 'on' : '', onclick: () => set(o.v) }, o.label));
      const el = h('div', { class: 'seg' }, btns);
      function set(v, silent) { cur = v; opts.forEach((o, i) => btns[i].classList.toggle('on', o.v === v)); if (!silent && on) on(v); }
      el.set = set; el.get = () => cur; return el;
    },
    select(label, options, value, on) {
      const s = h('select', { onchange: () => on && on(s.value) }, options.map(o => { const op = typeof o === 'object' ? o : { v: o, label: o }; return h('option', { value: op.v, selected: op.v === value }, op.label); }));
      const el = h('div', { class: 'ctl' }, h('label', null, label), s); el.get = () => s.value; el.set = v => { s.value = v; }; return el;
    },
    chk(label, checked, on) {
      const i = h('input', { type: 'checkbox' }); i.checked = !!checked; i.addEventListener('change', () => on && on(i.checked));
      const el = h('label', { class: 'chk' }, i, h('span', null, label)); el.get = () => i.checked; el.set = v => { i.checked = v; }; return el;
    },
    stat(label, init = '–') { const b = h('b', null, init); const el = h('div', { class: 'stat' }, h('small', null, label), b); el.set = v => { b.textContent = v; }; return el; },
    card(title, ...kids) { return h('div', { class: 'card' }, title ? h('h4', null, title) : null, kids); },
    row(...k) { return h('div', { class: 'row' }, k); },
    col(...k) { return h('div', { class: 'col' }, k); },
    grid(cls, ...k) { return h('div', { class: 'grid ' + cls }, k); },
    note(html) { return h('div', { class: 'note', html }); },
    legend(items) { return h('div', { class: 'legend' }, items.map(([c, t]) => h('span', null, h('i', { style: 'background:' + c }), t))); },
    table(head, rows, o = {}) {
      return h('div', { class: 'scroll', style: o.maxH ? 'max-height:' + o.maxH : '' }, h('table', { class: 't' },
        h('thead', null, h('tr', null, head.map(x => h('th', null, x)))),
        h('tbody', null, rows.map(r => h('tr', null, r.map(c => h('td', null, c)))))));
    },
    toast(msg) { const t = document.getElementById('toast'); if (!t) return; t.textContent = msg; t.classList.add('on'); clearTimeout(Ui._tt); Ui._tt = setTimeout(() => t.classList.remove('on'), 1800); },
    canvas,
  };

  /* --------------------------------------------------------- line charts */
  function niceTicks(lo, hi, n = 5) {
    if (hi === lo) { hi = lo + 1; }
    const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const r = step0 / mag, step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(+v.toPrecision(12)); return out;
  }
  class LineChart {
    constructor(cv, o = {}) { this.cv = cv; this.o = o; }
    draw(series, o2 = {}) {
      const o = Object.assign({}, this.o, o2), { g, w, h: H } = this.cv;
      const m = { l: 46, r: 12, t: o.title ? 26 : 12, b: o.xlabel ? 34 : 22 };
      g.clearRect(0, 0, w, H);
      g.font = '11.5px "Instrument Sans"'; let lrows = 1, lacc = 0; const lavail = w - m.l - m.r - 6;
      for (const s of series) if (s.name && ((s.data && s.data.length) || (s.pts && s.pts.length))) { const wd = 24 + g.measureText(s.name).width; if (lacc + wd > lavail && lacc > 0) { lrows++; lacc = 0; } lacc += wd; }
      m.t += (lrows - 1) * 14;
      let xs0 = Infinity, xs1 = -Infinity, ys0 = Infinity, ys1 = -Infinity;
      const S = series.filter(s => (s.data && s.data.length) || (s.pts && s.pts.length)).map(s => {
        const pts = s.pts || s.data.map((y, i) => [i, y]); return Object.assign({}, s, { pts });
      });
      for (const s of S) for (const [x, y] of s.pts) { if (!Number.isFinite(y)) continue; xs0 = Math.min(xs0, x); xs1 = Math.max(xs1, x); ys0 = Math.min(ys0, y); ys1 = Math.max(ys1, y); }
      if (!S.length) { g.fillStyle = '#6f6859'; g.font = '13px "Instrument Sans"'; g.fillText(o.empty || 'no data yet', w / 2 - 36, H / 2); return; }
      if (o.xMin !== undefined) xs0 = o.xMin; if (o.xMax !== undefined) xs1 = o.xMax;
      if (o.yMin !== undefined) ys0 = o.yMin; if (o.yMax !== undefined) ys1 = o.yMax;
      if (ys1 === ys0) { ys1 += 1; ys0 -= 1; } if (xs1 === xs0) xs1 = xs0 + 1;
      const pad = (ys1 - ys0) * 0.04; if (o.yMin === undefined) ys0 -= pad; if (o.yMax === undefined) ys1 += pad;
      const X = x => m.l + (x - xs0) / (xs1 - xs0) * (w - m.l - m.r), Y = y => H - m.b - (y - ys0) / (ys1 - ys0) * (H - m.t - m.b);
      g.font = '11px "JetBrains Mono",monospace'; g.lineWidth = 1;
      for (const v of niceTicks(ys0, ys1, 5)) { const y = Y(v); g.strokeStyle = '#2a2620'; g.beginPath(); g.moveTo(m.l, y); g.lineTo(w - m.r, y); g.stroke(); g.fillStyle = '#9b9384'; g.textAlign = 'right'; g.fillText(fmt(v, 2), m.l - 5, y + 4); }
      for (const v of niceTicks(xs0, xs1, 6)) { const x = X(v); g.strokeStyle = '#25211c'; g.beginPath(); g.moveTo(x, m.t); g.lineTo(x, H - m.b); g.stroke(); g.fillStyle = '#9b9384'; g.textAlign = 'center'; g.fillText(fmt(v, 2), x, H - m.b + 14); }
      if (o.title) { g.fillStyle = '#ece6d8'; g.font = '600 12.5px "Instrument Sans"'; g.textAlign = 'left'; g.fillText(o.title, m.l, 16); }
      if (o.xlabel) { g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.textAlign = 'center'; g.fillText(o.xlabel, (w + m.l) / 2, H - 4); }
      for (const hl of (o.hlines || [])) { const y = Y(hl.y); g.strokeStyle = hl.color || '#6f6859'; g.setLineDash([5, 4]); g.beginPath(); g.moveTo(m.l, y); g.lineTo(w - m.r, y); g.stroke(); g.setLineDash([]); if (hl.label) { g.fillStyle = hl.color || '#9b9384'; g.textAlign = 'right'; g.fillText(hl.label, w - m.r - 2, y - 3); } }
      g.save(); g.beginPath(); g.rect(m.l, m.t - 2, w - m.l - m.r, H - m.t - m.b + 4); g.clip();
      for (const s of S) {
        g.strokeStyle = s.color || '#7aa2c8'; g.lineWidth = s.width || 2; g.globalAlpha = s.alpha ?? 1; g.setLineDash(s.dash || []);
        g.beginPath(); let st = false;
        for (const [x, y] of s.pts) { if (!Number.isFinite(y)) { st = false; continue; } if (!st) { g.moveTo(X(x), Y(y)); st = true; } else g.lineTo(X(x), Y(y)); }
        g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
        if (s.dots) for (const [x, y] of s.pts) { g.fillStyle = s.color; g.beginPath(); g.arc(X(x), Y(y), 3, 0, 7); g.fill(); }
      }
      g.restore();
      // legend
      let lx = m.l + 6, ly = m.t - (lrows - 1) * 14 + 10; g.font = '11.5px "Instrument Sans"'; g.textAlign = 'left';
      for (const s of S) if (s.name) { const wd = 24 + g.measureText(s.name).width; if (lx + wd > w - m.r && lx > m.l + 6) { lx = m.l + 6; ly += 14; } g.fillStyle = s.color; g.fillRect(lx, ly - 7, 12, 3); g.fillStyle = '#cfc8b8'; g.fillText(s.name, lx + 16, ly - 2); lx += wd; }
      this.map = { X, Y, xs0, xs1, ys0, ys1 };
    }
  }

  /* ------------------------------------------------------- force layout */
  class Force {
    constructor(nodes, edges, o = {}) {
      this.nodes = nodes; this.edges = edges; this.w = o.w || 600; this.h = o.h || 400;
      this.k = o.k || Math.sqrt(this.w * this.h / Math.max(nodes.length, 1)) * 0.75; this.energy = 1;
      nodes.forEach((n, i) => { if (n.x === undefined) { const a = i / nodes.length * 6.283; n.x = this.w / 2 + Math.cos(a) * this.w * 0.3 + Math.random() * 8; n.y = this.h / 2 + Math.sin(a) * this.h * 0.3 + Math.random() * 8; } n.vx = n.vy = 0; });
    }
    step(iter = 1) {
      const N = this.nodes, k = this.k, k2 = k * k;
      for (let it = 0; it < iter; it++) {
        for (let i = 0; i < N.length; i++) {
          const a = N[i];
          for (let j = i + 1; j < N.length; j++) {
            const b = N[j]; let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01; const d = Math.sqrt(d2);
            const f = k2 / d2 * 0.9; dx = dx / d * f; dy = dy / d * f; a.vx += dx; a.vy += dy; b.vx -= dx; b.vy -= dy;
          }
        }
        for (const [u, v] of this.edges) {
          const a = N[u], b = N[v]; const dx = b.x - a.x, dy = b.y - a.y, d = Math.sqrt(dx * dx + dy * dy) + 0.01;
          const f = (d - k * 0.9) * 0.08; a.vx += dx / d * f * 2; a.vy += dy / d * f * 2; b.vx -= dx / d * f * 2; b.vy -= dy / d * f * 2;
        }
        let e = 0;
        for (const n of N) {
          n.vx += (this.w / 2 - n.x) * 0.012; n.vy += (this.h / 2 - n.y) * 0.012;
          if (n.fixed) { n.vx = n.vy = 0; continue; }
          const sp = Math.hypot(n.vx, n.vy), mx = 18; if (sp > mx) { n.vx *= mx / sp; n.vy *= mx / sp; }
          n.x += n.vx * 0.5; n.y += n.vy * 0.5; n.vx *= 0.6; n.vy *= 0.6; e += sp;
          n.x = clamp(n.x, 14, this.w - 14); n.y = clamp(n.y, 14, this.h - 14);
        }
        this.energy = e / N.length;
      }
    }
  }

  root.Lab = root.Lab || { tasks: [], register(def) { def.order = this.tasks.length; this.tasks.push(def); } };
  root.h = h; root.Ui = Ui; root.Colors = Colors; root.LineChart = LineChart; root.Force = Force; root.drawGray = drawGray;
})(window);
