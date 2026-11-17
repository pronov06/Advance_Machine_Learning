/* Unit 2 — ML with Graphs: T4 GCN, T5 Knowledge graph + SPARQL + TransE, T6 Frequent subgraph mining, B2 PageRank */
(function () {
  'use strict';
  const { RNG } = ML;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = (tag, a = {}, ...kids) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); kids.flat().forEach(c => c != null && e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)); return e; };

  /* Zachary's karate club (34 nodes, 78 edges) */
  const ADJ = { 0: [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 17, 19, 21, 31], 1: [2, 3, 7, 13, 17, 19, 21, 30], 2: [3, 7, 8, 9, 13, 27, 28, 32], 3: [7, 12, 13], 4: [6, 10], 5: [6, 10, 16], 6: [16], 8: [30, 32, 33], 9: [33], 13: [33], 14: [32, 33], 15: [32, 33], 18: [32, 33], 19: [33], 20: [32, 33], 22: [32, 33], 23: [25, 27, 29, 32, 33], 24: [25, 27, 31], 25: [31], 26: [29, 33], 27: [33], 28: [31, 33], 29: [32, 33], 30: [32, 33], 31: [32, 33], 32: [33] };
  const KE = []; for (const u in ADJ) ADJ[u].forEach(v => KE.push([+u, v]));
  const MRHI = new Set([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 16, 17, 19, 21]);
  const TRUTH2 = Array.from({ length: 34 }, (_, i) => (MRHI.has(i) ? 0 : 1));
  const TRUTH4 = [1, 1, 1, 1, 3, 3, 3, 1, 0, 1, 3, 1, 1, 1, 0, 0, 3, 1, 0, 1, 0, 1, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0];

  /* ============================================================ T4 ====== */
  Lab.register({
    id: 't4', unit: 2, num: 'T4', title: 'Graph Neural Network (GCN)',
    tag: 'Semi-supervised node classification on Zachary’s Karate Club — click to label nodes, rewire the graph, stack layers and watch each node’s receptive field grow until it covers the whole graph.',
    star: 'Paint your own labels & edges · receptive-field inspector · depth vs. neighbourhood size',
    idea: `<p>A <b>Graph Convolutional Network</b> (Kipf &amp; Welling) updates each node by averaging its neighbours’ (and its own) features with symmetric normalisation: <code>H′ = ReLU( Â · H · W )</code>, <code>Â = D<sup>−½</sup>(A+I)D<sup>−½</sup></code>. With only a handful of labelled nodes, information <i>propagates along edges</i> to label everyone else.</p>
    <p>The notebook trains once and prints an accuracy. Here <b>you choose the labelled nodes</b> (the “Label” tool), rewire edges, change the number of layers, and see the hidden embeddings (right) separate and how deeper stacks widen each node’s neighbourhood. The <b>Inspect</b> tool shows a node’s message-passing weights and its L-hop receptive field.</p>`,
    tryIt: ['Press <b>Train</b> with the default two labels (the two faction leaders, nodes 0 and 33). Colours spread through the network; accuracy on all 34 nodes is usually ≥ 90 %.',
      'Use <b>Label</b> and click nodes to cycle their class — try labelling a <i>peripheral</i> node wrongly and see how the error floods its neighbourhood.',
      'Switch to <b>4 communities</b> (the PyG label set used by the notebook) — you then need one label per community.',
      'Choose <b>Inspect</b> and raise <b>layers</b>: the receptive field grows by one hop per layer, and at 5 layers every node already sees all 34 nodes (the graph’s diameter is 5). On this tiny graph accuracy stays near 97 % up to 6 layers and dips only at ~8 — over-smoothing is a risk that grows with depth, and bites harder on larger graphs and with fewer labels.',
      '<b>Inspect</b> a node: the orange rings show its L-hop receptive field; thick edges show the Â weights 1/√(dᵤdᵥ).'],
    hood: `<pre>Â = D^-½ (A + I) D^-½                  X = I  (one-hot node ids, as PyG's KarateClub)
H¹ = ReLU(Â X W¹)   → dropout 0.3 →   Z = Â H¹ W²   → softmax
loss = CE only on the labelled nodes ;  Adam lr 0.01, weight-decay 5e-4</pre>
    <p>Gradients are derived by hand in <code>ML.GCN</code>: <code>dZ = softmax − onehot</code> on labelled rows, then <code>dW = (ÂH)ᵀ dZ</code>, <code>dH = Â (dZ Wᵀ)</code> (Â is symmetric) and the ReLU mask — verifying the same chain-rule you code in T14.</p>`,
    quiz: [
      { q: 'In a 2-layer GCN, node v’s output depends on…', o: ['only v', 'v and its direct neighbours', 'all nodes within 2 hops', 'every node in the graph'], a: 2, why: 'Each layer aggregates one hop, so two layers give a 2-hop receptive field.' },
      { q: 'Why can adding many GCN layers hurt?', o: ['too few parameters', 'over-smoothing: node representations become nearly identical', 'ReLU saturates', 'Â becomes singular'], a: 1, why: 'Repeated neighbourhood averaging is a low-pass filter that drives features toward a constant on connected graphs.' },
      { q: 'The loss in the notebook is computed on…', o: ['all nodes', 'only the labelled training nodes', 'only edges', 'only unlabelled nodes'], a: 1, why: 'Semi-supervised: supervision comes from a few labels, structure supplies the rest.' }],
    mount(root, ctx) {
      const N = 34, rng = new RNG(0);
      const P = { C: 2, L: 2, hid: 16, lr: 0.01, drop: 0.3, speed: 4, tool: 'label' };
      let edges = KE.map(e => e.slice()), labels = new Map([[0, 0], [33, 1]]), model, A, X = new Float64Array(N * N), pred = null, running = false, epoch = 0, hist = { loss: [], acc: [], accU: [] }, sel = null, edgeFrom = null, pos = 0;
      for (let i = 0; i < N; i++) X[i * N + i] = 1;
      const truth = () => (P.C === 2 ? TRUTH2 : TRUTH4);
      const nodes = Array.from({ length: N }, () => ({})); const force = new Force(nodes, edges, { w: 640, h: 500 }); force.step(250);
      function build() { A = ML.normAdj(N, edges); model = new ML.GCN(N, N, P.hid, P.C, P.L, { rng: new RNG(1), lr: P.lr, dropout: P.drop }); model.setGraph(A, X); epoch = 0; hist = { loss: [], acc: [], accU: [] }; pred = null; }
      build();
      const mask = () => Array.from({ length: N }, (_, i) => labels.has(i)), yv = () => Array.from({ length: N }, (_, i) => (labels.has(i) ? labels.get(i) : 0));
      function evalNow() { pred = model.predict(); const tr = truth(); let ok = 0, okU = 0, nU = 0; for (let i = 0; i < N; i++) { const c = pred.pred[i] === tr[i]; if (c) ok++; if (!labels.has(i)) { nU++; if (c) okU++; } } return { acc: ok / N, accU: nU ? okU / nU : NaN }; }
      function trainEpochs(k) { let l = 0; for (let i = 0; i < k; i++) { const r = model.trainStep(yv(), mask()); l = r.loss; epoch++; } const e = evalNow(); hist.loss.push([epoch, l]); hist.acc.push([epoch, e.acc * 100]); hist.accU.push([epoch, (e.accU || 0) * 100]); return { l, e }; }
      const cv = Ui.canvas(640, 500), cvE = Ui.canvas(420, 300), cvL = Ui.canvas(420, 200), lossChart = new LineChart(cvL, { xlabel: 'epoch', title: 'accuracy % (all nodes / unlabelled nodes)', yMin: 0, yMax: 100 });
      const info = h('div', { class: 'note', style: 'margin:10px 0 0;min-height:48px' }, 'Tool: Label — click a node to cycle its class (or clear it).');

      const hop = (v) => { const seen = new Map([[v, 0]]); let fr = [v]; for (let d = 1; d <= P.L; d++) { const nx = []; for (const u of fr) for (const [a, b] of edges) { const w = a === u ? b : b === u ? a : -1; if (w >= 0 && !seen.has(w)) { seen.set(w, d); nx.push(w); } } fr = nx; } return seen; };
      function draw() {
        const { g, w, h: H } = cv; g.clearRect(0, 0, w, H);
        const rf = sel !== null && P.tool === 'inspect' ? hop(sel) : null;
        for (const [u, v] of edges) {
          const a = nodes[u], b = nodes[v]; let lw = 1.2, col = '#3a342b';
          if (rf && (u === sel || v === sel)) { lw = 1 + 7 * A[u * N + v]; col = '#d9825b'; } else if (rf && rf.has(u) && rf.has(v)) col = '#6a4d4a';
          g.strokeStyle = col; g.lineWidth = lw; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
        }
        for (let i = 0; i < N; i++) {
          const n = nodes[i], lab = labels.has(i); let fill = '#2b2721';
          if (pred && epoch > 0) { const p = pred.probs[i], c = pred.pred[i]; fill = Colors.rgba(Colors.classColor(c), 0.35 + 0.65 * p[c]); }
          g.beginPath(); g.arc(n.x, n.y, 12, 0, 7); g.fillStyle = fill; g.fill();
          g.lineWidth = lab ? 4 : 1.5; g.strokeStyle = lab ? Colors.classColor(labels.get(i)) : (pred && epoch > 0 && pred.pred[i] !== truth()[i] ? '#cf6a76' : '#5c5547'); g.stroke();
          if (lab) { g.lineWidth = 1.5; g.strokeStyle = '#fff'; g.beginPath(); g.arc(n.x, n.y, 15.5, 0, 7); g.stroke(); }
          if (rf && rf.has(i)) { g.lineWidth = 2; g.strokeStyle = '#d7b56d'; g.setLineDash([3, 3]); g.beginPath(); g.arc(n.x, n.y, 19, 0, 7); g.stroke(); g.setLineDash([]); }
          g.fillStyle = '#fff'; g.font = '10px "Instrument Sans"'; g.textAlign = 'center'; g.fillText(i, n.x, n.y + 3.5);
        }
        g.textAlign = 'left'; g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; g.fillText('fill = prediction (opacity = confidence) · white ring = labelled · red outline = wrong · drag nodes to rearrange', 8, H - 8);
      }
      function drawEmb() {
        const { g, w, h: H } = cvE; g.clearRect(0, 0, w, H); g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"';
        const hid = model.hidden(); const Hm = hid ? hid.H : model.logits, dim = hid ? hid.dim : P.C;
        if (!pred || epoch === 0 || dim < 2) { g.fillText('train to see the embedding of the last hidden layer', 14, 24); return; }
        const rows = Array.from({ length: N }, (_, i) => Float64Array.from(Hm.subarray(i * dim, (i + 1) * dim)));
        const pc = new ML.PCA(2).fit(rows), Z = rows.map(r => pc.transform(r, 2));
        let mx = 1e-9; Z.forEach(z => { mx = Math.max(mx, Math.abs(z[0]), Math.abs(z[1])); });
        const spread = Math.hypot(...Z.map(z => Math.hypot(z[0], z[1]))) / Math.sqrt(N);
        const sc = Math.min(w, H) * 0.42 / mx; const tr = truth();
        for (let i = 0; i < N; i++) { const x = w / 2 + Z[i][0] * sc, y = H / 2 - Z[i][1] * sc; g.beginPath(); g.arc(x, y, 6, 0, 7); g.fillStyle = Colors.rgba(Colors.classColor(tr[i]), 0.85); g.fill(); if (labels.has(i)) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.stroke(); } }
        g.fillStyle = '#9b9384'; g.fillText('PCA of last hidden layer · colour = TRUE class · white ring = labelled', 10, 16); g.fillText('embedding spread (unnormalised): ' + spread.toExponential(1), 10, H - 8);
      }
      cvE.c.parentNode; // placeholder
      /* interaction */
      let drag = null, moved = false;
      const pick = (x, y) => { let b = -1, bd = 18; nodes.forEach((n, i) => { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; b = i; } }); return b; };
      cv.c.addEventListener('pointerdown', e => { const [x, y] = cv.xy(e), i = pick(x, y); if (i >= 0) { drag = i; moved = false; nodes[i].fixed = true; cv.c.setPointerCapture(e.pointerId); } });
      cv.c.addEventListener('pointermove', e => { if (drag === null) return; const [x, y] = cv.xy(e); nodes[drag].x = x; nodes[drag].y = y; moved = true; });
      cv.c.addEventListener('pointerup', e => {
        if (drag === null) return; const i = drag; nodes[i].fixed = false; drag = null; if (moved) return;
        if (P.tool === 'label') { const cur = labels.has(i) ? labels.get(i) : -1; if (cur + 1 >= P.C) labels.delete(i); else labels.set(i, cur + 1); evalNow(); }
        else if (P.tool === 'inspect') { sel = i; const nb = edges.filter(([a, b]) => a === i || b === i).map(([a, b]) => (a === i ? b : a)); info.innerHTML = `<b>Node ${i}</b> · degree ${nb.length} · true class ${truth()[i]}${pred && epoch ? ' · predicted ' + pred.pred[i] + ' (' + (pred.probs[i][pred.pred[i]] * 100).toFixed(0) + '%)' : ''}<br>Self-weight Â<sub>vv</sub> = ${A[i * N + i].toFixed(3)} · neighbour weights 1/√(d<sub>v</sub>d<sub>u</sub>): ${nb.slice(0, 8).map(u => u + ':' + A[i * N + u].toFixed(2)).join('  ')}${nb.length > 8 ? ' …' : ''}<br>Receptive field with ${P.L} layer(s): <b>${hop(i).size}</b> of 34 nodes.`; }
        else if (P.tool === 'edge') { if (edgeFrom === null) { edgeFrom = i; info.textContent = 'Node ' + i + ' selected — click another node to add/remove the edge.'; } else if (edgeFrom !== i) { const k = edges.findIndex(([a, b]) => (a === edgeFrom && b === i) || (a === i && b === edgeFrom)); if (k >= 0) edges.splice(k, 1); else edges.push([edgeFrom, i]); force.edges = edges; A = ML.normAdj(N, edges); model.setGraph(A, X); evalNow(); info.textContent = (k >= 0 ? 'Removed' : 'Added') + ' edge ' + edgeFrom + '–' + i + '. Keep training to adapt.'; edgeFrom = null; } else edgeFrom = null; }
      });
      const toolSeg = Ui.seg([{ v: 'label', label: 'Label' }, { v: 'edge', label: 'Edit edges' }, { v: 'inspect', label: 'Inspect' }], 'label', v => { P.tool = v; edgeFrom = null; sel = null; info.textContent = { label: 'Label: click a node to cycle its class (or clear it).', edge: 'Edit edges: click two nodes to add or remove an edge.', inspect: 'Inspect: click a node to see its message-passing weights and receptive field.' }[v]; });
      const stEp = Ui.stat('epoch'), stL = Ui.stat('loss'), stA = Ui.stat('acc (all)'), stU = Ui.stat('acc (unlabelled)'), stN = Ui.stat('labelled');
      const trainBtn = Ui.btn('Train', () => { running = !running; trainBtn.textContent = running ? 'Pause' : 'Train'; }, 'pri');
      const rebuild = () => { build(); running = false; trainBtn.textContent = 'Train'; };
      const cls = Ui.seg([{ v: 2, label: '2 factions' }, { v: 4, label: '4 communities (PyG)' }], 2, v => { P.C = v; labels = v === 2 ? new Map([[0, 0], [33, 1]]) : new Map([[0, 1], [4, 3], [8, 0], [24, 2]]); rebuild(); });
      const ctl = Ui.card('Model & training', Ui.row(trainBtn, Ui.btn('Step ×1', () => { trainEpochs(1); }), Ui.btn('Re-initialise', rebuild), Ui.btn('Clear labels', () => { labels.clear(); evalNow(); })), h('div', { style: 'margin:6px 0' }, cls),
        Ui.row(Ui.slider({ label: 'GCN layers', min: 1, max: 6, step: 1, value: 2, on: v => { P.L = v; rebuild(); } }), Ui.slider({ label: 'hidden units', min: 2, max: 32, step: 2, value: 16, on: v => { P.hid = v; rebuild(); } })),
        Ui.row(Ui.slider({ label: 'learning rate', min: 0.002, max: 0.05, step: 0.002, value: 0.01, on: v => { P.lr = v; model.opt.lr = v; } }), Ui.slider({ label: 'dropout', min: 0, max: 0.7, step: 0.1, value: 0.3, on: v => { P.drop = v; model.dropout = v; } })),
        Ui.row(stEp, stL, stA, stU, stN));
      root.appendChild(Ui.grid('g21', h('div', null, Ui.card(null, h('div', { class: 'row', style: 'margin-bottom:8px' }, toolSeg, h('span', { class: 'sp' }), Ui.btn('Re-layout', () => { force.step(200); }, 'sm')), cv.c, info)),
        h('div', null, ctl, Ui.card('Accuracy', cvL.c), Ui.card('Hidden embedding', cvE.c))));
      let f = 0;
      ctx.raf(() => {
        force.step(drag !== null || force.energy > 0.4 ? 2 : 0);
        if (running) { const r = trainEpochs(P.speed); stL.set(r.l.toFixed(3)); }
        draw(); if (f++ % 4 === 0) {
          drawEmb(); lossChart.draw([{ name: 'all nodes', color: '#86b894', pts: hist.acc }, { name: 'unlabelled', color: '#d7b56d', pts: hist.accU }], { yMin: 0, yMax: 100 });
          stEp.set(epoch); stN.set(labels.size + '/34'); if (pred && epoch) { const tr = truth(); let ok = 0, ou = 0, nu = 0; for (let i = 0; i < N; i++) { if (pred.pred[i] === tr[i]) ok++; if (!labels.has(i)) { nu++; if (pred.pred[i] === tr[i]) ou++; } } stA.set((ok / N * 100).toFixed(0) + '%'); stU.set(nu ? (ou / nu * 100).toFixed(0) + '%' : '–'); }
        }
      });
    },
  });

  /* ============================================================ T5 ====== */
  const BASES = { ex: 'http://example.org/', foaf: 'http://xmlns.com/foaf/0.1/', rdf: 'http://www.w3.org/1999/02/22-rdf-syntax-ns#', rdfs: 'http://www.w3.org/2000/01/rdf-schema#' };
  function tokenize(q) {
    q = q.replace(/(^|\s)#.*$/gm, ' '); const re = /\s*(?:(<[^>\s]*>)|("(?:[^"\\]|\\.)*")|(\?[A-Za-z_]\w*)|([A-Za-z_][\w-]*:[\w.-]*)|(!=|<=|>=|=|<|>)|(-?\d+(?:\.\d+)?)|([A-Za-z_]\w*)|([{}().;,*]))/y; const out = []; let m; re.lastIndex = 0;
    while (re.lastIndex < q.length) { const s = re.lastIndex; m = re.exec(q); if (!m) { if (/^\s*$/.test(q.slice(s))) break; throw new Error('Unexpected text near “' + q.slice(s, s + 14).trim() + '”'); } const [, iri, lit, v, pn, op, num, word, punct] = m; if (iri) out.push({ t: 'iri', v: iri.slice(1, -1) }); else if (lit) out.push({ t: 'lit', v: lit }); else if (v) out.push({ t: 'var', v }); else if (pn) out.push({ t: 'pn', v: pn }); else if (op) out.push({ t: 'op', v: op }); else if (num) out.push({ t: 'lit', v: '"' + num + '"' }); else if (word) out.push({ t: 'w', v: word }); else if (punct) out.push({ t: 'p', v: punct }); }
    return out;
  }
  function compact(iri) { for (const k in BASES) if (iri.startsWith(BASES[k])) return k + ':' + iri.slice(BASES[k].length); return '<' + iri + '>'; }
  function parseSparql(q) {
    const tk = tokenize(q); let i = 0; const peek = () => tk[i], next = () => tk[i++], isW = (t, w) => t && t.t === 'w' && t.v.toUpperCase() === w;
    const term = t => { if (!t) throw new Error('Unexpected end of query'); if (t.t === 'var') return t.v; if (t.t === 'iri') return compact(t.v); if (t.t === 'lit') return t.v; if (t.t === 'pn') return t.v; if (t.t === 'w' && t.v === 'a') return 'rdf:type'; throw new Error('Bad term “' + t.v + '”'); };
    while (isW(peek(), 'PREFIX')) { next(); next(); next(); }
    if (!isW(next(), 'SELECT')) throw new Error('Expected SELECT'); const sel = []; let distinct = false; if (isW(peek(), 'DISTINCT')) { next(); distinct = true; }
    while (peek() && !isW(peek(), 'WHERE') && !(peek().t === 'p' && peek().v === '{')) {
      const t = next(); if (t.t === 'var') sel.push({ v: t.v }); else if (t.t === 'p' && t.v === '*') sel.push({ star: true }); else if (t.t === 'p' && t.v === '(') {
        const agg = next(); next(); let dist = false; if (isW(peek(), 'DISTINCT')) { next(); dist = true; } const arg = next(); next(); if (!isW(next(), 'AS')) throw new Error('Expected AS'); const alias = next().v; next(); sel.push({ agg: agg.v.toUpperCase(), arg: arg.t === 'p' ? '*' : arg.v, alias, dist });
      } else throw new Error('Unexpected “' + t.v + '” in SELECT');
    }
    if (isW(peek(), 'WHERE')) next(); if (!(peek() && peek().v === '{')) throw new Error('Expected {'); next();
    const pats = [], filters = []; let s = null;
    while (peek() && !(peek().t === 'p' && peek().v === '}')) {
      if (isW(peek(), 'FILTER')) { next(); next(); const l = next(), op = next(), r = next(); next(); filters.push({ l: term(l), op: op.v, r: term(r) }); if (peek() && peek().v === '.') next(); continue; }
      s = term(next()); let p = term(next()); let o = term(next()); pats.push([s, p, o]);
      while (peek() && (peek().v === ';' || peek().v === ',')) { const sep = next().v; if (sep === ';') { p = term(next()); } o = term(next()); pats.push([s, p, o]); }
      if (peek() && peek().v === '.') next();
    }
    next(); let groupBy = [], order = null, limit = null;
    while (peek()) {
      if (isW(peek(), 'GROUP')) { next(); next(); while (peek() && peek().t === 'var') groupBy.push(next().v); }
      else if (isW(peek(), 'ORDER')) { next(); next(); let desc = false; let t = next(); if (isW(t, 'DESC') || isW(t, 'ASC')) { desc = t.v.toUpperCase() === 'DESC'; next(); t = next(); next(); } order = { v: t.v, desc }; }
      else if (isW(peek(), 'LIMIT')) { next(); limit = parseInt(next().v.replace(/"/g, ''), 10); } else throw new Error('Unexpected “' + peek().v + '”');
    }
    return { sel, distinct, pats, filters, groupBy, order, limit };
  }
  const isVar = x => x[0] === '?', disp = x => (x[0] === '"' ? x.slice(1, -1) : x);
  function runSparql(q, triples) {
    const Q = parseSparql(q); let sols = [{ b: {}, used: [] }];
    for (const [s, p, o] of Q.pats) { const nx = []; for (const sol of sols) triples.forEach((t, ti) => { const b = Object.assign({}, sol.b); let ok = true; [[s, t[0]], [p, t[1]], [o, t[2]]].forEach(([pt, tv]) => { if (!ok) return; if (isVar(pt)) { if (b[pt] === undefined) b[pt] = tv; else if (b[pt] !== tv) ok = false; } else if (pt !== tv) ok = false; }); if (ok) nx.push({ b, used: sol.used.concat(ti) }); }); sols = nx; }
    const val = (x, b) => (isVar(x) ? b[x] : x);
    for (const f of Q.filters) sols = sols.filter(s => { const l = val(f.l, s.b), r = val(f.r, s.b); return f.op === '!=' ? l !== r : f.op === '=' ? l === r : f.op === '<' ? +disp(l) < +disp(r) : f.op === '>' ? +disp(l) > +disp(r) : true; });
    const used = new Set(); sols.forEach(s => s.used.forEach(u => used.add(u))); const boundVars = new Set(); sols.forEach(s => Object.keys(s.b).forEach(k => boundVars.add(k)));
    let cols, rows; const hasAgg = Q.sel.some(x => x.agg);
    if (hasAgg) {
      const groups = new Map(); for (const s of sols) { const key = Q.groupBy.map(v => s.b[v]).join('\u0001'); if (!groups.has(key)) groups.set(key, []); groups.get(key).push(s); }
      if (!Q.groupBy.length && !groups.size) groups.set('', []);
      cols = Q.sel.map(x => x.alias || x.v); rows = [...groups.values()].map(g => Q.sel.map(x => { if (!x.agg) return g[0].b[x.v]; const vs = x.arg === '*' ? g.map((s, i) => i) : g.map(s => s.b[x.arg]); return String(x.dist ? new Set(vs).size : vs.length); }));
    } else { const vars = Q.sel.some(x => x.star) ? [...Q.pats.flat().filter(isVar)].filter((v, i, a) => a.indexOf(v) === i) : Q.sel.map(x => x.v); cols = vars; rows = sols.map(s => vars.map(v => s.b[v] ?? '')); }
    if (Q.distinct) { const seen = new Set(); rows = rows.filter(r => { const k = r.join('\u0001'); if (seen.has(k)) return false; seen.add(k); return true; }); }
    if (Q.order) { const ci = cols.indexOf(Q.order.v.replace('?', '').length ? Q.order.v : Q.order.v); const k = ci >= 0 ? ci : cols.indexOf(Q.order.v); rows.sort((a, b) => { const x = disp(a[k] || ''), y = disp(b[k] || ''); const c = (!isNaN(+x) && !isNaN(+y)) ? +x - +y : String(x).localeCompare(String(y)); return Q.order.desc ? -c : c; }); }
    if (Q.limit != null) rows = rows.slice(0, Q.limit);
    return { cols, rows, used, bound: boundVars, nSol: sols.length };
  }
  const CAMPUS = (() => { const t = [], p = { Raktim: ['SRM', 'Chennai', ['Python', 'ML']], Juhi: ['SRM', 'Chennai', ['Python', 'Research']], Meera: ['SRM', 'Chennai', ['ML', 'Statistics']], Arjun: ['Acme', 'Pune', ['Java', 'DevOps']], Alice: ['Acme', 'Pune', ['Java']], Bob: ['Acme', 'Pune', ['Python', 'DevOps']], Chen: ['Globex', 'Singapore', ['Python', 'ML', 'Research']], Dana: ['Globex', 'Singapore', ['Statistics']] }; for (const n in p) { t.push(['ex:' + n, 'rdf:type', 'foaf:Person'], ['ex:' + n, 'foaf:name', '"' + n + '"'], ['ex:' + n, 'ex:worksAt', 'ex:' + p[n][0]]); p[n][2].forEach(s => t.push(['ex:' + n, 'ex:hasSkill', '"' + s + '"'])); } [['SRM', 'Chennai'], ['Acme', 'Pune'], ['Globex', 'Singapore']].forEach(([o, c]) => { t.push(['ex:' + o, 'rdf:type', 'ex:Organization'], ['ex:' + o, 'ex:locatedIn', 'ex:' + c]); }); t.push(['ex:Raktim', 'foaf:knows', 'ex:Juhi'], ['ex:Juhi', 'foaf:knows', 'ex:Meera'], ['ex:Bob', 'foaf:knows', 'ex:Alice'], ['ex:Chen', 'foaf:knows', 'ex:Raktim']); return t; })();
  const NOTEBOOK = (() => { const t = []; const p = { Raktim: ['SRM', ['Python', 'ML']], Juhi: ['SRM', ['Python', 'Research']], Alice: ['Acme', ['Java']], Bob: ['Acme', ['Python', 'DevOps']] }; for (const n in p) { t.push(['ex:' + n, 'rdf:type', 'foaf:Person'], ['ex:' + n, 'foaf:name', '"' + n + '"'], ['ex:' + n, 'ex:worksAt', 'ex:' + p[n][0]]); p[n][1].forEach(s => t.push(['ex:' + n, 'ex:hasSkill', '"' + s + '"'])); } ['SRM', 'Acme'].forEach(o => t.push(['ex:' + o, 'rdf:type', 'ex:Organization'])); return t; })();

  Lab.register({
    id: 't5', unit: 2, num: 'T5', title: 'Knowledge graphs & SPARQL',
    tag: 'Build an RDF graph, query it with a real SPARQL engine (written from scratch), see the matches light up — then learn embeddings (TransE) to predict missing facts.',
    star: 'Matches highlight in the graph · ask in plain English · TransE link prediction with held-out facts',
    idea: `<p>A <b>knowledge graph</b> stores facts as <i>(subject, predicate, object)</i> triples. <b>SPARQL</b> queries are graph patterns: variables (<code>?x</code>) are bound by joining triple patterns. The notebook uses rdflib; here the whole stack — tokenizer, parser, join engine, <code>FILTER</code>, <code>COUNT</code>, <code>GROUP BY</code>, <code>ORDER BY</code>, <code>LIMIT</code> — is implemented in JavaScript, so you can <b>see which triples each query touches</b> in the graph.</p>
    <p>The syllabus also covers <b>knowledge-graph embeddings and reasoning</b>. The second tab trains <b>TransE</b> (<code>h + r ≈ t</code>) on the triples so the model can <i>rank</i> candidate answers for facts it was never told, and you can hold a fact out to test it.</p>`,
    tryIt: ['Pick the example queries and press <b>Run</b>: the orange nodes/edges are exactly the triples the engine matched.',
      'Write your own: <code>SELECT ?p WHERE { ?p ex:hasSkill "ML" . }</code> — or add a new triple on the right and re-run.',
      'Use <b>Ask in English</b> (“who works at SRM and knows Python?”) — it shows the SPARQL it generated.',
      'Switch to the <b>Campus KG</b> (bigger), then open <b>Embeddings</b>: train TransE, ask <code>(Meera, worksAt, ?)</code>. Then hold out that triple, retrain, and see where the true answer ranks (tiny KGs are hard — that is the point).'],
    hood: `<pre>SELECT ?name WHERE { ?p foaf:name ?name . ?p ex:worksAt ex:SRM . ?p ex:hasSkill "Python" }
 → start with one empty binding, then for each triple pattern extend every binding with every
   triple that unifies (a nested-loop join), apply FILTERs, aggregate, sort.

TransE: score(h,r,t) = −‖h + r − t‖₂        loss = max(0, γ + d(h,r,t) − d(h',r,t'))
 negatives (h' or t') are sampled by corrupting a true triple; entity vectors are re-normalised each step.</pre>`,
    quiz: [
      { q: 'In SPARQL, what does the pattern { ?p ex:worksAt ex:SRM . ?p ex:hasSkill "Python" } do?', o: ['unions two result sets', 'joins on the shared variable ?p', 'deletes triples', 'creates two people'], a: 1, why: 'Triple patterns sharing a variable are joined: ?p must satisfy both.' },
      { q: 'TransE models a relation as…', o: ['a rotation', 'a translation vector in embedding space', 'a probability table', 'a matrix product'], a: 1, why: 'It wants h + r ≈ t, i.e. relation r translates the head embedding to the tail embedding.' },
      { q: 'Why add negative sampling?', o: ['to speed up SPARQL', 'so true triples score better than corrupted ones', 'to delete facts', 'to normalise vectors'], a: 1, why: 'The margin loss pushes true triples to be closer than corrupted triples; without negatives all embeddings could collapse.' }],
    mount(root, ctx) {
      let which = 'campus', triples = CAMPUS.map(t => t.slice()), result = null, W = 700, Hh = 520, nodes = [], idxOf = new Map(), nodeEdges = [], force = null;
      const examples = {
        'Who works at SRM and knows Python? (notebook q1)': `PREFIX ex:   <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?name WHERE {\n  ?p foaf:name ?name .\n  ?p ex:worksAt ex:SRM .\n  ?p ex:hasSkill "Python" .\n}`,
        'Head-count per organisation (notebook q2)': `PREFIX ex:   <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?org (COUNT(?p) AS ?n) WHERE {\n  ?p ex:worksAt ?org .\n  ?p a foaf:Person .\n} GROUP BY ?org ORDER BY DESC(?n)`,
        'Colleagues who share a skill': `PREFIX ex: <http://example.org/>\nSELECT DISTINCT ?a ?b ?skill WHERE {\n  ?a ex:worksAt ?org .\n  ?b ex:worksAt ?org .\n  ?a ex:hasSkill ?skill .\n  ?b ex:hasSkill ?skill .\n  FILTER(?a != ?b)\n} ORDER BY ?skill`,
        'Which skills are most common?': `PREFIX ex: <http://example.org/>\nSELECT ?skill (COUNT(?p) AS ?people) WHERE {\n  ?p ex:hasSkill ?skill .\n} GROUP BY ?skill ORDER BY DESC(?people)`,
        'Who knows someone in another city? (campus KG)': `PREFIX ex:   <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?a ?b WHERE {\n  ?a foaf:knows ?b .\n  ?a ex:worksAt ?o1 .\n  ?b ex:worksAt ?o2 .\n  ?o1 ex:locatedIn ?c1 .\n  ?o2 ex:locatedIn ?c2 .\n  FILTER(?c1 != ?c2)\n}`,
      };
      const ta = h('textarea', { rows: 11, spellcheck: false }); ta.value = examples['Who works at SRM and knows Python? (notebook q1)'];
      const out = h('div'), err = h('div', { class: 'bad', style: 'font-size:13px;margin-top:6px;white-space:pre-wrap' });
      const cv = Ui.canvas(700, 520);
      function buildGraph() {
        const ents = []; idxOf = new Map(); const get = t => { if (!idxOf.has(t)) { idxOf.set(t, ents.length); ents.push(t); } return idxOf.get(t); };
        nodeEdges = []; triples.forEach((t, ti) => { if (t[1] === 'rdf:type') return; nodeEdges.push([get(t[0]), get(t[2]), t[1], ti]); });
        triples.forEach(t => { if (t[1] === 'rdf:type') get(t[0]); });
        const types = new Map(); triples.forEach(t => { if (t[1] === 'rdf:type') types.set(t[0], t[2]); });
        const old = new Map(nodes.map(n => [n.id, n])); nodes = ents.map(e => { const o = old.get(e); return o ? Object.assign(o, { type: types.get(e) }) : { id: e, type: types.get(e), lit: e[0] === '"' }; });
        force = new Force(nodes, nodeEdges.map(e => [e[0], e[1]]), { w: W, h: Hh, k: 112 }); force.step(450);
      }
      buildGraph();
      function drawKG() {
        const { g, w, h: H } = cv; g.clearRect(0, 0, w, H); const used = result ? result.used : new Set(), bound = result ? result.rowsTerms : new Set();
        g.font = '10px "Instrument Sans"';
        nodeEdges.forEach(([u, v, p, ti]) => { const a = nodes[u], b = nodes[v], hot = used.has(ti); g.strokeStyle = hot ? '#d9825b' : '#3a342b'; g.lineWidth = hot ? 3 : 1.2; g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke(); const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2; g.fillStyle = hot ? '#f0cdb8' : '#6f6859'; g.textAlign = 'center'; g.fillText(p.replace('ex:', ''), mx, my - 2); });
        nodes.forEach(n => { const hot = bound.has(n.id); const isOrg = n.type === 'ex:Organization', isP = n.type === 'foaf:Person'; const r = n.lit ? 7 : 12; g.beginPath(); if (n.lit) g.rect(n.x - 22, n.y - 8, 44, 16); else g.arc(n.x, n.y, r, 0, 7); g.fillStyle = n.lit ? '#2c2822' : isOrg ? '#7e62a0' : isP ? '#4f8f86' : '#4a4338'; g.fill(); g.lineWidth = hot ? 3 : 1; g.strokeStyle = hot ? '#d7b56d' : '#6f6859'; g.stroke(); g.fillStyle = '#ece6d8'; g.textAlign = 'center'; g.font = (n.lit ? '9.5px' : '10.5px') + ' "Instrument Sans"'; g.fillText(disp(n.id).replace('ex:', '').replace('foaf:', ''), n.x, n.lit ? n.y + 3.5 : n.y + r + 11); });
        g.fillStyle = '#9b9384'; g.textAlign = 'left'; g.font = '11px "Instrument Sans"'; g.fillText(`${triples.length} triples · ${nodes.length} nodes · ● person ● organisation box = literal · orange = matched by the last query (drag nodes)`, 8, H - 8);
      }
      let drag = null; cv.c.addEventListener('pointerdown', e => { const [x, y] = cv.xy(e); let b = -1, bd = 22; nodes.forEach((n, i) => { const d = Math.hypot(n.x - x, n.y - y); if (d < bd) { bd = d; b = i; } }); if (b >= 0) { drag = b; nodes[b].fixed = true; cv.c.setPointerCapture(e.pointerId); } });
      cv.c.addEventListener('pointermove', e => { if (drag === null) return; const [x, y] = cv.xy(e); nodes[drag].x = x; nodes[drag].y = y; }); cv.c.addEventListener('pointerup', () => { if (drag !== null) { nodes[drag].fixed = false; drag = null; } });
      function run(q) {
        err.textContent = ''; try {
          result = runSparql(q, triples); result.rowsTerms = new Set(); result.rows.forEach(r => r.forEach(c => c && result.rowsTerms.add(c))); result.bound.forEach(() => { });
          // also highlight every bound term from matched triples
          result.used.forEach(ti => { result.rowsTerms.add(triples[ti][0]); result.rowsTerms.add(triples[ti][2]); });
          out.innerHTML = ''; out.appendChild(h('div', { class: 'muted', style: 'font-size:12.5px;margin-bottom:6px' }, `${result.rows.length} row(s) · ${result.used.size} triple(s) matched`));
          out.appendChild(Ui.table(result.cols.map(c => c.replace('?', '')), result.rows.map(r => r.map(c => disp(String(c)).replace(/^ex:|^foaf:/, ''))), { maxH: '240px' }));
        } catch (e) { err.textContent = 'Error: ' + e.message; }
      }
      /* English -> SPARQL */
      const eng = h('input', { type: 'text', placeholder: 'e.g. who works at SRM and knows Python?', style: 'flex:1' });
      function askEnglish() {
        const s = eng.value.toLowerCase(); const orgs = triples.filter(t => t[1] === 'rdf:type' && t[2] === 'ex:Organization').map(t => t[0]), skills = [...new Set(triples.filter(t => t[1] === 'ex:hasSkill').map(t => t[2]))];
        const org = orgs.find(o => s.includes(o.replace('ex:', '').toLowerCase())), skill = skills.find(k => s.includes(disp(k).toLowerCase())); const count = /how many|number of|count/.test(s);
        let q; const pre = 'PREFIX ex: <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\n';
        if (count && org) q = pre + `SELECT (COUNT(?p) AS ?n) WHERE {\n  ?p ex:worksAt ${org} .\n}`; else if (count && skill) q = pre + `SELECT (COUNT(?p) AS ?n) WHERE {\n  ?p ex:hasSkill ${skill} .\n}`; else if (org && skill) q = pre + `SELECT ?name WHERE {\n  ?p foaf:name ?name .\n  ?p ex:worksAt ${org} .\n  ?p ex:hasSkill ${skill} .\n}`; else if (org) q = pre + `SELECT ?name WHERE {\n  ?p foaf:name ?name .\n  ?p ex:worksAt ${org} .\n}`; else if (skill) q = pre + `SELECT ?name WHERE {\n  ?p foaf:name ?name .\n  ?p ex:hasSkill ${skill} .\n}`; else { Ui.toast('Mention an organisation (' + orgs.map(o => o.replace('ex:', '')).join(', ') + ') and/or a skill (' + skills.slice(0, 5).map(disp).join(', ') + '…)'); return; }
        ta.value = q; run(q);
      }
      const sel = Ui.select('Example queries', Object.keys(examples), Object.keys(examples)[0], v => { ta.value = examples[v]; });
      const dsSeg = Ui.seg([{ v: 'campus', label: 'Campus KG (bigger)' }, { v: 'notebook', label: 'Notebook KG' }], 'campus', v => { which = v; triples = (v === 'campus' ? CAMPUS : NOTEBOOK).map(t => t.slice()); result = null; out.innerHTML = ''; buildGraph(); refreshTurtle(); resetTE(); });
      const turtle = h('pre', { style: 'max-height:230px' });
      function refreshTurtle() { const by = new Map(); triples.forEach(t => { if (!by.has(t[0])) by.set(t[0], []); by.get(t[0]).push(t); }); turtle.textContent = '@prefix ex: <http://example.org/> .\n@prefix foaf: <http://xmlns.com/foaf/0.1/> .\n\n' + [...by].map(([s, ts]) => s + '\n' + ts.map(t => '    ' + (t[1] === 'rdf:type' ? 'a' : t[1]) + ' ' + t[2]).join(' ;\n') + ' .').join('\n\n'); }
      refreshTurtle();
      const ns = h('input', { type: 'text', placeholder: 'subject  ex:Juhi', style: 'width:32%' }), np = h('input', { type: 'text', placeholder: 'predicate  ex:hasSkill', style: 'width:32%' }), no = h('input', { type: 'text', placeholder: 'object  "Go" or ex:SRM', style: 'width:32%' });
      const addT = () => { const s = ns.value.trim(), p = np.value.trim(), o = no.value.trim(); if (!s || !p || !o) return Ui.toast('Fill subject, predicate, object'); triples.push([s, p === 'a' ? 'rdf:type' : p, o]); buildGraph(); refreshTurtle(); resetTE(); Ui.toast('Triple added'); };

      /* ---------- TransE ---------- */
      const te = { dim: 12, ents: [], rels: [], E: null, R: null, loss: [], epoch: 0, held: -1, rng: new RNG(5), run: false };
      const cvT = Ui.canvas(520, 330), cvTL = Ui.canvas(520, 150), tl = new LineChart(cvTL, { xlabel: 'epoch', title: 'TransE margin loss' });
      function resetTE() {
        const tr = triples.filter(t => t[1] !== 'rdf:type' || true); te.ents = [...new Set(tr.flatMap(t => [t[0], t[2]]))]; te.rels = [...new Set(tr.map(t => t[1]))];
        te.E = te.ents.map(() => Float64Array.from({ length: te.dim }, () => te.rng.normal() / Math.sqrt(te.dim))); te.R = te.rels.map(() => Float64Array.from({ length: te.dim }, () => te.rng.normal() / Math.sqrt(te.dim))); te.E.forEach(normV); te.loss = []; te.epoch = 0; te.run = false; te.held = -1; fillTE();
      }
      const normV = v => { const n = Math.hypot(...v) || 1; for (let i = 0; i < v.length; i++) v[i] /= n; };
      const dist = (h1, r, t1) => { let s = 0; for (let i = 0; i < te.dim; i++) { const d = te.E[h1][i] + te.R[r][i] - te.E[t1][i]; s += d * d; } return Math.sqrt(s); };
      function trainTE(epochs) {
        const tr = triples.map((t, i) => [i, te.ents.indexOf(t[0]), te.rels.indexOf(t[1]), te.ents.indexOf(t[2])]).filter(t => t[0] !== te.held), lr = 0.05, gam = 1.0; let tot = 0;
        for (let e = 0; e < epochs; e++) {
          te.rng.shuffle(tr); tot = 0;
          for (const [, hh, r, t] of tr) {
            let h2 = hh, t2 = t; if (te.rng.rand() < 0.5) h2 = te.rng.int(te.ents.length); else t2 = te.rng.int(te.ents.length); if (h2 === hh && t2 === t) continue;
            const dp = dist(hh, r, t), dn = dist(h2, r, t2), l = gam + dp - dn; if (l <= 0) continue; tot += l;
            for (let i = 0; i < te.dim; i++) {
              const gp = (te.E[hh][i] + te.R[r][i] - te.E[t][i]) / (dp + 1e-9), gn = (te.E[h2][i] + te.R[r][i] - te.E[t2][i]) / (dn + 1e-9);
              te.E[hh][i] -= lr * gp; te.E[t][i] += lr * gp; te.R[r][i] -= lr * (gp - gn); te.E[h2][i] += lr * gn; te.E[t2][i] -= lr * gn;
            }
            normV(te.E[hh]); normV(te.E[t]); normV(te.E[h2]); normV(te.E[t2]);
          }
          te.epoch++; te.loss.push([te.epoch, tot / Math.max(1, tr.length)]);
        }
      }
      const hSel = h('select'), rSel = h('select'), holdSel = h('select'); const rank = h('div');
      function fillTE() { const keep = [hSel.value, rSel.value]; hSel.innerHTML = ''; te.ents.filter(e => e[0] !== '"').forEach(e => hSel.appendChild(h('option', { value: e }, e))); rSel.innerHTML = ''; te.rels.forEach(e => rSel.appendChild(h('option', { value: e }, e))); holdSel.innerHTML = ''; holdSel.appendChild(h('option', { value: -1 }, '— none —')); triples.forEach((t, i) => { if (t[1] !== 'rdf:type') holdSel.appendChild(h('option', { value: i }, t.join('  '))); }); if (keep[0]) hSel.value = keep[0]; if (keep[1]) rSel.value = keep[1]; }
      function predict() {
        const hh = te.ents.indexOf(hSel.value), r = te.rels.indexOf(rSel.value); if (hh < 0 || r < 0) return; const known = new Set(triples.map(t => t.join('|')));
        const sc = te.ents.map((e, i) => ({ e, d: dist(hh, r, i), known: known.has([hSel.value, rSel.value, e].join('|')), held: te.held >= 0 && triples[te.held].join('|') === [hSel.value, rSel.value, e].join('|') })).filter(x => x.e !== hSel.value).sort((a, b) => a.d - b.d);
        rank.innerHTML = ''; rank.appendChild(Ui.table(['rank', 'candidate tail', 'distance', 'status'], sc.slice(0, 10).map((x, i) => [i + 1, disp(x.e), x.d.toFixed(3), x.held ? h('span', { class: 'tag warn' }, 'HELD-OUT truth') : x.known ? h('span', { class: 'tag' }, 'known fact') : ''])));
        if (te.held >= 0) { const t = triples[te.held]; const all = te.ents.map((e, i) => ({ e, d: dist(te.ents.indexOf(t[0]), te.rels.indexOf(t[1]), i) })).filter(x => x.e !== t[0]).sort((a, b) => a.d - b.d); const rk = all.findIndex(x => x.e === t[2]) + 1; rank.prepend(h('div', { class: 'note' }, `Held-out fact (${t.join(' ')}) ranks #${rk} of ${all.length} candidates.`)); }
      }
      function drawTE() {
        const { g, w, H = cvT.h } = cvT; g.clearRect(0, 0, w, H); if (!te.epoch) { g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.fillText('train TransE to see entity embeddings (PCA 2-D)', 14, 24); return; }
        const pc = new ML.PCA(2).fit(te.E), Z = te.E.map(e => pc.transform(e, 2)); let mx = 1e-9; Z.forEach(z => { mx = Math.max(mx, Math.abs(z[0]), Math.abs(z[1])); }); const sc = Math.min(w / 2, H / 2) * 0.82 / mx;
        const types = new Map(); triples.forEach(t => { if (t[1] === 'rdf:type') types.set(t[0], t[2]); });
        te.ents.forEach((e, i) => { const x = w / 2 + Z[i][0] * sc, y = H / 2 - Z[i][1] * sc; g.fillStyle = e[0] === '"' ? '#6f6859' : types.get(e) === 'foaf:Person' ? '#4f8f86' : types.get(e) === 'ex:Organization' ? '#a78bc4' : '#d7b56d'; g.beginPath(); g.arc(x, y, 5, 0, 7); g.fill(); g.fillStyle = '#cfc8b8'; g.font = '10px "Instrument Sans"'; g.textAlign = 'left'; g.fillText(disp(e).replace('ex:', ''), x + 7, y + 3); });
      }
      const teBtn = Ui.btn('Train TransE', () => { te.run = !te.run; teBtn.textContent = te.run ? 'Pause' : 'Train TransE'; }, 'pri');
      resetTE();
      const kgPane = h('div'), tePane = h('div', { style: 'display:none' });
      kgPane.appendChild(Ui.grid('g2', Ui.card('Knowledge graph', cv.c),
        h('div', null, Ui.card('SPARQL', Ui.row(sel), ta, h('div', { class: 'row', style: 'margin-top:8px' }, Ui.btn('Run query', () => run(ta.value), 'pri'), dsSeg), err, out),
          Ui.card('Ask in English → SPARQL', h('div', { class: 'row' }, eng, Ui.btn('Ask', askEnglish))),
          Ui.card('Add a triple', h('div', { class: 'row' }, ns, np, no), h('div', { style: 'margin-top:8px' }, Ui.btn('Add', addT))), Ui.card('Turtle serialisation', turtle))));
      tePane.appendChild(Ui.grid('g2', h('div', null, Ui.card('Entity embeddings (PCA of TransE vectors)', cvT.c), Ui.card(null, cvTL.c)),
        h('div', null, Ui.card('Train', Ui.row(teBtn, Ui.btn('Re-init', resetTE)), Ui.row(Ui.slider({ label: 'embedding dim', min: 4, max: 32, step: 4, value: te.dim, on: v => { te.dim = v; resetTE(); } })), h('div', { style: 'margin-top:8px' }, h('label', { class: 'muted', style: 'font-size:12px' }, 'Hold out a fact (then Re-init + train):'), holdSel, Ui.btn('Hold out', () => { resetTE(); te.held = +holdSel.value; holdSel.value = te.held; Ui.toast(te.held >= 0 ? 'Fact hidden from training — train now' : 'No fact held out'); }, 'sm'))),
          Ui.card('Link prediction  (head, relation, ?)', Ui.row(hSel, rSel, Ui.btn('Predict tails', predict, 'pri')), rank))));
      const tabs = Ui.seg([{ v: 'kg', label: 'SPARQL on the graph' }, { v: 'te', label: 'Embeddings (TransE)' }], 'kg', v => { kgPane.style.display = v === 'kg' ? '' : 'none'; tePane.style.display = v === 'te' ? '' : 'none'; });
      root.appendChild(h('div', { style: 'margin-bottom:14px' }, tabs)); root.appendChild(kgPane); root.appendChild(tePane);
      run(ta.value);
      let f = 0;
      ctx.raf(() => {
        if (kgPane.style.display !== 'none') { force.step(drag !== null || force.energy > 0.3 ? 2 : 0); drawKG(); }
        else { if (te.run) { trainTE(5); } if (f++ % 3 === 0) { drawTE(); tl.draw([{ name: 'loss', color: '#d7b56d', pts: te.loss }]); } }
      });
    },
  });

  /* ============================================================ T6 ====== */
  const LBL = { C: '#a8a08e', O: '#d9825b', H: '#ece6d8', N: '#7aa2c8', S: '#d7b56d' };
  const hstr = s => { let x = 5381; for (let i = 0; i < s.length; i++) x = ((x * 33) ^ s.charCodeAt(i)) >>> 0; return x.toString(36); };
  function wlSig(labels, nbrs, iters = 3) { let lab = labels.slice(); const all = lab.slice(); for (let it = 0; it < iters; it++) { lab = lab.map((l, i) => hstr(l + '|' + nbrs[i].map(j => lab[j]).sort().join(','))); all.push(...lab); } return all.sort().join(''); }
  function wlColors(g, iters) { let lab = g.labels.slice(); for (let it = 0; it < iters; it++) lab = lab.map((l, i) => hstr(l + '|' + g.adj[i].map(j => lab[j]).sort().join(','))); return lab; }
  function mkGraph(edges, labels) { const adj = labels.map(() => []); edges.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); }); return { edges, labels, adj }; }
  function connectedSubsets(g, size) { const n = g.labels.length, out = []; const ext = (cur, ex, root, closed) => { if (cur.length === size) { out.push(cur.slice()); return; } ex = ex.slice(); while (ex.length) { const w = ex.pop(), nx = ex.slice(), nc = new Set(closed); for (const u of g.adj[w]) { if (u > root && !closed.has(u)) nx.push(u); nc.add(u); } nc.add(w); ext([...cur, w], nx, root, nc); } }; for (let s = 0; s < n; s++) ext([s], g.adj[s].filter(u => u > s), s, new Set([s, ...g.adj[s]])); return out; }
  function subSig(g, nodes) { const idx = new Map(nodes.map((v, i) => [v, i])), lab = nodes.map(v => g.labels[v]), nb = nodes.map(v => g.adj[v].filter(u => idx.has(u)).map(u => idx.get(u))); return wlSig(lab, nb, 3); }
  function mineFSM(db, minSup, maxSize) { const sup = new Map(), ex = new Map(), occ = new Map(); db.forEach((g, gi) => { for (let size = 2; size <= maxSize; size++) for (const nodes of connectedSubsets(g, size)) { const sig = subSig(g, nodes); if (!sup.has(sig)) { sup.set(sig, new Set()); occ.set(sig, db.map(() => [])); } sup.get(sig).add(gi); occ.get(sig)[gi].push(nodes); if (!ex.has(sig)) ex.set(sig, { g, nodes, size }); } }); return [...sup].filter(([, s]) => s.size >= minSup).map(([sig, s]) => ({ sig, support: s.size, ex: ex.get(sig), occ: occ.get(sig) })).sort((a, b) => b.support - a.support || a.ex.size - b.ex.size); }
  function layoutGraph(g, w, hh) { const nodes = g.labels.map(() => ({})); const f = new Force(nodes, g.edges, { w, h: hh, k: Math.min(w, hh) / (Math.sqrt(g.labels.length) + 0.8) }); f.step(200); return nodes.map(n => ({ x: n.x, y: n.y })); }
  function drawSvgGraph(g, pos, w, hh, opts = {}) {
    const s = svg('svg', { viewBox: `0 0 ${w} ${hh}`, width: '100%', style: 'display:block' }); const hl = opts.hl || new Set(), cols = opts.colors;
    g.edges.forEach(([a, b]) => s.appendChild(svg('line', { x1: pos[a].x, y1: pos[a].y, x2: pos[b].x, y2: pos[b].y, stroke: hl.has(a) && hl.has(b) ? '#d7b56d' : '#4a4338', 'stroke-width': hl.has(a) && hl.has(b) ? 4 : 2 })));
    g.labels.forEach((l, i) => { const fill = cols ? `hsl(${parseInt(cols[i].slice(0, 4), 36) % 360},65%,58%)` : LBL[l] || '#a8a08e'; s.appendChild(svg('circle', { cx: pos[i].x, cy: pos[i].y, r: 11, fill, stroke: hl.has(i) ? '#d7b56d' : '#100e0c', 'stroke-width': hl.has(i) ? 3.5 : 1.5 })); s.appendChild(svg('text', { x: pos[i].x, y: pos[i].y + 4, 'text-anchor': 'middle', 'font-size': 11, fill: '#14110d', 'font-weight': 700 }, l)); });
    return s;
  }
  Lab.register({
    id: 't6', unit: 2, num: 'T6', title: 'Frequent subgraph mining',
    tag: 'Find the structural motifs that recur across a database of molecule-like graphs, canonicalised with Weisfeiler–Lehman — and watch WL colour refinement run.',
    star: 'Click a pattern to highlight every occurrence · WL refinement animator · isomorphism-invariance test',
    idea: `<p><b>Frequent subgraph mining</b> asks: which connected labelled subgraphs occur in at least <i>min-support</i> graphs of a database (think recurring functional groups in molecules)? The notebook enumerates connected node subsets, canonicalises each with the <b>Weisfeiler–Lehman (WL) hash</b> and counts support. WL colour refinement is also exactly the expressive power of standard message-passing GNNs (T4), which is why this task sits in the “graphs” unit.</p>
    <p>Here you can change the support threshold and size range, generate random molecule databases, click a mined pattern to <b>highlight each occurrence</b>, shuffle node ids to prove the hash is isomorphism-invariant, and step through <b>WL refinement</b> to see nodes get distinguished by their neighbourhoods.</p>`,
    tryIt: ['Lower <b>min support</b> to 1: many more patterns appear. Raise it to the number of graphs: only the universal motif(s) survive. Support never increases when a pattern grows (anti-monotone) — the same downward-closure idea as Apriori in Unit 4.',
      'Click a row in the pattern table: occurrences are outlined in gold in every graph.',
      'In the WL lab raise the iteration slider: nodes with identical neighbourhood structure keep the same colour; others split.',
      'Press <b>Relabel &amp; rehash</b>: the node ids are randomly permuted and the WL signature stays identical (isomorphism-invariant). WL cannot separate some non-isomorphic regular graphs — a known GNN limitation.'],
    hood: `<pre>for each graph G, for each size k in [2..max]:
    for each connected node subset S of size k:     sig = WL(G[S], labels)
        support[sig].add(graph id)
frequent = { sig : |support[sig]| ≥ min_support }

WL(G): colour_0(v) = label(v);  colour_{t+1}(v) = hash( colour_t(v) , sorted{ colour_t(u) : u ∈ N(v) } )
signature = sorted multiset of all colours over t = 0..3</pre>
    <p>The notebook uses <code>networkx.weisfeiler_lehman_graph_hash</code>; this page implements the same refinement directly. Connected subsets are enumerated without duplicates by rooting each at its smallest node.</p>`,
    quiz: [
      { q: 'What does “support” of a subgraph pattern count?', o: ['occurrences inside one graph', 'the number of graphs in the database containing it', 'its number of nodes', 'its edge density'], a: 1, why: 'The notebook counts each pattern once per graph; support = number of graphs that contain it.' },
      { q: 'Why is the WL hash used?', o: ['it is faster than anything else', 'it gives an isomorphism-invariant canonical signature (up to rare collisions)', 'it colours nodes randomly', 'it removes edges'], a: 1, why: 'Isomorphic labelled graphs always get the same WL signature, so equal signatures group occurrences of the same pattern.' },
      { q: 'If pattern P has support 3, an extension P′ ⊃ P can have support…', o: ['at most 3', 'at least 3', 'exactly 4', 'any number'], a: 0, why: 'Anti-monotone: every graph containing P′ must also contain P.' }],
    mount(root, ctx) {
      const NB = [mkGraph([[0, 1], [1, 2], [2, 3]], ['C', 'C', 'O', 'H']), mkGraph([[0, 1], [1, 2], [2, 3]], ['C', 'C', 'O', 'N']), mkGraph([[0, 1], [1, 2]], ['C', 'C', 'O']), mkGraph([[0, 1], [1, 2], [1, 3]], ['C', 'C', 'O', 'C'])];
      let db = NB, positions = db.map(g => layoutGraph(g, 200, 150)), minSup = 2, maxSize = 3, mined = [], selPat = null, wlG = 0, wlIt = 1, seed = 1;
      const grid = h('div', { class: 'grid', style: 'grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px' });
      const tableBox = h('div'), wlBox = h('div'), wlSlider = Ui.slider({ label: 'WL iterations', min: 0, max: 3, step: 1, value: 1, on: v => { wlIt = v; renderWL(); } });
      const stP = Ui.stat('patterns'), stG = Ui.stat('graphs');
      function renderDB() { grid.innerHTML = ''; db.forEach((g, gi) => { const hl = new Set(); if (selPat) selPat.occ[gi].forEach(nodes => nodes.forEach(v => hl.add(v))); const has = selPat && selPat.occ[gi].length; const card = h('div', { class: 'card', style: 'margin:0;padding:8px;' + (has ? 'border-color:#d7b56d' : '') + (gi === wlG ? ';box-shadow:0 0 0 2px #35d0ba55' : ''), onclick: () => { wlG = gi; renderDB(); renderWL(); } }, h('div', { class: 'muted', style: 'font-size:11px;margin-bottom:2px' }, `G${gi} · ${g.labels.length} nodes` + (has ? ` · ${selPat.occ[gi].length} occurrence(s)` : '')), drawSvgGraph(g, positions[gi], 200, 150, { hl })); card.style.cursor = 'pointer'; grid.appendChild(card); }); }
      function mine() {
        mined = mineFSM(db, minSup, maxSize); selPat = null; stP.set(mined.length); stG.set(db.length);
        tableBox.innerHTML = ''; const rows = mined.map((p, i) => { const sg = p.ex.nodes, g = p.ex.g; const sub = mkGraph(g.edges.filter(([a, b]) => sg.includes(a) && sg.includes(b)).map(([a, b]) => [sg.indexOf(a), sg.indexOf(b)]), sg.map(v => g.labels[v])); const pos = layoutGraph(sub, 70, 46); const tr = h('tr', { style: 'cursor:pointer', onclick: () => { selPat = selPat === p ? null : p; [...tbody.children].forEach((r, j) => r.style.background = mined[j] === selPat ? '#3a342b' : ''); renderDB(); } }, h('td', null, h('div', { style: 'width:80px' }, drawSvgGraph(sub, pos, 70, 46))), h('td', null, h('b', null, p.support + '/' + db.length), ' ', h('span', { class: 'muted' }, [...new Set(p.occ.flatMap((o, gi) => (o.length ? [gi] : [])))].map(x => 'G' + x).join(' '))), h('td', null, sub.labels.join('–'), h('div', { class: 'muted', style: 'font-size:11px' }, `${sub.labels.length} nodes · ${sub.edges.length} edges`))); return tr; });
        const tbody = h('tbody', null, rows); tableBox.appendChild(h('div', { class: 'scroll', style: 'max-height:380px' }, h('table', { class: 't' }, h('thead', null, h('tr', null, ['pattern', 'support', 'labels'].map(x => h('th', null, x)))), tbody))); renderDB();
      }
      function renderWL() { wlBox.innerHTML = ''; const g = db[wlG], cols = wlColors(g, wlIt); wlBox.appendChild(h('div', { style: 'max-width:300px' }, drawSvgGraph(g, positions[wlG], 200, 150, { colors: wlIt ? cols : null }))); wlBox.appendChild(h('div', { class: 'muted', style: 'font-size:12.5px;margin-top:4px' }, `G${wlG}: ${new Set(cols).size} distinct node colours after ${wlIt} WL iteration(s)` + (wlIt === 0 ? ' (= the atom labels)' : '') + '.')); wlBox.appendChild(h('div', { class: 'mono muted', style: 'font-size:11px;word-break:break-all;margin-top:4px' }, 'signature: ' + hstr(wlSig(g.labels, g.adj, 3)))); }
      function relabel() { const g = db[wlG], n = g.labels.length, perm = new RNG(Date.now() % 1000).shuffle(Array.from({ length: n }, (_, i) => i)); const g2 = mkGraph(g.edges.map(([a, b]) => [perm[a], perm[b]]), g.labels.map((_, i) => g.labels[perm.indexOf(i)])); const a = hstr(wlSig(g.labels, g.adj, 3)), b = hstr(wlSig(g2.labels, g2.adj, 3)); Ui.toast(`ids permuted ${perm.join(',')} → signature ${a} ${a === b ? '=' : '≠'} ${b}`); }
      function randomDB(n) { const r = new RNG(seed++ * 7919), atoms = ['C', 'C', 'C', 'O', 'N', 'H']; return Array.from({ length: n }, () => { const m = 4 + r.int(4), labels = Array.from({ length: m }, () => r.choice(atoms)), edges = []; for (let i = 1; i < m; i++) edges.push([r.int(i), i]); if (r.rand() < 0.4 && m > 3) { const a = r.int(m), b = r.int(m); if (a !== b && !edges.some(e => (e[0] === a && e[1] === b) || (e[0] === b && e[1] === a))) edges.push([a, b]); } return mkGraph(edges, labels); }); }
      const setDB = d => { db = d; positions = db.map(g => layoutGraph(g, 200, 150)); wlG = 0; sSup.set(Math.min(minSup, db.length)); sSup.querySelector('input').max = db.length; mine(); renderWL(); };
      const sSup = Ui.slider({ label: 'min support (graphs)', min: 1, max: 4, step: 1, value: 2, on: v => { minSup = v; mine(); } }), sSize = Ui.slider({ label: 'max pattern size (nodes)', min: 2, max: 5, step: 1, value: 3, on: v => { maxSize = v; mine(); } });
      const ctl = Ui.card('Mining', Ui.row(sSup, sSize), Ui.row(Ui.btn('Notebook DB (4 graphs)', () => setDB(NB)), Ui.btn('Random molecule DB (6)', () => setDB(randomDB(6)), 'pri')), Ui.row(stP, stG));
      root.appendChild(Ui.grid('g12', h('div', null, ctl, Ui.card('Frequent patterns — click to highlight', tableBox), Ui.card('WL colour-refinement lab', wlSlider, Ui.row(Ui.btn('Relabel & rehash', relabel))), Ui.card(null, wlBox)), Ui.card('Graph database (click a graph to inspect it in the WL lab)', grid)));
      mine(); renderWL();
    },
  });

  /* ============================================================ B2 ====== */
  Lab.register({
    id: 'b2', unit: 2, num: 'B2', bonus: true, title: 'PageRank & label propagation',
    tag: 'Syllabus extras: link analysis and non-neural semi-supervised classification on the same karate graph.',
    star: 'Random-surfer teleport slider · compare label propagation with the GCN',
    idea: `<p><b>PageRank</b> scores nodes by the stationary distribution of a random surfer who follows an edge with probability <code>d</code> and teleports with probability <code>1−d</code>. <b>Label propagation</b> spreads known labels along edges by repeated averaging — the non-learned ancestor of the GCN in T4. Click nodes to seed labels, step the iteration and watch both processes converge.</p>`,
    tryIt: ['Lower the damping factor: ranks flatten toward uniform (the surfer teleports a lot). Raise it to 0.99: the hubs (0, 33) dominate.', 'Seed node 0 and node 33 with different labels and press Propagate: every other node takes the label of the seed it is “closer” to in random-walk terms.'],
    hood: `<pre>PageRank: r ← (1−d)/n · 1 + d · Pᵀ r          (P = row-normalised adjacency)
Label propagation: F ← α · S F + (1−α) · Y            (S = D^-½ A D^-½, Y = seed one-hots)</pre>`,
    quiz: [{ q: 'In PageRank, damping d = 0.85 means…', o: ['15% of the time the surfer teleports', '85% of nodes are removed', 'the graph has 85 nodes', 'a learning rate'], a: 0, why: 'With probability 1−d the surfer jumps to a uniformly random node.' }, { q: 'Label propagation vs GCN: the GCN adds…', o: ['learned weights and features', 'edges', 'normalisation', 'nothing'], a: 0, why: 'LP is a fixed diffusion; a GCN learns transformations W per layer on node features.' }],
    mount(root, ctx) {
      const N = 34, nodes = Array.from({ length: N }, () => ({})), force = new Force(nodes, KE, { w: 640, h: 480 }); force.step(250);
      let d = 0.85, pr = new Float64Array(N).fill(1 / N), it = 0, seeds = new Map([[0, 0], [33, 1]]), F = null, running = false;
      const deg = new Float64Array(N); KE.forEach(([a, b]) => { deg[a]++; deg[b]++; });
      const stepPR = () => { const nr = new Float64Array(N).fill((1 - d) / N); KE.forEach(([a, b]) => { nr[b] += d * pr[a] / deg[a]; nr[a] += d * pr[b] / deg[b]; }); pr = nr; it++; };
      const S = ML.normAdj(N, KE, false); let alpha = 0.9;
      const resetLP = () => { F = Array.from({ length: N }, (_, i) => [seeds.get(i) === 0 ? 1 : 0, seeds.get(i) === 1 ? 1 : 0]); };
      const stepLP = () => { const Y = Array.from({ length: N }, (_, i) => [seeds.get(i) === 0 ? 1 : 0, seeds.get(i) === 1 ? 1 : 0]); const nf = Y.map((y, i) => { let a = 0, b = 0; for (let j = 0; j < N; j++) if (S[i * N + j]) { a += S[i * N + j] * F[j][0]; b += S[i * N + j] * F[j][1]; } return [alpha * a + (1 - alpha) * y[0], alpha * b + (1 - alpha) * y[1]]; }); F = nf; };
      resetLP();
      const cv = Ui.canvas(640, 480); let mode = 'pr';
      function draw() {
        const { g, w, h: H } = cv; g.clearRect(0, 0, w, H); KE.forEach(([a, b]) => { g.strokeStyle = '#3a342b'; g.lineWidth = 1.2; g.beginPath(); g.moveTo(nodes[a].x, nodes[a].y); g.lineTo(nodes[b].x, nodes[b].y); g.stroke(); });
        const mx = Math.max(...pr);
        for (let i = 0; i < N; i++) {
          const n = nodes[i]; let r = 9, fill = '#3a342b';
          if (mode === 'pr') { r = 5 + 12 * pr[i] / mx; fill = Colors.viridis(pr[i] / mx); } else { const f = F[i], t = f[0] + f[1]; if (t > 1e-6) { const c = f[0] > f[1] ? 0 : 1; fill = Colors.rgba(Colors.classColor(c), 0.35 + 0.65 * Math.abs(f[0] - f[1]) / t); } r = 12; }
          g.beginPath(); g.arc(n.x, n.y, r, 0, 7); g.fillStyle = fill; g.fill(); g.lineWidth = seeds.has(i) && mode === 'lp' ? 4 : 1.2; g.strokeStyle = seeds.has(i) && mode === 'lp' ? Colors.classColor(seeds.get(i)) : '#8a8171'; g.stroke(); g.fillStyle = mode === 'pr' && pr[i] / mx > 0.5 ? '#000' : '#fff'; g.font = '10px "Instrument Sans"'; g.textAlign = 'center'; g.fillText(i, n.x, n.y + 3.5);
        }
      }
      let drag = null, moved = false;
      cv.c.addEventListener('pointerdown', e => { const [x, y] = cv.xy(e); let b = -1, bd = 20; nodes.forEach((n, i) => { const dd = Math.hypot(n.x - x, n.y - y); if (dd < bd) { bd = dd; b = i; } }); if (b >= 0) { drag = b; moved = false; nodes[b].fixed = true; cv.c.setPointerCapture(e.pointerId); } });
      cv.c.addEventListener('pointermove', e => { if (drag === null) return; const [x, y] = cv.xy(e); nodes[drag].x = x; nodes[drag].y = y; moved = true; });
      cv.c.addEventListener('pointerup', () => { if (drag === null) return; nodes[drag].fixed = false; if (!moved && mode === 'lp') { const c = seeds.has(drag) ? seeds.get(drag) : -1; if (c >= 1) seeds.delete(drag); else seeds.set(drag, c + 1); resetLP(); } drag = null; });
      const seg = Ui.seg([{ v: 'pr', label: 'PageRank' }, { v: 'lp', label: 'Label propagation' }], 'pr', v => { mode = v; });
      const stI = Ui.stat('iterations'), stT = Ui.stat('top-ranked node');
      const run = Ui.btn('Run', () => { running = !running; run.textContent = running ? 'Pause' : 'Run'; }, 'pri');
      root.appendChild(Ui.grid('g21', Ui.card(null, h('div', { class: 'row', style: 'margin-bottom:8px' }, seg, h('span', { class: 'muted', style: 'font-size:12.5px' }, 'In label-propagation mode click nodes to seed class 0 / class 1 / none')), cv.c),
        Ui.card('Controls', Ui.row(run, Ui.btn('Step', () => { stepPR(); stepLP(); }), Ui.btn('Reset', () => { pr = new Float64Array(N).fill(1 / N); it = 0; resetLP(); })), Ui.slider({ label: 'PageRank damping d', min: 0.1, max: 0.99, step: 0.01, value: d, on: v => d = v }), Ui.slider({ label: 'label-prop α', min: 0.1, max: 0.99, step: 0.01, value: alpha, on: v => alpha = v }), Ui.row(stI, stT))));
      let f = 0; ctx.raf(() => { force.step(drag !== null || force.energy > 0.4 ? 2 : 0); if (running && f % 3 === 0) { stepPR(); stepLP(); } draw(); if (f++ % 6 === 0) { stI.set(it); stT.set(pr.indexOf(Math.max(...pr))); } });
    },
  });
})();
