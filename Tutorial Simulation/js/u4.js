/* Unit 4 — Association Rule Mining: T10 market-basket, T11 Apriori lattice, T12 FP-tree & pattern-based classification */
(function () {
  'use strict';
  const { RNG } = ML;

  const ITEMS = [{ id: 'milk', e: 'Mk', c: '#7aa2c8' }, { id: 'bread', e: 'Br', c: '#d9825b' }, { id: 'butter', e: 'Bu', c: '#d7b56d' }, { id: 'eggs', e: 'Eg', c: '#86b894' }, { id: 'cheese', e: 'Ch', c: '#a78bc4' }, { id: 'jam', e: 'Ja', c: '#c97b9e' }, { id: 'coffee', e: 'Co', c: '#6bb3aa' }, { id: 'sugar', e: 'Su', c: '#b4c46a' }];
  const EM = Object.fromEntries(ITEMS.map(i => [i.id, i.e])), COL = Object.fromEntries(ITEMS.map(i => [i.id, i.c]));
  const dot = id => h('i', { class: 'dot', style: 'background:' + (COL[id] || '#8a8171') });
  const itemBtn = (i, on, cls) => Ui.btn([dot(i.id), i.id], on, cls);
  const NOTEBOOK = [['milk', 'bread', 'butter'], ['bread', 'butter'], ['milk', 'bread'], ['milk', 'bread', 'butter', 'eggs'], ['bread', 'eggs'], ['milk', 'butter', 'eggs'], ['milk', 'bread', 'butter'], ['bread', 'butter', 'eggs']];
  const SHARED = { db: NOTEBOOK.map(t => t.slice()) }; // edited in T10, reused by T11 and T12
  const chips = (arr, cls = '') => h('span', null, arr.map(i => h('span', { class: 'tag ' + cls }, dot(i), i)));
  const setKey = s => [...s].sort().join('|');

  /* ---------------------------------------------------------------- engines */
  const countOf = (T, s) => { let c = 0; for (const t of T) { let ok = true; for (const i of s) if (!t.has(i)) { ok = false; break; } if (ok) c++; } return c; };
  function subsets(arr) { const out = []; for (let m = 1; m < (1 << arr.length); m++) out.push(arr.filter((_, i) => m & (1 << i))); return out; }
  // Apriori with a full trace of each level (for the lattice / step viewer)
  function apriori(tx, minSup) {
    const T = tx.map(t => new Set(t)), n = T.length, minCount = Math.max(1, Math.ceil(minSup * n - 1e-9));
    const items = [...new Set(tx.flat())].sort(); const levels = [], freq = new Map(); let scans = 0, counted = 0;
    let cands = items.map(i => ({ set: [i], count: countOf(T, [i]), status: 'counted' })); scans++; counted += cands.length;
    let k = 1;
    while (cands.length) {
      cands.forEach(c => { if (c.status === 'counted') c.status = c.count >= minCount ? 'frequent' : 'infrequent'; });
      levels.push({ k, cands, generated: cands.length, pruned: cands.filter(c => c.status === 'pruned').length });
      const L = cands.filter(c => c.status === 'frequent'); L.forEach(c => freq.set(setKey(c.set), c));
      if (!L.length) break;
      const next = [], seen = new Set(); k++;
      for (let a = 0; a < L.length; a++) for (let b = a + 1; b < L.length; b++) {
        const A = L[a].set, B = L[b].set; let ok = true; for (let i = 0; i < k - 2; i++) if (A[i] !== B[i]) { ok = false; break; } if (!ok || A[k - 2] === B[k - 2]) continue;
        const u = [...A, B[k - 2]].sort(); const key = setKey(u); if (seen.has(key)) continue; seen.add(key);
        let bad = null; for (let r = 0; r < u.length; r++) { const sub = u.filter((_, i) => i !== r); if (!freq.has(setKey(sub))) { bad = sub; break; } }
        if (bad) next.push({ set: u, count: null, status: 'pruned', why: bad }); else { next.push({ set: u, count: countOf(T, u), status: 'counted' }); counted++; }
      }
      if (next.some(c => c.status === 'counted')) scans++;
      cands = next.filter(c => c.status === 'pruned' || true);
      if (!cands.some(c => c.status === 'counted')) { if (cands.length) levels.push({ k, cands, generated: cands.length, pruned: cands.length }); break; }
    }
    return { n, minCount, items, levels, freq, scans, counted };
  }
  function bruteForce(tx, minSup) { const T = tx.map(t => new Set(t)), minCount = Math.max(1, Math.ceil(minSup * T.length - 1e-9)), items = [...new Set(tx.flat())].sort(), out = new Map(); for (const s of subsets(items)) { const c = countOf(T, s); if (c >= minCount) out.set(setKey(s), c); } return out; }
  function makeRules(res, minConf) {
    const rules = [], n = res.n, sup = s => res.freq.get(setKey(s)).count / n;
    for (const { set, count } of res.freq.values()) { if (set.length < 2) continue; for (const A of subsets(set)) { if (A.length === set.length) continue; const C = set.filter(i => !A.includes(i)), sAC = count / n, sA = sup(A), sC = sup(C), conf = sAC / sA; if (conf + 1e-12 < minConf) continue; rules.push({ A, C, support: sAC, confidence: conf, lift: conf / sC, leverage: sAC - sA * sC, conviction: conf >= 1 ? Infinity : (1 - sC) / (1 - conf) }); } }
    return rules;
  }
  // FP-tree --------------------------------------------------------------
  class FPNode { constructor(item, parent) { this.item = item; this.count = 0; this.parent = parent; this.children = new Map(); } }
  function fpOrder(tx, minCount) { const c = new Map(); tx.forEach(t => new Set(t).forEach(i => c.set(i, (c.get(i) || 0) + 1))); const order = [...c].filter(([, v]) => v >= minCount).sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).map(x => x[0]); return { order, counts: c }; }
  function fpInsert(root, header, path, w = 1) { let n = root; for (const it of path) { let ch = n.children.get(it); if (!ch) { ch = new FPNode(it, n); n.children.set(it, ch); if (!header.has(it)) header.set(it, []); header.get(it).push(ch); } ch.count += w; n = ch; } }
  function fpGrowth(tx, minCount) {
    const { order } = fpOrder(tx, minCount), results = [];
    const rank = new Map(order.map((it, i) => [it, i]));
    const build = (paths) => { const root = new FPNode(null, null), header = new Map(); for (const [p, w] of paths) fpInsert(root, header, p, w); return { root, header }; };
    const mine = (paths, suffix) => {
      const cnt = new Map(); for (const [p, w] of paths) for (const it of p) cnt.set(it, (cnt.get(it) || 0) + w);
      const its = [...cnt].filter(([, v]) => v >= minCount).sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : 1));
      for (const [it, c] of its) {
        const pat = [it, ...suffix]; results.push({ set: pat, count: c });
        const base = []; for (const [p, w] of paths) { const idx = p.indexOf(it); if (idx > 0) base.push([p.slice(0, idx), w]); }
        const cc = new Map(); base.forEach(([p, w]) => p.forEach(i => cc.set(i, (cc.get(i) || 0) + w))); const filt = base.map(([p, w]) => [p.filter(i => (cc.get(i) || 0) >= minCount), w]).filter(([p]) => p.length);
        if (filt.length) mine(filt, pat);
      }
    };
    const paths = tx.map(t => [[...new Set(t)].filter(i => rank.has(i)).sort((a, b) => rank.get(a) - rank.get(b)), 1]).filter(([p]) => p.length);
    mine(paths, []); return results;
  }
  const lbl = s => s.map(i => EM[i] || i).join('·');

  /* ============================================================ T10 ===== */
  Lab.register({
    id: 't10', unit: 4, num: 'T10', title: 'Market-basket analysis',
    tag: 'Build your own shopping baskets, mine association rules, see support × confidence × lift in one plot — and let the rules recommend products.',
    star: 'Click-to-build baskets with planted patterns · rule scatter · live “customers also bought” recommender',
    idea: `<p>Association-rule mining finds <i>items that occur together</i>. For a rule <code>A → C</code>: <b>support</b> = P(A∧C) (how common), <b>confidence</b> = P(C|A) (how reliable), <b>lift</b> = confidence / P(C) (how much A <i>raises</i> the chance of C: &gt;1 positive, 1 independent, &lt;1 substitutes). The notebook mines 8 hard-coded baskets with mlxtend; here <b>you build the baskets</b> (they are reused in T11 and T12) or generate hundreds with planted rules and watch the miner rediscover them.</p>`,
    tryIt: ['Press <b>Generate 60 baskets (planted rules)</b>: coffee→sugar, bread→butter and milk→eggs are planted; the top of the rule table should recover them with lift well above 1.',
      'Drag <b>min support</b> up: rare-but-strong rules disappear first (lower it to 0.25 to also surface the planted milk→eggs rule) — support filters by popularity, not strength.',
      'High confidence is not enough: find a rule with confidence ≥ 0.8 but lift ≈ 1 (the consequent is just very popular).',
      'In the <b>recommender</b> click items to put them in a cart and read the suggestions (rules whose antecedent is inside the cart).',
      'Add your own basket: click items in the palette, then <b>Add basket</b>.'],
    hood: `<pre>support(X)       = |{t ⊇ X}| / |T|
confidence(A→C)  = support(A∪C) / support(A)
lift(A→C)        = confidence / support(C)          leverage = s(A∪C) − s(A)s(C)
conviction       = (1 − s(C)) / (1 − confidence)

pipeline = one-hot encode (TransactionEncoder) → Apriori(min_support) → association_rules(metric, threshold)</pre>
    <p>Rules are generated from every frequent itemset I by splitting it into all (A, I∖A) pairs and keeping those above the confidence threshold — exactly what <code>association_rules</code> does.</p>`,
    quiz: [
      { q: 'Lift = 1 means…', o: ['the rule always holds', 'A and C are statistically independent', 'C implies A', 'support is 1'], a: 1, why: 'If confidence equals the baseline frequency of C, knowing A tells you nothing about C.' },
      { q: 'High confidence but low lift usually means…', o: ['a great rule', 'the consequent is simply very common', 'a data error', 'negative support'], a: 1, why: 'conf(A→C) can be high just because C appears in almost every basket.' },
      { q: 'Raising min_support will…', o: ['add more itemsets', 'remove infrequent itemsets and the rules built from them', 'raise lift', 'change confidence of surviving rules'], a: 1, why: 'Fewer itemsets pass the support filter; the confidence of rules among survivors is unchanged.' }],
    mount(root, ctx) {
      const P = { minSup: 0.3, minConf: 0.6, sort: 'lift' }; let cur = new Set(), cart = new Set(), hoverRule = null;
      const palette = h('div', { class: 'row' }), basketBox = h('div', { class: 'scroll', style: 'max-height:210px' }), curBox = h('div', { style: 'min-height:30px;margin:8px 0' });
      const rulesBox = h('div'), freqBox = h('div'), recBox = h('div'), stats = h('div', { class: 'row' }); const cv = Ui.canvas(520, 340), tip = h('div', { class: 'muted', style: 'font-size:12.5px;min-height:34px;margin-top:6px' }, 'Hover a dot in the plot to read the rule.');
      let dots = [], res, rules;
      const mineIt = () => { res = apriori(SHARED.db, P.minSup); rules = makeRules(res, P.minConf).sort((a, b) => b[P.sort] - a[P.sort]); };
      function renderPalette() { palette.innerHTML = ''; ITEMS.forEach(i => palette.appendChild(itemBtn(i, () => { cur.has(i.id) ? cur.delete(i.id) : cur.add(i.id); renderCur(); }, 'sm' + (cur.has(i.id) ? ' on' : '')))); }
      function renderCur() { renderPalette(); curBox.innerHTML = ''; curBox.appendChild(cur.size ? chips([...cur]) : h('span', { class: 'muted' }, 'click items above to build a basket…')); }
      function renderBaskets() { basketBox.innerHTML = ''; SHARED.db.forEach((t, i) => basketBox.appendChild(h('div', { class: 'row', style: 'padding:3px 0;border-bottom:1px solid #2a2620' }, h('span', { class: 'muted mono', style: 'width:26px' }, '#' + (i + 1)), chips(t), h('span', { class: 'sp' }), h('button', { class: 'btn sm', onclick: () => { SHARED.db.splice(i, 1); refresh(); } }, '×')))); }
      function planted(n, seed) { const r = new RNG(seed), out = []; for (let i = 0; i < n; i++) { const b = new Set(); if (r.rand() < 0.45) { b.add('coffee'); if (r.rand() < 0.85) b.add('sugar'); } if (r.rand() < 0.55) { b.add('bread'); if (r.rand() < 0.8) b.add('butter'); } if (r.rand() < 0.4) { b.add('milk'); if (r.rand() < 0.7) b.add('eggs'); } ['cheese', 'jam', 'sugar', 'eggs', 'milk', 'butter'].forEach(x => { if (r.rand() < 0.08) b.add(x); }); if (!b.size) b.add(r.choice(ITEMS).id); out.push([...b]); } return out; }
      function renderRules() {
        rulesBox.innerHTML = ''; rulesBox.appendChild(Ui.table(['rule', 'support', 'confidence', 'lift', 'leverage'], rules.slice(0, 60).map((r, i) => [h('span', { onmouseenter: () => { hoverRule = r; }, onmouseleave: () => { hoverRule = null; } }, chips(r.A), ' → ', chips(r.C, 'good')), r.support.toFixed(2), r.confidence.toFixed(2), h('b', { class: r.lift > 1.05 ? 'good' : r.lift < 0.95 ? 'bad' : '' }, r.lift.toFixed(2)), r.leverage.toFixed(3)]), { maxH: '300px' }));
        freqBox.innerHTML = ''; freqBox.appendChild(Ui.table(['frequent itemset', 'support', 'count'], [...res.freq.values()].sort((a, b) => b.count - a.count || a.set.length - b.set.length).map(c => [chips(c.set), (c.count / res.n).toFixed(2), c.count]), { maxH: '240px' }));
        stats.innerHTML = ''; [['baskets', res.n], ['items', res.items.length], ['frequent sets', res.freq.size], ['rules', rules.length]].forEach(([a, b]) => stats.appendChild(Ui.stat(a, b)));
      }
      function renderRec() {
        recBox.innerHTML = ''; const pal = h('div', { class: 'row' }); ITEMS.forEach(i => pal.appendChild(itemBtn(i, () => { cart.has(i.id) ? cart.delete(i.id) : cart.add(i.id); renderRec(); }, 'sm' + (cart.has(i.id) ? ' on' : '')))); recBox.appendChild(pal);
        const sug = new Map(); rules.filter(r => r.A.every(a => cart.has(a))).forEach(r => r.C.forEach(c => { if (!cart.has(c)) { const o = sug.get(c); if (!o || r.confidence > o.conf) sug.set(c, { conf: r.confidence, lift: r.lift, rule: r }); } }));
        recBox.appendChild(h('div', { style: 'margin-top:10px' }, cart.size ? (sug.size ? [...sug].sort((a, b) => b[1].conf - a[1].conf).map(([c, o]) => h('div', { class: 'row', style: 'margin:4px 0' }, dot(c), h('b', null, c), h('span', { class: 'muted' }, `because ${o.rule.A.join(' + ')} → confidence ${(o.conf * 100).toFixed(0)}%, lift ${o.lift.toFixed(2)}`))) : h('span', { class: 'muted' }, 'No rule fires for this cart at the current thresholds.')) : h('span', { class: 'muted' }, 'Add items to the cart to see recommendations.')));
      }
      function drawPlot() {
        const { g, w, h: H } = cv, m = { l: 44, r: 14, t: 14, b: 34 }; g.clearRect(0, 0, w, H); g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.textAlign = 'center'; g.fillText('support →', w / 2, H - 6); g.save(); g.translate(12, H / 2); g.rotate(-Math.PI / 2); g.fillText('confidence →', 0, 0); g.restore();
        const sx = v => m.l + v * (w - m.l - m.r), sy = v => H - m.b - v * (H - m.t - m.b); g.strokeStyle = '#2a2620'; for (let i = 0; i <= 4; i++) { const v = i / 4; g.beginPath(); g.moveTo(sx(0), sy(v)); g.lineTo(sx(1), sy(v)); g.stroke(); g.beginPath(); g.moveTo(sx(v), sy(0)); g.lineTo(sx(v), sy(1)); g.stroke(); g.fillStyle = '#6f6859'; g.textAlign = 'right'; g.fillText(v.toFixed(2), m.l - 4, sy(v) + 4); g.textAlign = 'center'; g.fillText(v.toFixed(2), sx(v), H - m.b + 13); }
        dots = []; const mxL = Math.max(2, ...rules.map(r => r.lift)); rules.forEach(r => { const x = sx(r.support), y = sy(r.confidence), rad = 4 + 7 * Math.min(1, r.lift / mxL); dots.push({ x, y, r: rad, rule: r }); const hot = hoverRule === r; g.beginPath(); g.arc(x, y, rad + (hot ? 3 : 0), 0, 7); g.fillStyle = r.lift < 1 ? Colors.rgba('#7aa2c8', 0.7) : Colors.viridis(Math.min(1, (r.lift - 1) / (mxL - 1 + 1e-9))); g.globalAlpha = 0.85; g.fill(); g.globalAlpha = 1; g.lineWidth = hot ? 3 : 1; g.strokeStyle = hot ? '#fff' : '#100e0c'; g.stroke(); });
        g.fillStyle = '#9b9384'; g.textAlign = 'left'; g.fillText('dot size & colour = lift (blue < 1, yellow = highest)', m.l + 6, m.t + 8);
      }
      cv.c.addEventListener('pointermove', e => { const [x, y] = cv.xy(e); let b = null, bd = 1e9; dots.forEach(d => { const dd = Math.hypot(d.x - x, d.y - y); if (dd < d.r + 5 && dd < bd) { bd = dd; b = d; } }); hoverRule = b ? b.rule : null; tip.textContent = b ? `${b.rule.A.join(' + ')} → ${b.rule.C.join(' + ')}   support ${b.rule.support.toFixed(2)} · confidence ${b.rule.confidence.toFixed(2)} · lift ${b.rule.lift.toFixed(2)}` : 'Hover a dot in the plot to read the rule.'; });
      function refresh() { mineIt(); renderBaskets(); renderRules(); renderRec(); }
      const sSup = Ui.slider({ label: 'min support', min: 0.05, max: 0.9, step: 0.05, value: P.minSup, on: v => { P.minSup = v; refresh(); } }), sConf = Ui.slider({ label: 'min confidence', min: 0.1, max: 1, step: 0.05, value: P.minConf, on: v => { P.minConf = v; refresh(); } });
      const sortSeg = Ui.seg([{ v: 'lift', label: 'lift' }, { v: 'confidence', label: 'confidence' }, { v: 'support', label: 'support' }], 'lift', v => { P.sort = v; refresh(); });
      root.appendChild(Ui.grid('g2', h('div', null,
        Ui.card('Basket builder', palette, curBox, Ui.row(Ui.btn('Add basket', () => { if (!cur.size) return Ui.toast('Pick at least one item'); SHARED.db.push([...cur]); cur = new Set(); renderCur(); refresh(); }, 'pri'), Ui.btn('Generate 60 baskets (planted rules)', () => { SHARED.db = planted(60, Math.floor(Math.random() * 1e6)); refresh(); }), Ui.btn('Notebook data (8 baskets)', () => { SHARED.db = NOTEBOOK.map(t => t.slice()); refresh(); }), Ui.btn('Clear', () => { SHARED.db = []; refresh(); }))),
        Ui.card('Baskets (shared with T11 and T12)', basketBox), Ui.card('Recommender — “customers who bought … also bought”', recBox)),
        h('div', null, Ui.card('Thresholds', Ui.row(sSup, sConf), h('div', { style: 'margin-top:6px' }, 'sort rules by ', sortSeg), h('div', { style: 'margin-top:10px' }, stats)), Ui.card('Rules: support × confidence × lift', cv.c, tip), Ui.card('Association rules', rulesBox), Ui.card('Frequent itemsets (Apriori)', freqBox))));
      renderCur(); refresh(); ctx.raf(() => drawPlot());
    },
  });

  /* ============================================================ T11 ===== */
  Lab.register({
    id: 't11', unit: 4, num: 'T11', title: 'The Apriori algorithm',
    tag: 'Step through candidate generation, support counting and downward-closure pruning on the itemset lattice — and verify it against brute force.',
    star: 'Itemset lattice with pruned regions · counts saved vs brute force · independent correctness check',
    idea: `<p><b>Apriori</b> exploits the <i>downward-closure</i> (anti-monotone) property: <b>every subset of a frequent itemset is frequent</b>, so if any subset is infrequent the superset cannot be frequent and need not even be counted. It proceeds level by level: <code>L₁ → candidates C₂ (join) → prune → count → L₂ → …</code>. The notebook implements this and checks it against mlxtend; here you see <b>the lattice of all 2ⁿ−1 itemsets</b> light up level by level, with the pruned region greyed out, and a brute-force enumeration confirms the result.</p>`,
    tryIt: ['Press <b>Next level</b> and read the log: how many candidates were generated, how many were pruned <i>without being counted</i>, and which infrequent subset killed each one (hover a grey node).',
      'Drop <b>min support</b>: fewer nodes are pruned and the frontier reaches deeper levels — the saving versus brute force shrinks.',
      'Edit the baskets in T10 (shared!) and come back — the lattice is rebuilt from your data.',
      'The correctness badge compares the frequent itemsets with an exhaustive 2ⁿ scan — they must agree.'],
    hood: `<pre>L1 = { i : support(i) ≥ min_sup }
for k = 2, 3, …
    C_k = { A ∪ B : A, B ∈ L_{k-1}, |A ∩ B| = k-2 }                 (join)
    C_k ← { c ∈ C_k : every (k-1)-subset of c is in L_{k-1} }       (prune: downward closure)
    count support of the survivors with one scan of the database    (count)
    L_k = { c ∈ C_k : support(c) ≥ min_sup }                         (filter)</pre>
    <p>The “counted” statistic is the number of itemsets whose support was actually computed; brute force would compute all 2ⁿ−1.</p>`,
    quiz: [
      { q: 'If {milk, eggs} is infrequent, then {milk, eggs, bread} is…', o: ['frequent', 'infrequent (so it can be pruned unseen)', 'unknown, must be counted', 'always frequent'], a: 1, why: 'Support can only shrink when items are added (anti-monotone).' },
      { q: 'The prune step checks…', o: ['that all (k−1)-subsets of a candidate are frequent', 'the confidence', 'the lift', 'that the candidate is sorted'], a: 0, why: 'A candidate with an infrequent subset cannot be frequent, so it is dropped before counting.' },
      { q: 'Why does Apriori need one database scan per level?', o: ['supports of all candidates of that size are counted together', 'it re-sorts the data', 'it checks lift', 'it has no choice'], a: 0, why: 'All candidates of size k are counted in one pass over the transactions.' }],
    mount(root, ctx) {
      const P = { minSup: 0.3, level: 1 }; let res, nodes = [], edges = [], hover = null; const cv = Ui.canvas(900, 430), log = h('div', { class: 'scroll', style: 'max-height:250px;font-size:13px' }), verdict = h('div'), tip = h('div', { class: 'muted', style: 'font-size:12.5px;min-height:36px;margin-top:6px' }, 'Hover a node.'); const stats = h('div', { class: 'row' });
      function build() {
        const db = SHARED.db; if (!db.length) { res = null; return; } res = apriori(db, P.minSup); const its = res.items; const n = its.length; const levelsRevealed = P.level;
        const info = new Map(); res.levels.forEach(L => L.cands.forEach(c => info.set(setKey(c.set), Object.assign({ k: L.k }, c)))); nodes = []; edges = [];
        const byK = Array.from({ length: n + 1 }, () => []); subsets(its).forEach(s => byK[s.length].push(s)); const { w, h: H } = cv;
        for (let k = 1; k <= n; k++) byK[k].forEach((s, i) => { const key = setKey(s), inf = info.get(key); let status = 'unreached'; if (k <= levelsRevealed) status = inf ? inf.status : 'pruned'; nodes.push({ s, k, key, x: 30 + (i + 0.5) * (w - 60) / byK[k].length, y: H - 36 - (k - 1) * (H - 80) / Math.max(1, n - 1), status, inf }); });
        const idx = new Map(nodes.map((nd, i) => [nd.key, i])); nodes.forEach((nd, i) => { if (nd.k < 2) return; nd.s.forEach(it => { const sub = nd.s.filter(x => x !== it); edges.push([idx.get(setKey(sub)), i]); }); });
        const bf = bruteForce(db, P.minSup); const same = bf.size === res.freq.size && [...bf.keys()].every(k => res.freq.has(k)); verdict.innerHTML = ''; verdict.appendChild(h('div', { class: 'note', style: 'border-color:' + (same ? 'var(--good)' : 'var(--bad)') }, same ? `Verified: Apriori's ${res.freq.size} frequent itemsets are identical to an exhaustive scan of all ${(1 << n) - 1} subsets.` : 'MISMATCH with brute force (this should never happen)'));
        stats.innerHTML = ''; [['baskets', res.n], ['min count', res.minCount], ['frequent', res.freq.size], ['counted', res.counted + ' / ' + ((1 << n) - 1)], ['db scans', res.scans]].forEach(([a, b]) => stats.appendChild(Ui.stat(a, b)));
        log.innerHTML = ''; res.levels.forEach(L => { if (L.k > levelsRevealed) return; const fr = L.cands.filter(c => c.status === 'frequent'), inf = L.cands.filter(c => c.status === 'infrequent'), pr = L.cands.filter(c => c.status === 'pruned'); log.appendChild(h('div', { style: 'margin:0 0 8px;padding:6px 8px;border-left:3px solid #a78bc4;background:#1a1713;border-radius:0 8px 8px 0' }, h('b', null, `Level ${L.k}`), L.k === 1 ? `: counted all ${L.cands.length} single items → ${fr.length} frequent` : `: join produced ${L.cands.length} candidates, prune removed ${pr.length} (no counting needed), counted ${L.cands.length - pr.length} → ${fr.length} frequent, ${inf.length} infrequent`, fr.length ? h('div', null, 'L', h('sub', null, L.k), ' = ', chips(fr.map(c => c.set).map(s => s.join('+')).slice(0, 12))) : null)); });
        
      }
      function draw() {
        const { g, w, h: H } = cv; g.clearRect(0, 0, w, H); if (!res) { g.fillStyle = '#9b9384'; g.fillText('No baskets — add some in T10', 20, 30); return; }
        g.lineWidth = 1; edges.forEach(([a, b]) => { const A = nodes[a], B = nodes[b]; const live = A.status === 'frequent' && B.status !== 'unreached' && B.status !== 'pruned'; g.strokeStyle = live ? '#544c3f' : '#27231e'; g.beginPath(); g.moveTo(A.x, A.y); g.lineTo(B.x, B.y); g.stroke(); });
        const n = res.items.length, big = n >= 6; nodes.forEach(nd => {
          const col = { frequent: '#86b894', infrequent: '#d9825b', pruned: '#4a4338', unreached: '#1c1915', counted: '#d7b56d' }[nd.status], r = big ? 9 : 14;
          g.beginPath(); g.arc(nd.x, nd.y, r, 0, 7); g.fillStyle = col; g.fill(); g.lineWidth = hover === nd ? 3 : 1; g.strokeStyle = hover === nd ? '#fff' : (nd.status === 'pruned' ? '#6f6859' : '#100e0c'); if (nd.status === 'pruned') g.setLineDash([3, 3]); g.stroke(); g.setLineDash([]);
          g.fillStyle = nd.status === 'unreached' || nd.status === 'pruned' ? '#7d7566' : '#14110d'; g.font = (big ? '9' : '11') + 'px "JetBrains Mono",monospace'; g.textAlign = 'center'; g.fillText(nd.inf && nd.inf.count != null && nd.status !== 'unreached' ? nd.inf.count : nd.status === 'pruned' ? '×' : '', nd.x, nd.y + 3.5);
          if (!big) { g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.fillText(lbl(nd.s), nd.x, nd.y - r - 4); }
        });
        g.textAlign = 'left'; g.fillStyle = '#9b9384'; g.font = '11.5px "Instrument Sans"'; for (let k = 1; k <= n; k++) { const y = H - 36 - (k - 1) * (H - 80) / Math.max(1, n - 1); g.fillText('k=' + k, 4, y + 4); }
        [['#86b894', 'frequent'], ['#d9825b', 'counted, infrequent'], ['#4a4338', 'pruned (never counted)'], ['#1c1915', 'not reached yet']].forEach(([c, t], i) => { g.fillStyle = c; g.fillRect(w - 190, 8 + i * 17, 11, 11); g.fillStyle = '#cfc8b8'; g.fillText(t, w - 174, 18 + i * 17); });
      }
      cv.c.addEventListener('pointermove', e => { const [x, y] = cv.xy(e); hover = null; let bd = 16; nodes.forEach(nd => { const d = Math.hypot(nd.x - x, nd.y - y); if (d < bd) { bd = d; hover = nd; } }); if (!hover) { tip.textContent = 'Hover a node.'; return; } const nd = hover; tip.innerHTML = `<b>${nd.s.join(' + ')}</b> · ` + (nd.status === 'frequent' ? `support ${(nd.inf.count / res.n).toFixed(2)} (${nd.inf.count}/${res.n}) ≥ min → <span class="good">frequent</span>` : nd.status === 'infrequent' ? `support ${(nd.inf.count / res.n).toFixed(2)} < min → <span class="warn">infrequent</span>` : nd.status === 'pruned' ? (nd.inf && nd.inf.why ? `<span class="bad">pruned</span>: its subset {${nd.inf.why.join(', ')}} is infrequent, so it is never counted` : `<span class="bad">pruned</span>: contains an infrequent subset (never generated)`) : 'not reached yet — press Next level');
      });
      cv.c.addEventListener('pointerleave', () => { hover = null; });
      const lvlTxt = h('span', { class: 'pill' }, 'level 1');
      const maxLevels = () => (res ? res.items.length : 1);
      const next = () => { P.level = Math.min(P.level + 1, maxLevels() + 0); build(); lvlTxt.textContent = 'level ' + P.level; }, prev = () => { P.level = Math.max(1, P.level - 1); build(); lvlTxt.textContent = 'level ' + P.level; }, all = () => { P.level = maxLevels(); lvlTxt.textContent = 'all levels'; build(); };
      const sSup = Ui.slider({ label: 'min support', min: 0.05, max: 0.9, step: 0.05, value: P.minSup, on: v => { P.minSup = v; P.level = 1; lvlTxt.textContent = 'level 1'; build(); } });
      build();
      root.appendChild(Ui.card('Itemset lattice  (circles show support counts; level k = itemsets of size k)', h('div', { class: 'row', style: 'margin-bottom:8px' }, Ui.btn('Previous', prev), Ui.btn('Next level', next, 'pri'), Ui.btn('Run to the end', all), lvlTxt, h('span', { class: 'sp' }), h('div', { style: 'width:260px' }, sSup)), cv.c, tip));
      root.appendChild(Ui.grid('g2', h('div', null, Ui.card('What happened', log)), h('div', null, Ui.card('Cost & correctness', stats, h('div', { style: 'margin-top:10px' }, verdict)), Ui.card('Baskets used', h('div', { class: 'muted', style: 'font-size:13px' }, `${SHARED.db.length} baskets from T10 (edit them there). ${res ? 'Items: ' + res.items.join(', ') : ''}`)))));
      ctx.raf(draw);
    },
  });

  /* ============================================================ T12 ===== */
  const IRIS = [[[5.006, .352], [3.428, .379], [1.462, .174], [.246, .105]], [[5.936, .516], [2.770, .314], [4.260, .470], [1.326, .198]], [[6.588, .636], [2.974, .322], [5.552, .552], [2.026, .275]]];
  const FNAMES = ['sepal length', 'sepal width', 'petal length', 'petal width'], CLS = ['setosa', 'versicolor', 'virginica'];
  Lab.register({
    id: 't12', unit: 4, num: 'T12', title: 'FP-tree & patterns that classify',
    tag: 'Watch an FP-tree grow transaction by transaction, mine it with FP-Growth, then turn the patterns into features for a classifier.',
    star: 'Animated FP-tree construction · conditional pattern bases · patterns vs raw features · flower-probe',
    idea: `<p>The <b>FP-tree</b> compresses the transaction database into a prefix tree: items are re-ordered by global frequency so common prefixes share nodes. <b>FP-Growth</b> then mines frequent patterns <i>without candidate generation</i> by recursively building conditional trees from each item’s prefix paths. Part A lets you build the tree step by step and inspect each item’s conditional pattern base. Part B is the notebook’s classification use: discretise the flowers into low/med/high <i>items</i>, mine frequent patterns, turn each pattern into a binary feature and train a classifier.</p>
    <p><i>Data note:</i> Iris isn’t bundled, so Part B samples 150 flowers from Gaussians with the real Iris class means and standard deviations (50 per class, a fixed seed) — the same structure with slightly different numbers.</p>`,
    tryIt: ['Part A: press <b>Next transaction</b> a few times. Watch counts increase on shared prefixes (e.g. bread:6 …) while new branches appear only when the ordered path diverges. Header-table dotted links join all nodes of the same item.',
      'Pick an item in the dropdown: its <b>conditional pattern base</b> (prefix paths + counts) is what FP-Growth recurses on.',
      'Part B: lower <b>min support</b> to mine more patterns. Pattern features usually land a few points below raw numeric features (discretising into 3 bins throws away detail) but stay fully interpretable — read the class-pure patterns on the right: they are human-readable rules.',
      'Use the <b>flower probe</b> sliders: see which mined patterns fire and the predicted class probabilities.'],
    hood: `<pre>1. count items, drop those below min_count, order the rest by descending count
2. insert each ordered transaction into the tree, incrementing node counts along shared prefixes
3. for each item (least frequent first):  pattern base = its prefix paths (with counts)
      → conditional FP-tree → recurse, appending the item to the suffix
patterns = every (suffix ∪ item) found with count ≥ min_count</pre>
    <p>Classification: <code>x_pattern[p] = 1 if the flower contains every item of pattern p</code>; then multinomial logistic regression on those binary features (the notebook uses scikit-learn’s <code>LogisticRegression</code>).</p>`,
    quiz: [
      { q: 'Why does the FP-tree order items by descending frequency?', o: ['alphabetical order is slow', 'so frequent items form shared prefixes and the tree stays compact', 'to compute confidence', 'to remove duplicates'], a: 1, why: 'Frequent items near the root maximise prefix sharing between transactions.' },
      { q: 'FP-Growth avoids…', o: ['counting support', 'generating candidate itemsets', 'the database', 'recursion'], a: 1, why: 'It mines directly from conditional trees instead of generating and testing candidates like Apriori.' },
      { q: 'In Part B each mined pattern becomes…', o: ['a class label', 'a binary feature “does this sample contain the pattern?”', 'a neuron', 'a rule confidence'], a: 1, why: 'Frequent patterns act as engineered features for the classifier.' }],
    mount(root, ctx) {
      /* ---------- Part A ---------- */
      const A = { minCount: 2, pos: 0, sel: '', useShared: true }; let tree, ordered = [], lastPath = [], order = [], counts;
      const HAN = [['I1', 'I2', 'I5'], ['I2', 'I4'], ['I2', 'I3'], ['I1', 'I2', 'I4'], ['I1', 'I3'], ['I2', 'I3'], ['I1', 'I3'], ['I1', 'I2', 'I3', 'I5'], ['I1', 'I2', 'I3']];
      const dbSel = () => (A.useShared ? SHARED.db : HAN);
      const cvT = Ui.canvas(1160, 380), info = h('div', { class: 'scroll', style: 'max-height:230px;font-size:13px' }), mineBox = h('div'), cmp = h('div'), txBox = h('div', { class: 'scroll', style: 'max-height:200px' }), itemSel = h('select', { onchange: () => { A.sel = itemSel.value; showBase(); } });
      const itemCol = it => Colors.classColor(Math.max(0, order.indexOf(it)));
      function rebuildA(upTo) {
        const db = dbSel(); const o = fpOrder(db, A.minCount); order = o.order; counts = o.counts; tree = { root: new FPNode(null, null), header: new Map() }; ordered = db.map(t => [...new Set(t)].filter(i => order.includes(i)).sort((a, b) => order.indexOf(a) - order.indexOf(b)));
        for (let i = 0; i < upTo; i++) fpInsert(tree.root, tree.header, ordered[i]); lastPath = upTo ? ordered[upTo - 1] : []; A.pos = upTo;
        itemSel.innerHTML = ''; [...order].reverse().forEach(it => itemSel.appendChild(h('option', { value: it }, it))); if (!order.includes(A.sel)) A.sel = order[order.length - 1] || ''; itemSel.value = A.sel; renderTx(); showBase(); mineAll();
      }
      function renderTx() { const db = dbSel(); txBox.innerHTML = ''; txBox.appendChild(Ui.table(['#', 'transaction', 'ordered (frequent items)'], db.map((t, i) => [h('span', { style: i === A.pos - 1 ? 'color:#d7b56d;font-weight:700' : i < A.pos ? '' : 'opacity:.4' }, i + 1), chips(t), ordered[i] ? h('span', { class: 'mono', style: i < A.pos ? '' : 'opacity:.4' }, ordered[i].join(' → ')) : '']), { maxH: '200px' })); info.innerHTML = ''; info.appendChild(h('div', { class: 'muted' }, `min count ${A.minCount} · item order (by frequency): ${order.map(i => i + ':' + counts.get(i)).join(', ')} · ${A.pos}/${db.length} transactions inserted`)); }
      function layout() { let leaf = 0, maxD = 0; const nodes = []; (function rec(n, d) { n.depth = d; maxD = Math.max(maxD, d); if (!n.children.size) { n.x = leaf++; } else { const xs = []; [...n.children.values()].sort((a, b) => order.indexOf(a.item) - order.indexOf(b.item)).forEach(c => { rec(c, d + 1); xs.push(c.x); }); n.x = (Math.min(...xs) + Math.max(...xs)) / 2; } nodes.push(n); })(tree.root, 0); return { nodes, leaves: Math.max(1, leaf), maxD }; }
      function drawTree() {
        const { g, w, h: H } = cvT; g.clearRect(0, 0, w, H); const { nodes, leaves, maxD } = layout(); const L = 92, sx = n => L + (n.x + 0.5) * (w - L - 20) / leaves, sy = n => 30 + n.depth * Math.min(62, (H - 60) / Math.max(1, maxD)); const inPath = new Set(); { let n = tree.root; for (const it of lastPath) { n = n.children.get(it); if (!n) break; inPath.add(n); } }
        nodes.forEach(n => n.children.forEach(c => { g.strokeStyle = inPath.has(c) ? '#d7b56d' : '#544c3f'; g.lineWidth = inPath.has(c) ? 3 : 1.6; g.beginPath(); g.moveTo(sx(n), sy(n)); g.lineTo(sx(c), sy(c)); g.stroke(); }));
        g.setLineDash([3, 4]); g.lineWidth = 1; tree.header.forEach((list, it) => { g.strokeStyle = Colors.rgba(itemCol(it), 0.55); g.beginPath(); list.forEach((n, i) => { i ? g.lineTo(sx(n), sy(n)) : g.moveTo(sx(n), sy(n)); }); g.stroke(); }); g.setLineDash([]);
        nodes.forEach(n => { const root0 = n.item === null; g.beginPath(); g.arc(sx(n), sy(n), root0 ? 14 : 17, 0, 7); g.fillStyle = root0 ? '#3a342b' : itemCol(n.item); g.fill(); g.lineWidth = inPath.has(n) ? 3 : 1; g.strokeStyle = inPath.has(n) ? '#fff' : '#100e0c'; g.stroke(); g.fillStyle = root0 ? '#fff' : '#14110d'; g.font = '600 10.5px "JetBrains Mono",monospace'; g.textAlign = 'center'; g.fillText(root0 ? 'root' : n.item, sx(n), sy(n) - 1); if (!root0) g.fillText(n.count, sx(n), sy(n) + 10); });
        g.textAlign = 'left'; g.font = '11px "Instrument Sans"'; g.fillStyle = '#9b9384'; g.fillText('header table (dotted links connect nodes of the same item)', 6, 14); order.forEach((it, i) => { g.fillStyle = itemCol(it); g.fillRect(6, 24 + i * 19, 74, 16); g.fillStyle = '#14110d'; g.font = '600 10.5px "JetBrains Mono",monospace'; g.fillText(it + ' : ' + counts.get(it), 10, 36 + i * 19); });
      }
      function showBase() {
        mineBox.innerHTML = ''; if (!A.sel || !tree.header.has(A.sel)) { mineBox.appendChild(h('span', { class: 'muted' }, 'Insert transactions to see this item’s conditional pattern base.')); return; }
        const base = []; tree.header.get(A.sel).forEach(n => { const p = []; let q = n.parent; while (q && q.item !== null) { p.unshift(q.item); q = q.parent; } if (p.length) base.push([p, n.count]); });
        const cc = new Map(); base.forEach(([p, w]) => p.forEach(i => cc.set(i, (cc.get(i) || 0) + w))); const keep = [...cc].filter(([, v]) => v >= A.minCount).map(x => x[0]);
        mineBox.appendChild(h('div', null, h('b', null, `Conditional pattern base of ${A.sel}`), ` (prefix paths ending just above each ${A.sel} node):`)); mineBox.appendChild(Ui.table(['prefix path', 'count'], base.length ? base.map(([p, w]) => [p.join(' → '), w]) : [['(none — item is directly under the root)', '']]));
        mineBox.appendChild(h('div', { class: 'note' }, 'Conditional FP-tree keeps only items with count ≥ ' + A.minCount + ': ', keep.length ? keep.map(k => k + ':' + cc.get(k)).join(', ') : '(empty)', keep.length ? ' → each combined with ' + A.sel + ' is a frequent pattern.' : ' → only {' + A.sel + '} itself.'));
      }
      function mineAll() {
        const db = dbSel(); if (!db.length) { cmp.innerHTML = ''; return; } const fp = fpGrowth(db, A.minCount), ap = apriori(db, A.minCount / db.length), same = fp.length === ap.freq.size && fp.every(p => ap.freq.has(setKey(p.set)) && ap.freq.get(setKey(p.set)).count === p.count);
        cmp.innerHTML = ''; cmp.appendChild(h('div', { class: 'note', style: 'border-color:' + (same ? 'var(--good)' : 'var(--bad)') }, same ? `FP-Growth found ${fp.length} frequent patterns — identical (sets and counts) to Apriori from T11.` : 'FP-Growth and Apriori disagree')); cmp.appendChild(Ui.table(['pattern', 'count'], fp.sort((a, b) => b.count - a.count || a.set.length - b.set.length).map(p => [chips(p.set.slice().sort()), p.count]), { maxH: '220px' }));
      }
      const dsSeg = Ui.seg([{ v: true, label: 'My baskets (T10)' }, { v: false, label: 'Textbook example (9 tx)' }], true, v => { A.useShared = v; A.sel = ''; rebuildA(0); });
      const sMin = Ui.slider({ label: 'min support count', min: 1, max: 5, step: 1, value: A.minCount, on: v => { A.minCount = v; rebuildA(0); } });
      const partA = h('div', null, Ui.card('FP-tree', h('div', { class: 'row', style: 'margin-bottom:8px' }, Ui.btn('Next transaction', () => { if (A.pos < dbSel().length) rebuildA(A.pos + 1); }, 'pri'), Ui.btn('Insert all', () => rebuildA(dbSel().length)), Ui.btn('Reset', () => rebuildA(0)), h('span', { class: 'sp' }), dsSeg, h('div', { style: 'width:230px' }, sMin)), cvT.c, info), Ui.grid('g2', Ui.card('Transactions', txBox), h('div', null, Ui.card('Mining with FP-Growth', h('div', { class: 'row' }, h('span', { class: 'muted' }, 'inspect item'), itemSel), mineBox), Ui.card('All frequent patterns', cmp))));
      /* ---------- Part B ---------- */
      const B = { minSup: 0.15, seed: 3, split: 0.3 }; let data, edges, items, pats, Xp, clf, base1, base2, accs = {}, probe = [5.8, 3.0, 4.3, 1.3];
      const genIris = () => { const r = new RNG(B.seed * 101 + 7), X = [], y = []; for (let c = 0; c < 3; c++) for (let i = 0; i < 50; i++) { X.push(IRIS[c].map(([m, s]) => Math.max(0.1, m + r.normal() * s))); y.push(c); } return { X, y }; };
      const itemsOf = x => x.map((v, j) => FNAMES[j] + '=' + (v <= edges[j][0] ? 'low' : v <= edges[j][1] ? 'med' : 'high'));
      const qv = (arr, p) => { const a = arr.slice().sort((x, y) => x - y), i = p * (a.length - 1), lo = Math.floor(i), hi = Math.ceil(i); return a[lo] + (a[hi] - a[lo]) * (i - lo); };
      const featOf = its => { const s = new Set(its); return pats.map(p => (p.set.every(i => s.has(i)) ? 1 : 0)); };
      const resB = h('div'), patTbl = h('div'), probeBox = h('div'), probeCv = Ui.canvas(420, 140); const cvAcc = Ui.canvas(420, 160);
      function trainB() {
        data = genIris(); edges = [0, 1, 2, 3].map(j => { const col = data.X.map(x => x[j]); return [qv(col, 0.33), qv(col, 0.66)]; }); items = data.X.map(itemsOf); const n = items.length, minCount = Math.ceil(B.minSup * n);
        pats = fpGrowth(items, minCount).filter(p => p.set.length >= 2); Xp = items.map(featOf);
        const r = new RNG(B.seed), tr = [], te = []; for (let c = 0; c < 3; c++) { const idx = r.shuffle(Array.from({ length: 50 }, (_, i) => c * 50 + i)), k = Math.round(50 * B.split); te.push(...idx.slice(0, k)); tr.push(...idx.slice(k)); }
        const mean = [0, 1, 2, 3].map(j => data.X.reduce((s, x) => s + x[j], 0) / n), sd = [0, 1, 2, 3].map(j => Math.sqrt(data.X.reduce((s, x) => s + (x[j] - mean[j]) ** 2, 0) / n)), std = x => x.map((v, j) => (v - mean[j]) / sd[j]);
        clf = new ML.Logistic(pats.length, 3, { lr: 0.5, l2: 1e-3 }).fit(tr.map(i => Xp[i]), tr.map(i => data.y[i]), 300);
        const oneHot = its => { const s = new Set(its), names = []; FNAMES.forEach(f => ['low', 'med', 'high'].forEach(l => names.push(f + '=' + l))); return names.map(nm => (s.has(nm) ? 1 : 0)); };
        base1 = new ML.Logistic(4, 3, { lr: 0.5, l2: 1e-3 }).fit(tr.map(i => std(data.X[i])), tr.map(i => data.y[i]), 300); base2 = new ML.Logistic(12, 3, { lr: 0.5, l2: 1e-3 }).fit(tr.map(i => oneHot(items[i])), tr.map(i => data.y[i]), 300);
        const acc = (m, f) => te.reduce((s, i) => s + (m.predict(f(i)) === data.y[i] ? 1 : 0), 0) / te.length; accs = { pat: acc(clf, i => Xp[i]), raw: acc(base1, i => std(data.X[i])), oh: acc(base2, i => oneHot(items[i])), n: te.length }; accs.mean = mean; accs.sd = sd;
        resB.innerHTML = ''; resB.appendChild(h('div', { class: 'row' }, Ui.stat('mined patterns (size ≥2)', pats.length), Ui.stat('train / test', (tr.length) + ' / ' + te.length), Ui.stat('pattern features', (accs.pat * 100).toFixed(1) + '%'), Ui.stat('raw numeric', (accs.raw * 100).toFixed(1) + '%'), Ui.stat('items only (12 binary)', (accs.oh * 100).toFixed(1) + '%')));
        const dist = pats.map(p => { const cnt = [0, 0, 0]; items.forEach((its, i) => { if (Xp[i][pats.indexOf(p)]) cnt[data.y[i]]++; }); const tot = cnt[0] + cnt[1] + cnt[2]; const mx = Math.max(...cnt); return { p, cnt, tot, pur: mx / tot, cls: cnt.indexOf(mx) }; }).sort((a, b) => b.pur - a.pur || b.tot - a.tot);
        patTbl.innerHTML = ''; patTbl.appendChild(Ui.table(['pattern', 'covers', 'class split (set/ver/vir)', 'purity'], dist.slice(0, 25).map(d => [h('span', { class: 'mono', style: 'font-size:11.5px' }, d.p.set.map(s => s.replace('sepal ', 's.').replace('petal ', 'p.').replace('length', 'len').replace('width', 'wid')).join(' & ')), d.tot, d.cnt.join(' / '), h('b', { style: 'color:' + Colors.classColor(d.cls) }, (d.pur * 100).toFixed(0) + '% ' + CLS[d.cls])]), { maxH: '300px' }));
        drawAcc(); runProbe();
      }
      function drawAcc() { const { g, w, h: H } = cvAcc; g.clearRect(0, 0, w, H); [['pattern features', accs.pat, '#a78bc4'], ['raw numeric', accs.raw, '#7aa2c8'], ['12 binary items', accs.oh, '#86b894']].forEach(([n, v, c], i) => { const y = 14 + i * 46; g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.textAlign = 'left'; g.fillText(n, 8, y + 14); g.fillStyle = c; g.fillRect(120, y, (w - 190) * v, 24); g.fillStyle = '#fff'; g.font = '12px "JetBrains Mono",monospace'; g.fillText((v * 100).toFixed(1) + '%', 126 + (w - 190) * v, y + 17); }); }
      function runProbe() {
        const its = itemsOf(probe), f = featOf(its), p = clf.proba(f), fired = pats.filter((_, i) => f[i]); probeBox.innerHTML = ''; probeBox.appendChild(h('div', null, 'Items: ', chips(its.map(s => s.replace(/^(\w+) (\w+)=/, '$1 $2=')))));
        probeBox.appendChild(h('div', { class: 'muted', style: 'font-size:12.5px;margin-top:4px' }, `${fired.length} of ${pats.length} mined patterns fire for this flower.`));
        const { g, w, h: H } = probeCv; g.clearRect(0, 0, w, H); p.forEach((v, i) => { const y = 14 + i * 40; g.fillStyle = '#9b9384'; g.font = '12px "Instrument Sans"'; g.textAlign = 'left'; g.fillText(CLS[i], 8, y + 14); g.fillStyle = Colors.classColor(i); g.fillRect(90, y, (w - 150) * v, 22); g.fillStyle = '#fff'; g.font = '12px "JetBrains Mono",monospace'; g.fillText((v * 100).toFixed(0) + '%', 96 + (w - 150) * v, y + 16); });
      }
      const ranges = [[4.3, 7.9], [2.0, 4.4], [1.0, 6.9], [0.1, 2.5]];
      const probeSliders = FNAMES.map((nm, j) => Ui.slider({ label: nm + ' (cm)', min: ranges[j][0], max: ranges[j][1], step: 0.1, value: probe[j], on: v => { probe[j] = v; if (clf) runProbe(); } }));
      const sSup = Ui.slider({ label: 'min support (fraction)', min: 0.05, max: 0.5, step: 0.05, value: B.minSup, on: v => { B.minSup = v; trainB(); } });
      const partB = Ui.grid('g2', h('div', null, Ui.card('Pattern-feature classifier', Ui.row(sSup, Ui.slider({ label: 'random seed (data + split)', min: 1, max: 12, step: 1, value: B.seed, on: v => { B.seed = v; trainB(); } })), h('div', { style: 'margin-top:10px' }, resB), cvAcc.c), Ui.card('Flower probe', ...probeSliders, probeBox, probeCv.c)),
        Ui.card('Most class-pure frequent patterns (“class association rules”)', patTbl));
      root.appendChild(h('h3', { style: 'margin:0 0 10px' }, 'Part A — building and mining an FP-tree')); root.appendChild(partA);
      root.appendChild(h('h3', { style: 'margin:24px 0 10px' }, 'Part B — frequent patterns as features for classification (Iris-like data)')); root.appendChild(partB);
      rebuildA(0); setTimeout(trainB, 20); ctx.raf(drawTree);
    },
  });
})();
