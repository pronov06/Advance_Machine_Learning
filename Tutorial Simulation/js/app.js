/* AML Lab — shell: masthead, chapter ribbon, contents index, editorial pages, concept checks */
(function () {
  'use strict';
  const UNITS = {
    1: { name: 'Reinforcement Learning', roman: 'I', color: '#d9825b', blurb: 'Agents that learn from reward — tabular Q-learning, deep Q-networks, policy gradients and dynamic programming.' },
    2: { name: 'Machine Learning with Graphs', roman: 'II', color: '#6bb3aa', blurb: 'Message-passing networks, knowledge graphs with SPARQL, and frequent subgraph mining.' },
    3: { name: 'Ensemble Learning', roman: 'III', color: '#d7b56d', blurb: 'Boosting, bagging, the bias–variance trade-off, and recognising faces with ensembles.' },
    4: { name: 'Association Rule Mining', roman: 'IV', color: '#a78bc4', blurb: 'Market baskets, Apriori, FP-Growth, and patterns that learn to classify.' },
    5: { name: 'Neural Networks', roman: 'V', color: '#7aa2c8', blurb: 'Optimisers, back-propagation, convolutional face recognition and the perceptron.' },
  };
  const R = n => String(n).padStart(2, '0');
  const tasks = () => Lab.tasks.slice().sort((a, b) => a.unit - b.unit || a.order - b.order);
  const numLabel = t => t.num[0] + R(t.num.slice(1));
  let ctxStack = null, loops = [], lastT = 0;

  const store = {
    get(k = 'amllab.quiz') { try { return JSON.parse(localStorage.getItem(k) || '{}'); } catch (e) { return {}; } },
    set(id, v, k = 'amllab.quiz') { try { const s = this.get(k); s[id] = v; localStorage.setItem(k, JSON.stringify(s)); } catch (e) { /* storage unavailable */ } },
  };

  /* ---------- animation loop & per-page context ---------- */
  function frame(t) {
    const dt = Math.min(0.05, (t - lastT) / 1000 || 0.016); lastT = t;
    for (const l of loops.slice()) { try { l(dt, t / 1000); } catch (e) { console.error(e); const i = loops.indexOf(l); if (i >= 0) loops.splice(i, 1); } }
    requestAnimationFrame(frame);
  }
  function makeCtx() {
    const clean = [];
    return {
      raf(fn) { loops.push(fn); clean.push(() => { const i = loops.indexOf(fn); if (i >= 0) loops.splice(i, 1); }); return () => { const i = loops.indexOf(fn); if (i >= 0) loops.splice(i, 1); }; },
      timer(fn, ms) { const id = setInterval(fn, ms); clean.push(() => clearInterval(id)); return () => clearInterval(id); },
      cleanup(fn) { clean.push(fn); },
      on(target, ev, fn, o) { target.addEventListener(ev, fn, o); clean.push(() => target.removeEventListener(ev, fn, o)); },
      dispose() { clean.splice(0).forEach(f => { try { f(); } catch (e) { console.error(e); } }); },
    };
  }

  /* ---------- masthead pieces ---------- */
  function passed(t) { const d = store.get()[t.id]; return !!(d && d.done); }
  function updateMeter() {
    const main = tasks().filter(t => !t.bonus), n = main.filter(passed).length;
    document.getElementById('meterTxt').textContent = R(n) + ' / ' + R(main.length);
    document.getElementById('meterBar').style.width = (n / main.length * 100) + '%';
  }
  function buildRibbon(cur) {
    const rb = document.getElementById('ribbon'); rb.innerHTML = '';
    tasks().forEach(t => rb.appendChild(h('a', { href: '#/' + t.id, class: 'u' + t.unit + (t.bonus ? ' bonus' : '') + (passed(t) ? ' done' : '') + (cur && cur.id === t.id ? ' on' : ''), 'data-t': numLabel(t) + '  ' + t.title, 'aria-label': t.title })));
  }
  function buildTOC(cur) {
    const el = document.querySelector('#toc .inner'); el.innerHTML = '';
    el.appendChild(h('div', { class: 'top' }, h('a', { class: 'brand', href: '#/' }, 'AML ', h('i', null, 'Lab')), h('button', { class: 'pillbtn', onclick: closeTOC }, 'Close   Esc')));
    const cols = h('div', { class: 'cols' });
    for (const u of [1, 2, 3, 4, 5]) {
      const col = h('div', { class: 'col', style: '--uc:' + UNITS[u].color }, h('div', { class: 'num' }, UNITS[u].roman), h('h3', null, UNITS[u].name), h('p', null, UNITS[u].blurb));
      tasks().filter(t => t.unit === u).forEach(t => col.appendChild(h('a', { class: 'it' + (cur && cur.id === t.id ? ' cur' : ''), href: '#/' + t.id }, h('b', null, numLabel(t)), h('span', null, t.title), passed(t) ? h('i', null, 'done') : null)));
      cols.appendChild(col);
    }
    el.appendChild(cols);
  }
  const openTOC = () => { const t = document.getElementById('toc'); t.classList.add('open'); t.setAttribute('aria-hidden', 'false'); document.body.style.overflow = 'hidden'; };
  const closeTOC = () => { const t = document.getElementById('toc'); t.classList.remove('open'); t.setAttribute('aria-hidden', 'true'); document.body.style.overflow = ''; };

  /* ---------- home ---------- */
  function home(page) {
    document.documentElement.style.setProperty('--accent', '#d7b56d');
    const main = tasks().filter(t => !t.bonus), bonus = tasks().length - main.length;
    page.appendChild(h('section', { class: 'hero enter' },
      h('div', null, h('div', { class: 'k' }, 'Advanced Machine Learning · 21CSE425T'), h('h1', null, 'Machine learning,', h('br'), h('em', null, 'in motion.')),
        h('p', { class: 'lead' }, 'Every tutorial from the five units of the course, rebuilt as a live simulation. Pause an agent mid-fall, rewire a graph, race six optimisers down a ravine. Each algorithm is written from scratch and runs in this page — nothing to install, nothing to train in Colab.')),
      h('div', { class: 'facts' }, h('div', null, h('b', null, R(main.length)), h('span', null, 'Tutorials')), h('div', null, h('b', null, R(bonus)), h('span', null, 'Bonus labs')), h('div', null, h('b', null, R(0)), h('span', null, 'Installs')))));
    page.appendChild(h('div', { class: 'secthead' }, h('h2', null, 'Contents'), h('span', { class: 'k' }, 'Five units · eighteen chapters')));
    for (const u of [1, 2, 3, 4, 5]) {
      const row = h('section', { class: 'unit', style: '--uc:' + UNITS[u].color }, h('div', { class: 'head' }, h('div', { class: 'num' }, UNITS[u].roman), h('h3', null, UNITS[u].name), h('p', null, UNITS[u].blurb)));
      const list = h('div', null);
      tasks().filter(t => t.unit === u).forEach(t => list.appendChild(h('a', { class: 'chap' + (passed(t) ? ' done' : ''), href: '#/' + t.id },
        h('span', { class: 'n' }, numLabel(t)), h('div', null, h('h4', null, t.title, t.bonus ? h('span', { class: 'bonus' }, 'Bonus') : null), h('div', { class: 't' }, t.tag)),
        h('div', { class: 's' }, h('small', null, 'Novel here'), (t.star || '').split(' · ')[0]), h('span', { class: 'ar' }, '→'))));
      row.appendChild(list); page.appendChild(row);
    }
    page.appendChild(h('section', { class: 'colophon' },
      h('div', null, h('h5', null, 'How it maps'), h('p', null, 'Each chapter keeps the notebook’s task and defaults, then adds what a notebook cannot: sliders for every hyper-parameter, step-by-step execution, direct manipulation and “break it” switches.')),
      h('div', null, h('h5', null, 'Honest notes'), h('p', null, 'Datasets that need downloading (Olivetti faces, Iris, digits) are replaced by procedural stand-ins generated in the page. Each chapter says where it differs from the notebook.')),
      h('div', null, h('h5', null, 'Typography'), h('p', null, 'Set in Instrument Serif and Instrument Sans, with JetBrains Mono for mathematics. All typefaces are bundled and released under the SIL Open Font License.'))));
  }

  /* ---------- concept check ---------- */
  function quizPanel(t, tabBtn) {
    const wrap = h('div', { class: 'quiz' }); let correct = 0, answered = 0;
    const result = h('div', { class: 'muted', style: 'margin-top:6px' });
    const saved = store.get()[t.id];
    if (saved && saved.done) result.innerHTML = 'You passed this check earlier (' + saved.score + ' of ' + t.quiz.length + '). Answer again any time.';
    t.quiz.forEach(q => {
      const why = h('div', { class: 'why', style: 'display:none' }), opts = [], box = h('div', { class: 'q' }, h('div', { class: 'qq' }, q.q));
      q.o.forEach((txt, i) => {
        const b = h('button', {
          class: 'opt', onclick: () => {
            if (box.dataset.done) return; box.dataset.done = 1; answered++;
            opts.forEach((o, j) => { if (j === q.a) o.classList.add('ok'); else if (j === i) o.classList.add('no'); });
            if (i === q.a) correct++; why.style.display = 'block'; why.innerHTML = q.why;
            if (answered === t.quiz.length) {
              const ok = correct >= Math.ceil(t.quiz.length * 0.66); result.innerHTML = '';
              result.appendChild(h('div', { class: 'score' }, correct + ' / ' + t.quiz.length)); result.appendChild(h('div', null, ok ? 'Well done — chapter marked as checked.' : 'Not quite there yet. Re-read the briefing, play with the workbench, and try again.'));
              if (ok) { store.set(t.id, { done: true, score: correct }); updateMeter(); buildRibbon(t); buildTOC(t); if (!tabBtn.querySelector('.ok')) tabBtn.appendChild(h('span', { class: 'ok' })); }
            }
          },
        }, h('span', null, txt));
        opts.push(b); box.appendChild(b);
      });
      box.appendChild(why); wrap.appendChild(box);
    });
    wrap.appendChild(result); return wrap;
  }

  /* ---------- chapter page ---------- */
  function chapter(t, page) {
    const u = UNITS[t.unit], all = tasks(), i = all.indexOf(t); document.documentElement.style.setProperty('--accent', u.color);
    page.appendChild(h('header', { class: 'chead enter' },
      h('div', { class: 'kick' }, h('b', null, numLabel(t)), h('i'), h('span', null, 'Unit ' + u.roman + ' — ' + u.name), t.bonus ? h('span', { class: 'pill' }, 'Bonus') : null),
      h('h1', null, t.title), h('p', { class: 'deck' }, t.tag),
      t.star ? h('div', { class: 'k novel-k' }, 'Novel in this lab') : null,
      t.star ? h('ul', { class: 'novel' }, t.star.split(' · ').map(x => h('li', null, x))) : null));

    /* briefing: tabs, open on first visit */
    const seen = store.get('amllab.seen'); const firstVisit = !seen[t.id]; store.set(t.id, 1, 'amllab.seen');
    const tabsDef = [['idea', 'The idea', () => h('div', { class: 'body prose', html: t.idea })]];
    if (t.tryIt) tabsDef.push(['try', 'Try this', () => h('div', { class: 'body' }, h('ul', null, t.tryIt.map(x => h('li', { html: x }))))]);
    if (t.hood) tabsDef.push(['math', 'Mathematics', () => h('div', { class: 'body', html: t.hood })]);
    const tabBtns = {}, panels = {}; let current = firstVisit ? 'idea' : null;
    const tabs = h('div', { class: 'tabs', role: 'tablist' }), brief = h('section', { class: 'brief enter' }, tabs);
    const open = id => {
      current = current === id ? null : id;
      Object.keys(panels).forEach(k => { panels[k].classList.toggle('on', k === current); tabBtns[k].classList.toggle('on', k === current); });
      toggle.textContent = current ? 'Hide briefing' : 'Show briefing';
    };
    const addTab = (id, label, build) => { const b = h('button', { role: 'tab', onclick: () => open(id) }, label); tabBtns[id] = b; tabs.appendChild(b); const p = h('div', { class: 'panel' }, build(b)); panels[id] = p; brief.appendChild(p); };
    tabsDef.forEach(([id, label, build]) => addTab(id, label, build));
    if (t.quiz && t.quiz.length) addTab('check', 'Check yourself', b => quizPanel(t, b));
    if (passed(t)) tabBtns.check.appendChild(h('span', { class: 'ok' }));
    const toggle = h('button', { class: 'toggle', onclick: () => open(current || 'idea') }, current ? 'Hide briefing' : 'Show briefing'); tabs.appendChild(h('span', { class: 'sp' })); tabs.appendChild(toggle);
    if (current) { panels[current].classList.add('on'); tabBtns[current].classList.add('on'); }
    page.appendChild(brief);

    page.appendChild(h('div', { class: 'benchhead' }, h('span', { class: 'k' }, 'Workbench'), h('i'), h('span', null, numLabel(t))));
    const sim = h('div', { id: 'sim', class: 'enter' }); page.appendChild(sim);
    ctxStack = makeCtx();
    try { t.mount(sim, ctxStack); } catch (e) {
      console.error(e); sim.appendChild(h('div', { class: 'card', style: 'border-color:var(--bad)' }, h('b', { class: 'bad' }, 'This simulation crashed: '), h('pre', null, String(e && e.stack || e))));
    }
    const prev = all[i - 1], next = all[i + 1];
    page.appendChild(h('footer', { class: 'cfoot' },
      prev ? h('a', { class: 'prev', href: '#/' + prev.id }, h('small', null, 'Previous · ' + numLabel(prev)), h('span', null, prev.title)) : h('span'),
      next ? h('a', { class: 'next', href: '#/' + next.id }, h('small', null, 'Next · ' + numLabel(next)), h('span', null, next.title)) : h('a', { class: 'next', href: '#/' }, h('small', null, 'Finished'), h('span', null, 'Back to the contents'))));
  }

  function route() {
    if (ctxStack) { ctxStack.dispose(); ctxStack = null; }
    closeTOC();
    const id = (location.hash.replace(/^#\/?/, '') || '').trim(), pg = document.getElementById('page'); pg.innerHTML = '';
    const t = tasks().find(x => x.id === id);
    if (t) chapter(t, pg); else home(pg);
    const cr = document.getElementById('crumb'); cr.innerHTML = ''; if (t) cr.appendChild(h('span', null, 'Unit ' + UNITS[t.unit].roman + '  /  ', h('b', null, numLabel(t) + '  ' + t.title))); else cr.textContent = 'Interactive simulations for every tutorial';
    buildRibbon(t); buildTOC(t); updateMeter(); window.scrollTo(0, 0);
    document.title = (t ? t.title + ' — ' : '') + 'AML Lab';
  }

  Lab.UNITS = UNITS;
  window.addEventListener('hashchange', route);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeTOC();
    const tag = (e.target.tagName || '').toLowerCase(); if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    const all = tasks(), i = all.findIndex(x => x.id === location.hash.replace(/^#\/?/, ''));
    if (e.key === ']' && i < all.length - 1) location.hash = '#/' + all[i + 1].id; if (e.key === '[' && i > 0) location.hash = '#/' + all[i - 1].id; if (e.key === 'c' && !e.ctrlKey && !e.metaKey) openTOC();
  });
  function start() {
    document.getElementById('tocBtn').onclick = openTOC; route(); requestAnimationFrame(frame);
  }
  const fontsReady = (document.fonts && document.fonts.load) ? Promise.race([Promise.all(['400 16px "Instrument Sans"', '500 16px "Instrument Sans"', '400 40px "Instrument Serif"', 'italic 400 40px "Instrument Serif"', '400 12px "JetBrains Mono"'].map(f => document.fonts.load(f))), new Promise(r => setTimeout(r, 1800))]) : Promise.resolve();
  const boot = () => fontsReady.then(start, start);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
