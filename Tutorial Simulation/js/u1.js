/* Unit 1 — Reinforcement Learning: T1 chatbot, T2 DQN, T3 REINFORCE, B1 GridWorld */
(function () {
  'use strict';
  const { RNG } = ML;

  /* ============================================================ T1 ====== */
  Lab.register({
    id: 't1', unit: 1, num: 'T1', title: 'Q-learning chatbot',
    tag: 'A chatbot that learns which reply to give purely from reward — then lets YOU be the reward signal.',
    star: 'Human-feedback training (RLHF in miniature) · teach it new intents live',
    idea: `<p>Frame a tiny intent bot as RL: the <b>state</b> is the user’s intent, the <b>action</b> is which reply the bot picks, the <b>reward</b> is +1 if the reply fits and −1 if not. Tabular Q-learning learns <code>Q(intent, reply)</code> with no labelled (intent → reply) pairs — only reward.</p>
    <p>The notebook’s reward comes from a hard-coded rule. Here you can swap it for <b>your own ratings</b> (the same idea behind RLHF), corrupt it with noise, flip the “correct” answers mid-training and watch the Q-table re-adapt, or <b>teach the bot a brand-new intent</b> and see the state/action space grow.</p>`,
    tryIt: [
      'Press <b>Train</b> and watch the Q-table heat-map turn from grey into a bright diagonal (the dot marks the truly correct reply).',
      'Raise <b>reward noise</b> to 40% – the agent still learns, just slower: that is why Q-learning averages with α.',
      'Hit <b>Swap two answers</b> while trained: the policy is suddenly wrong, then re-learns. This is a <i>non-stationary</i> environment; try a larger ε and α.',
      'Tick <b>Learn from my ratings</b>, reset Q, then chat and rate replies yourself. With ε-greedy exploration the bot tries other replies and you shape it manually.',
      'Add a new intent such as <code>joke</code> with examples <code>tell me a joke, make me laugh</code> and train again.'],
    hood: `<p>Update (single-step / bandit episode, next state terminal so the target is just <code>r</code>):</p>
    <pre>Q[s,a] ← Q[s,a] + α · ( r + γ·0 − Q[s,a] )        a = ε-greedy(Q[s,:])</pre>
    <p>Intent recognition is the notebook’s bag-of-words cosine match against a per-intent prototype vector; only the <i>reply choice</i> is learned by RL.</p>
    <p>With constant α and noisy ±1 rewards, <code>Q[s,a]</code> converges to <code>α</code>-weighted average reward: ≈ (1−2·noise) for the right reply and ≈ −(1−2·noise) for wrong ones, so the arg-max is still correct for noise &lt; 50%.</p>`,
    quiz: [
      { q: 'Why does the notebook’s Q-update use r + γ·0?', o: ['γ is always 0 in Q-learning', 'Each chat turn is a one-step episode: the next state is terminal', 'The reward is negative', 'To make the table smaller'], a: 1, why: 'There is no future state to bootstrap from, so the TD target collapses to the immediate reward (a contextual bandit).' },
      { q: 'Reward noise of 60% (reward sign flipped 60% of the time) will most likely…', o: ['slow learning slightly', 'make the agent learn the opposite policy', 'change nothing', 'crash the table'], a: 1, why: 'With >50% flips the expected reward of the right reply is negative, so the agent confidently learns the wrong thing.' },
      { q: 'What does ε control?', o: ['learning speed', 'how much the future matters', 'how often a random reply is tried', 'the size of the table'], a: 2, why: 'ε-greedy: with probability ε pick a random action (explore), otherwise the arg-max (exploit).' }],
    mount(root, ctx) {
      const rng = new RNG(7);
      const S = [
        { name: 'greeting', ex: ['hi', 'hello', 'hey there', 'good morning', 'howdy'], reply: 'Hello! How can I help you today?' },
        { name: 'thanks', ex: ['thanks', 'thank you', 'much appreciated', 'thanks a lot'], reply: "You're welcome! Happy to help." },
        { name: 'help', ex: ['can you help', 'i need help', 'support please', 'help me'], reply: 'Sure — tell me what you need help with.' },
        { name: 'bye', ex: ['bye', 'goodbye', 'see you', 'catch you later'], reply: 'Goodbye! Have a great day.' },
        { name: 'weather', ex: ['what is the weather', 'is it raining', 'weather today', 'forecast'], reply: "I can't fetch live weather, but it's always sunny in here." },
      ];
      let Q, correct, vocab, proto, episodes = 0, recent = [], hist = { acc: [], rew: [] }, running = false, lastCell = null, flash = 0;
      const P = { alpha: 0.5, eps: 0.2, noise: 0, speed: 40, human: false };
      const n = () => S.length;
      function reset() { Q = S.map(() => new Float64Array(S.length)); correct = S.map((_, i) => i); episodes = 0; recent = []; hist = { acc: [], rew: [] }; lastCell = null; }
      function rebuild() {
        const set = new Set(); S.forEach(s => s.ex.forEach(p => p.split(/\s+/).forEach(w => set.add(w)))); vocab = [...set].sort();
        proto = S.map(s => { const v = new Float64Array(vocab.length); s.ex.forEach(p => { const x = vec(p); for (let i = 0; i < v.length; i++) v[i] += x[i] / s.ex.length; }); return v; });
      }
      function vec(t) { const v = new Float64Array(vocab.length); t.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(/\s+/).forEach(w => { const i = vocab.indexOf(w); if (i >= 0) v[i]++; }); return v; }
      function classify(text) {
        const v = vec(text); const nv = Math.hypot(...v); if (!nv) return { i: -1, sc: [] };
        const sc = proto.map(p => { let d = 0; for (let i = 0; i < v.length; i++) d += v[i] * p[i]; return d / (nv * Math.hypot(...p) + 1e-9); });
        let b = 0; sc.forEach((x, i) => { if (x > sc[b]) b = i; }); return { i: sc[b] > 0 ? b : -1, sc };
      }
      const greedy = s => { let b = 0; for (let a = 1; a < n(); a++) if (Q[s][a] > Q[s][b]) b = a; return b; };
      function trainOne() {
        const s = rng.int(n()); const a = rng.rand() < P.eps ? rng.int(n()) : greedy(s);
        let r = a === correct[s] ? 1 : -1; if (rng.rand() < P.noise) r = -r;
        Q[s][a] += P.alpha * (r - Q[s][a]); episodes++; recent.push(r); if (recent.length > 200) recent.shift(); lastCell = [s, a]; flash = 1;
        if (episodes % 20 === 0) { hist.acc.push([episodes, accuracy() * 100]); hist.rew.push([episodes, recent.reduce((x, y) => x + y, 0) / recent.length * 100]); }
      }
      const accuracy = () => S.reduce((c, _, s) => c + (greedy(s) === correct[s] ? 1 : 0), 0) / n();
      reset(); rebuild();

      /* ---------- Q-table canvas ---------- */
      const hm = Ui.canvas(560, 400);
      function drawQ() {
        const { g, w, h: H } = hm; g.clearRect(0, 0, w, H);
        const N = n(), L = 92, T = 66, cw = (w - L - 8) / N, ch = Math.min(56, (H - T - 8) / N);
        g.font = '11.5px "Instrument Sans"'; g.fillStyle = '#9b9384'; g.textAlign = 'center';
        g.fillText('reply the bot picks (action) →', L + (w - L) / 2, 14);
        S.forEach((s, a) => { g.save(); g.translate(L + a * cw + cw / 2, T - 6); g.rotate(-0.5); g.textAlign = 'left'; g.fillStyle = '#cfc8b8'; g.fillText(s.name, 0, 0); g.restore(); });
        g.textAlign = 'right'; g.fillStyle = '#9b9384'; g.fillText('intent (state) ↓', L - 6, T - 28);
        for (let s = 0; s < N; s++) {
          g.fillStyle = '#cfc8b8'; g.textAlign = 'right'; g.fillText(S[s].name, L - 8, T + s * ch + ch / 2 + 4);
          for (let a = 0; a < N; a++) {
            const x = L + a * cw, y = T + s * ch;
            g.fillStyle = Colors.diverge(Q[s][a]); g.fillRect(x + 1, y + 1, cw - 2, ch - 2);
            g.fillStyle = '#fff'; g.textAlign = 'center'; g.font = '12px "JetBrains Mono",monospace'; g.fillText(Q[s][a].toFixed(2), x + cw / 2, y + ch / 2 + 4); g.font = '11.5px "Instrument Sans"';
            if (a === greedy(s)) { g.strokeStyle = '#fff'; g.lineWidth = 2; g.strokeRect(x + 2, y + 2, cw - 4, ch - 4); }
            if (a === correct[s]) { g.fillStyle = '#86b894'; g.beginPath(); g.arc(x + 9, y + 9, 4, 0, 7); g.fill(); }
          }
        }
        if (lastCell && flash > 0) { const [s, a] = lastCell; g.strokeStyle = `rgba(215,181,109,${flash})`; g.lineWidth = 3; g.strokeRect(L + a * cw, T + s * ch, cw, ch); flash *= 0.9; }
        g.textAlign = 'left'; g.fillStyle = '#9b9384'; g.fillText('white box = greedy action   ● green dot = truly correct reply   colour = Q value (−1 … +1)', 8, H - 8);
      }
      const chartCv = Ui.canvas(560, 200), chart = new LineChart(chartCv, { xlabel: 'training episodes', yMin: -100, yMax: 100 });

      /* ---------- controls ---------- */
      const stEp = Ui.stat('episodes'), stAcc = Ui.stat('policy correct'), stRew = Ui.stat('avg reward (200)');
      const trainBtn = Ui.btn('Train', () => { running = !running; trainBtn.textContent = running ? 'Pause' : 'Train'; }, 'pri');
      const ctl = Ui.card('Training', Ui.row(trainBtn, Ui.btn('Reset Q', () => { reset(); drawQ(); }), Ui.btn('Swap two answers', () => {
        if (n() < 2) return; const a = rng.int(n()); let b = rng.int(n() - 1); if (b >= a) b++; [correct[a], correct[b]] = [correct[b], correct[a]]; Ui.toast(`“${S[a].name}” and “${S[b].name}” now want each other's reply`);
      })),
        Ui.row(Ui.slider({ label: 'learning rate α', min: 0.05, max: 1, step: 0.05, value: P.alpha, on: v => P.alpha = v }), Ui.slider({ label: 'exploration ε', min: 0, max: 1, step: 0.01, value: P.eps, on: v => P.eps = v })),
        Ui.row(Ui.slider({ label: 'reward noise (sign flips)', min: 0, max: 0.8, step: 0.05, value: 0, fmt: v => Math.round(v * 100) + '%', on: v => P.noise = v }), Ui.slider({ label: 'speed (episodes/frame)', min: 1, max: 200, step: 1, value: P.speed, on: v => P.speed = v })),
        Ui.row(stEp, stAcc, stRew));

      /* ---------- chat ---------- */
      const chat = h('div', { class: 'chat' }), input = h('input', { type: 'text', placeholder: 'say something… (hello, thanks a lot, i need help, see you, is it raining)', style: 'flex:1' });
      const qbars = Ui.canvas(420, 130);
      let lastState = -1;
      function drawBars() {
        const { g, w, h: H } = qbars; g.clearRect(0, 0, w, H); g.font = '11.5px "Instrument Sans"'; g.fillStyle = '#9b9384';
        if (lastState < 0) { g.fillText('Q-values of the last recognised intent appear here', 10, 22); return; }
        g.fillText('Q( ' + S[lastState].name + ' , reply )', 10, 16); const N = n(), bw = (w - 20) / N;
        for (let a = 0; a < N; a++) { const q = Q[lastState][a], x = 10 + a * bw, mid = 78; g.fillStyle = q >= 0 ? '#86b894' : '#cf6a76'; const hh = Math.abs(q) * 40; g.fillRect(x + 6, q >= 0 ? mid - hh : mid, bw - 12, hh); g.fillStyle = '#9b9384'; g.textAlign = 'center'; g.fillText(S[a].name, x + bw / 2, 122); }
        g.textAlign = 'left';
      }
      function say(cls, text, extra) { const m = h('div', { class: 'msg ' + cls }, text, extra); chat.appendChild(m); chat.scrollTop = chat.scrollHeight; return m; }
      function send() {
        const t = input.value.trim(); if (!t) return; input.value = ''; say('u', t);
        const c = classify(t); if (c.i < 0) { say('b', "I don't recognise that intent yet — teach me below!"); return; }
        lastState = c.i; const explore = fb.get() && rng.rand() < P.eps; const a = explore ? rng.int(n()) : greedy(c.i);
        const small = h('small', null, `intent = ${S[c.i].name} (${(c.sc[c.i] * 100).toFixed(0)}% match) · action = reply#${a}${explore ? ' · exploring' : ''}`);
        const extra = [small];
        if (fb.get()) {
          const rate = r => { Q[c.i][a] += P.alpha * (r - Q[c.i][a]); episodes++; up.disabled = dn.disabled = true; Ui.toast('Q(' + S[c.i].name + ', #' + a + ') updated with your reward ' + r); drawQ(); drawBars(); };
          const up = Ui.btn('Approve', () => rate(1), 'sm'), dn = Ui.btn('Reject', () => rate(-1), 'sm'); extra.push(h('div', { style: 'margin-top:5px;display:flex;gap:6px' }, up, dn));
        }
        say('b', S[a].reply, extra); drawBars();
      }
      input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });
      const fb = Ui.chk('Learn from my ratings (human feedback, ε-greedy replies)', false, () => { });
      const chatCard = Ui.card('Chat with the learned policy', chat, h('div', { class: 'row', style: 'margin-top:8px' }, input, Ui.btn('Send', send, 'pri')), h('div', { style: 'margin-top:8px' }, fb), h('div', { style: 'margin-top:8px' }, qbars.c));
      say('b', 'Hi! I start knowing nothing. Train me, or tick human feedback and rate my replies.');

      /* ---------- teach ---------- */
      const nm = h('input', { type: 'text', placeholder: 'intent name e.g. joke' }), ex = h('input', { type: 'text', placeholder: 'examples, comma separated' }), rp = h('input', { type: 'text', placeholder: 'the correct reply text' });
      const teach = Ui.card('Teach a new intent (state & action space grow)', h('div', { class: 'col' }, nm, ex, rp, Ui.btn('Add intent', () => {
        const name = nm.value.trim().toLowerCase().replace(/\s+/g, '_'), exs = ex.value.split(',').map(x => x.trim().toLowerCase()).filter(Boolean), rep = rp.value.trim();
        if (!name || !exs.length || !rep) { Ui.toast('Fill all three fields'); return; } if (S.some(s => s.name === name)) { Ui.toast('Intent exists'); return; } if (S.length >= 9) { Ui.toast('Max 9 intents'); return; }
        S.push({ name, ex: exs, reply: rep }); const old = Q; Q = S.map((_, i) => new Float64Array(S.length).map((__, j) => (i < old.length && j < old.length ? old[i][j] : 0))); correct.push(S.length - 1); rebuild(); drawQ(); Ui.toast('Added “' + name + '” — train again');
        nm.value = ex.value = rp.value = '';
      }, 'pri')));

      root.appendChild(Ui.grid('g2', h('div', null, Ui.card('Q-table  (rows: intent, columns: reply)', hm.c), ctl), h('div', null, chatCard, Ui.card('Learning curves', chartCv.c, Ui.legend([['#86b894', 'policy correct %'], ['#d7b56d', 'avg reward ×100']])), teach)));
      let k = 0;
      ctx.raf(() => {
        if (running) for (let i = 0; i < P.speed; i++) trainOne();
        drawQ(); if (k++ % 4 === 0) {
          chart.draw([{ name: 'policy correct %', color: '#86b894', pts: hist.acc }, { name: 'avg reward ×100', color: '#d7b56d', pts: hist.rew }]);
          stEp.set(episodes); stAcc.set(Math.round(accuracy() * 100) + '%'); stRew.set(recent.length ? (recent.reduce((a, b) => a + b, 0) / recent.length).toFixed(2) : '–'); if (lastState >= 0) drawBars();
        }
      });
      drawBars();
    },
  });

  /* ========================================================= CartPole kit */
  function drawCart(cv, s, o = {}) {
    const { g, w, h: H } = cv; g.clearRect(0, 0, w, H);
    const sc = (w - 80) / 4.8, gy = H * 0.74, cx = w / 2 + s[0] * sc, L = H * 0.46;
    g.fillStyle = '#100e0c'; g.fillRect(0, 0, w, H);
    g.strokeStyle = '#3a342b'; g.lineWidth = 3; g.beginPath(); g.moveTo(30, gy); g.lineTo(w - 30, gy); g.stroke();
    g.strokeStyle = '#cf6a76'; g.lineWidth = 2; for (const sx of [-2.4, 2.4]) { g.beginPath(); g.moveTo(w / 2 + sx * sc, gy - 22); g.lineTo(w / 2 + sx * sc, gy + 12); g.stroke(); }
    const lim = 12 * Math.PI / 180; g.fillStyle = 'rgba(134,184,148,.07)'; g.beginPath(); g.moveTo(cx, gy - 30); g.arc(cx, gy - 30, L * 1.05, -Math.PI / 2 - lim, -Math.PI / 2 + lim); g.closePath(); g.fill();
    const danger = Math.min(1, Math.abs(s[2]) / lim), col = `rgb(${Math.round(134 + 73 * danger)},${Math.round(184 - 78 * danger)},${Math.round(148 - 30 * danger)})`;
    g.fillStyle = '#4a4338'; g.fillRect(cx - 32, gy - 30, 64, 28); g.fillStyle = '#302b24'; g.beginPath(); g.arc(cx - 18, gy - 1, 6, 0, 7); g.arc(cx + 18, gy - 1, 6, 0, 7); g.fill();
    g.strokeStyle = col; g.lineWidth = 8; g.lineCap = 'round'; g.beginPath(); g.moveTo(cx, gy - 30); g.lineTo(cx + Math.sin(s[2]) * L, gy - 30 - Math.cos(s[2]) * L); g.stroke(); g.lineCap = 'butt';
    g.fillStyle = '#ece6d8'; g.beginPath(); g.arc(cx, gy - 30, 4, 0, 7); g.fill();
    if (o.pref !== undefined) { // preference arrow: o.pref in [-1,1] (>0 push right)
      g.strokeStyle = '#d7b56d'; g.fillStyle = '#d7b56d'; g.lineWidth = 4; const len = o.pref * 50; g.beginPath(); g.moveTo(cx, gy + 28); g.lineTo(cx + len, gy + 28); g.stroke();
      const d = Math.sign(len) || 1; g.beginPath(); g.moveTo(cx + len + d * 9, gy + 28); g.lineTo(cx + len, gy + 22); g.lineTo(cx + len, gy + 34); g.fill();
    }
    g.font = '12px "Instrument Sans"'; g.fillStyle = '#9b9384'; g.textAlign = 'left'; if (o.label) g.fillText(o.label, 10, 18); g.textAlign = 'right'; if (o.score !== undefined) { g.fillStyle = '#ece6d8'; g.font = '600 14px "JetBrains Mono",monospace'; g.fillText('steps ' + o.score, w - 10, 18); } g.textAlign = 'left';
    if (o.msg) { g.fillStyle = '#d7b56d'; g.font = '600 13px "Instrument Sans"'; g.textAlign = 'center'; g.fillText(o.msg, w / 2, H - 10); g.textAlign = 'left'; }
  }

  function makeTrainer(kind, seed, o = {}) {
    const rng = new RNG(seed), env = new ML.CartPole(rng);
    const agent = kind === 'dqn' ? new ML.DQN(Object.assign({ rng }, o)) : new ML.Reinforce(Object.assign({ rng }, o));
    const T = { kind, agent, env, obs: env.reset(), ret: 0, rewards: [], steps: 0, loss: 0 };
    T.tick = function () {
      const a = agent.act(T.obs), r = env.step(a);
      if (kind === 'dqn') { agent.remember(T.obs, a, 1, r.obs, r.terminated); const l = agent.learn(); if (l) T.loss = l; } else agent.record(T.obs, a, 1);
      T.obs = r.obs; T.ret++; T.steps++;
      if (r.terminated || r.truncated) { T.rewards.push(T.ret); T.ret = 0; agent.endEpisode(); T.obs = env.reset(); }
    };
    T.avg = (k = 20) => { const r = T.rewards.slice(-k); return r.length ? r.reduce((a, b) => a + b, 0) / r.length : 0; };
    T.pref = obs => { if (kind === 'dqn') { const q = agent.q(obs); return q[1] - q[0]; } const p = agent.probs(obs); return p[1] - p[0]; };
    T.run = ms => { const t0 = performance.now(); let c = 0; while (performance.now() - t0 < ms) { for (let i = 0; i < 20; i++) T.tick(); c += 20; } return c; };
    return T;
  }
  function curve(T, k = 20) { const r = T.rewards, pts = r.map((v, i) => [i + 1, v]); const ma = []; for (let i = k - 1; i < r.length; i++) { let s = 0; for (let j = i - k + 1; j <= i; j++) s += r[j]; ma.push([i + 1, s / k]); } return { pts, ma }; }

  // live greedy rollout + optional human control
  function makeDemo(T, seed) {
    const rng = new RNG(seed), env = new ML.CartPole(rng); let acc = 0, score = 0, best = 0, pause = 0, human = 0, kick = 0;
    const D = { env, get score() { return score; }, get best() { return best; }, humanMode: false, humanAct: 1, wind: 0 };
    D.advance = dt => {
      if (pause > 0) { pause -= dt; if (pause <= 0) { env.reset(); score = 0; } return; }
      acc += dt;
      while (acc >= 0.02) {
        acc -= 0.02; const obs = env.obs();
        const a = D.humanMode ? D.humanAct : T.agent.act(obs, true);
        if (D.wind && Math.random() < 0.04) env.s[3] += (Math.random() < 0.5 ? -1 : 1) * D.wind;
        const r = env.step(a); score++; if (r.terminated || r.truncated) { best = Math.max(best, score); pause = 0.7; D.lastMsg = r.truncated ? 'balanced 500 steps!' : 'fell'; break; }
      }
    };
    D.kick = () => { env.s[3] += (Math.random() < 0.5 ? -1.2 : 1.2); };
    D.reset = () => { env.reset(); score = 0; pause = 0; };
    return D;
  }

  function policyMap(cv, T, trail) {
    const { g, w, h: H } = cv, N = 36, m = { l: 40, b: 28, t: 10, r: 8 }, cw = (w - m.l - m.r) / N, ch = (H - m.t - m.b) / N; g.clearRect(0, 0, w, H);
    const TH = 0.21, TD = 2.2, vals = []; let mx = 1e-6;
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const th = -TH + (i + 0.5) / N * 2 * TH, td = TD - (j + 0.5) / N * 2 * TD;
      const v = T.pref(Float64Array.from([0, 0, th, td])); vals.push(v); mx = Math.max(mx, Math.abs(v));
    }
    if (T.kind === 'rf') mx = Math.max(mx, 1); else mx = Math.max(mx, 1e-3);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) { { const t = vals[j * N + i] / mx; g.fillStyle = Colors.diverge(Math.sign(t) * Math.pow(Math.abs(t), 0.5)); } g.fillRect(m.l + i * cw, m.t + j * ch, cw + 0.5, ch + 0.5); }
    g.strokeStyle = '#6f6859'; g.lineWidth = 1; g.beginPath(); g.moveTo(m.l + N * cw / 2, m.t); g.lineTo(m.l + N * cw / 2, m.t + N * ch); g.moveTo(m.l, m.t + N * ch / 2); g.lineTo(m.l + N * cw, m.t + N * ch / 2); g.stroke();
    g.fillStyle = '#9b9384'; g.font = '11px "Instrument Sans"'; g.textAlign = 'center'; g.fillText('pole angle θ  (−12° … +12°)', m.l + N * cw / 2, H - 8); g.save(); g.translate(12, m.t + N * ch / 2); g.rotate(-Math.PI / 2); g.fillText('angular velocity θ̇', 0, 0); g.restore();
    for (let k = 0; k < trail.length; k++) { const [th, td] = trail[k]; const x = m.l + (th + TH) / (2 * TH) * N * cw, y = m.t + (TD - td) / (2 * TD) * N * ch; g.fillStyle = `rgba(255,255,255,${0.15 + 0.85 * k / trail.length})`; g.beginPath(); g.arc(x, y, 2 + 2 * k / trail.length, 0, 7); g.fill(); }
    g.textAlign = 'left'; g.fillStyle = '#d9825b'; g.fillText('■ push right', m.l + 6, m.t + 14); g.fillStyle = '#7aa2c8'; g.fillText('■ push left', m.l + 6, m.t + 28);
  }

  /* ============================================================ T2 ====== */
  Lab.register({
    id: 't2', unit: 1, num: 'T2', title: 'Deep Q-Network on CartPole',
    tag: 'Watch a neural net learn to balance a pole — and switch off its stabilisers to see why they exist.',
    star: 'Ablate replay buffer / target network · live policy map · race a human',
    idea: `<p>A DQN approximates <code>Q(s,a)</code> with a neural net trained on the TD target <code>r + γ·max<sub>a′</sub> Q<sub>target</sub>(s′,a′)</code>. Three ingredients tame the instability of bootstrapping from your own estimates: <b>experience replay</b>, a <b>target network</b> and an <b>ε-greedy</b> schedule. The environment is the standard CartPole-v1 physics (same equations as Gymnasium), running here in JavaScript.</p>
    <p>The notebook can only train. Here you can <b>turn replay and the target net off</b> and watch learning stall, see the net’s <b>decision map</b> over (θ, θ̇) sharpen as it learns, shove the pole with a kick, and <b>race the agent yourself</b> with the arrow keys.</p>`,
    tryIt: ['Press <b>Train</b>. The left cart shows the <i>current greedy policy</i> (not the training episodes). Expect a decent balance after ~100–150 episodes.',
      'Untick <b>replay</b> and <b>target net</b>, press <b>Apply &amp; reset</b> and train again — the same net usually fails to learn. That is the instability the tricks fix.',
      'The <b>policy map</b> should develop a diagonal boundary: push right when the pole leans/falls right (orange), left otherwise (blue). White dots are the live cart’s recent states.',
      'Press <b>Kick</b> to disturb the pole, or <b>wind</b> for random shoves — a robust policy recovers.',
      'Switch to <b>Play vs agent</b>, use ← → (or A/D) and try to beat the agent’s step count.'],
    hood: `<pre>loss = Huber( Q(s,a) − [ r + γ · max<sub>a′</sub> Q_target(s′,a′) · (1−done) ] )
buffer: 20 000 transitions, minibatch 32          target: hard copy every 200 updates
net: 4 → 32 → 32 → 2 (ReLU), Adam lr 1e-3          ε: 1.0 × 0.98 per episode (min 0.02)</pre>
    <p>The network, back-propagation and Adam are the from-scratch classes in <code>js/ml.js</code> (gradient-checked against finite differences). The physics reproduces Gymnasium’s Euler integrator: gravity 9.8, cart 1.0 kg, pole 0.1 kg, half-length 0.5 m, force ±10 N, τ = 0.02 s, failure at |θ| &gt; 12° or |x| &gt; 2.4, truncation at 500 steps.</p>`,
    quiz: [
      { q: 'What problem does the replay buffer mainly solve?', o: ['Too little memory', 'Strongly correlated consecutive samples', 'Exploding rewards', 'Choosing ε'], a: 1, why: 'Consecutive transitions are highly correlated; random minibatches from the buffer approximate i.i.d. data and re-use experience.' },
      { q: 'Why is a separate target network used?', o: ['It trains faster', 'It makes the TD target stop chasing the very weights being updated', 'It removes the need for ε', 'It stores the buffer'], a: 1, why: 'A slowly-changing target stabilises the regression; otherwise the target moves every update.' },
      { q: 'In the policy map, a vertical boundary near θ = 0 mainly means the agent…', o: ['ignores the pole', 'pushes toward the side the pole leans', 'always pushes left', 'has not trained'], a: 1, why: 'To catch a falling pole you accelerate the cart under it, i.e. push toward the lean direction.' }],
    mount(root, ctx) {
      const P = { replay: true, target: true, lr: 1e-3, gamma: 0.99, decay: 0.98, speed: 12, seed: 11 };
      let T = null, D = null, running = false, trail = [], mode = 'map', frame = 0, humanD = null;
      const create = () => {
        T = makeTrainer('dqn', P.seed, { useReplay: P.replay, useTarget: P.target, lr: P.lr, gamma: P.gamma, epsDecay: P.decay });
        D = makeDemo(T, 99); humanD = makeDemo(T, 99); humanD.humanMode = true; trail = [];
      };
      create();
      const cvA = Ui.canvas(520, 250), cvB = Ui.canvas(520, 250), cvM = Ui.canvas(520, 300), cvC = Ui.canvas(560, 250), chart = new LineChart(cvC, { xlabel: 'episode', yMin: 0, yMax: 520 });
      const stEp = Ui.stat('episode'), stAvg = Ui.stat('avg20 reward'), stEps = Ui.stat('ε'), stLoss = Ui.stat('loss'), stBuf = Ui.stat('buffer');
      const trainBtn = Ui.btn('Train', () => { running = !running; trainBtn.textContent = running ? 'Pause' : 'Train'; }, 'pri');
      const cReplay = Ui.chk('experience replay', true, v => P.replay = v), cTarget = Ui.chk('target network', true, v => P.target = v);
      const sLr = Ui.slider({ label: 'learning rate (needs reset)', min: 1, max: 30, step: 1, value: 10, fmt: v => (v * 1e-4).toExponential(1), on: v => { P.lr = v * 1e-4; } });
      const sG = Ui.slider({ label: 'discount γ', min: 0.8, max: 0.999, step: 0.001, value: 0.99, on: v => { P.gamma = v; T.agent.gamma = v; } });
      const sD = Ui.slider({ label: 'ε decay / episode', min: 0.9, max: 0.999, step: 0.001, value: 0.98, on: v => { P.decay = v; T.agent.epsDecay = v; } });
      const sS = Ui.slider({ label: 'train speed (ms/frame)', min: 2, max: 40, step: 1, value: 12, on: v => P.speed = v });
      const modeSeg = Ui.seg([{ v: 'map', label: 'Policy map' }, { v: 'play', label: 'Play vs agent' }], 'map', v => { mode = v; humanD.reset(); D.reset(); });
      const windChk = Ui.chk('random wind', false, v => { D.wind = v ? 1.0 : 0; humanD.wind = D.wind; });
      ctx.on(window, 'keydown', e => { const k = e.key; if (k === 'ArrowLeft' || k === 'a' || k === 'A') { humanD.humanAct = 0; e.preventDefault(); } if (k === 'ArrowRight' || k === 'd' || k === 'D') { humanD.humanAct = 1; e.preventDefault(); } });
      const left = h('div', null,
        Ui.card('Training controls', Ui.row(trainBtn, Ui.btn('Apply & reset', () => { create(); running = false; trainBtn.textContent = 'Train'; }), Ui.btn('Kick the pole', () => { D.kick(); humanD.kick(); }), windChk), Ui.row(cReplay, cTarget), Ui.row(sLr, sG), Ui.row(sD, sS), Ui.row(stEp, stAvg, stEps, stLoss, stBuf)),
        Ui.card('Reward per episode', cvC.c, Ui.legend([['#4ea8ff55', 'episode reward'], ['#d7b56d', 'moving avg (20)'], ['#86b894', 'solved ≥ 475']])));
      const right = h('div', null, Ui.card('Greedy policy (live)', cvA.c), Ui.card(null, h('div', { class: 'row', style: 'margin-bottom:8px' }, modeSeg, h('span', { class: 'sp' }), h('span', { class: 'muted', id: 'plhint', style: 'font-size:12.5px' }, '')), cvM.c, cvB.c));
      root.appendChild(Ui.grid('g2', left, right));
      const plHint = right.querySelector('#plhint');
      ctx.raf(dt => {
        if (running) T.run(P.speed);
        const obs = D.env.obs(); trail.push([obs[2], obs[3]]); if (trail.length > 40) trail.shift();
        D.advance(dt); const o = D.env.obs();
        drawCart(cvA, D.env.s, { label: 'agent (greedy)', score: D.score, pref: Math.tanh(T.pref(o) * 0.5), msg: D.lastMsg && D.score < 5 ? D.lastMsg : '' });
        if (mode === 'map') { cvB.c.style.display = 'none'; cvM.c.style.display = 'block'; if (frame % 6 === 0) policyMap(cvM, T, trail); plHint.textContent = 'colour = which push the net prefers at x=0, ẋ=0'; }
        else { cvM.c.style.display = 'none'; cvB.c.style.display = 'block'; humanD.advance(dt); drawCart(cvB, humanD.env.s, { label: 'YOU  (← → / A D)', score: humanD.score, msg: humanD.score < 5 ? 'best: ' + humanD.best + '  vs agent best: ' + D.best : '' }); plHint.textContent = 'you vs the agent — same start'; }
        if (frame++ % 5 === 0) {
          const c = curve(T); chart.draw([{ name: 'reward', color: 'rgba(122,162,200,.45)', pts: c.pts, width: 1.4 }, { name: 'avg20', color: '#d7b56d', pts: c.ma, width: 2.6 }], { hlines: [{ y: 475, color: '#86b894' }] });
          stEp.set(T.rewards.length); stAvg.set(T.avg().toFixed(0)); stEps.set(T.agent.eps.toFixed(2)); stLoss.set(T.loss.toFixed(3)); stBuf.set(T.agent.buf.length);
        }
      });
    },
  });

  /* ============================================================ T3 ====== */
  Lab.register({
    id: 't3', unit: 1, num: 'T3', title: 'Policy gradient — REINFORCE',
    tag: 'Optimise the policy directly. Then drop in a DQN rival and compare the two learning philosophies on the same pole.',
    star: 'REINFORCE vs DQN head-to-head · toggle return standardisation to see variance',
    idea: `<p><b>REINFORCE</b> skips value functions: it nudges the policy π(a|s) up in proportion to the <i>return</i> that followed each action, <code>∇J = E[ ∇log π(aₜ|sₜ) · Gₜ ]</code>. It is unbiased but high-variance, which is why the notebook standardises the returns.</p>
    <p>Here you can switch standardisation off to see the variance problem, change the learning rate, and add a <b>DQN rival</b> that learns in parallel from the same physics so you can compare sample-efficiency and stability. The left cart is REINFORCE’s greedy policy; the <b>policy map</b> shows π(right) − π(left).</p>`,
    tryIt: ['Train REINFORCE alone: it improves in noisy jumps because every episode is a single high-variance gradient sample.',
      'Untick <b>standardise returns</b> and reset: learning is typically slower or erratic.',
      'Tick <b>add DQN rival</b>: both agents get the same compute budget. DQN usually reaches decent scores earlier (off-policy replay = more reuse), policy gradient often ends smoother.',
      'Raise the learning rate to 0.05 and watch it become unstable — a policy-gradient step that is too big can collapse the policy.'],
    hood: `<pre>G_t = Σ_k γ^k r_{t+k}          Ĝ_t = (G_t − mean(G)) / (std(G) + ε)
loss = − mean_t [ log π(a_t | s_t) · Ĝ_t ]          net: 4 → 32 → 2 (tanh) → softmax, Adam</pre>
    <p>Standardising acts like a baseline: it subtracts the average return so that only <i>better-than-usual</i> actions are reinforced. The gradient of the softmax cross-entropy w.r.t. logits is <code>(p − onehot(a)) · Ĝ</code>, which is what the code back-propagates.</p>`,
    quiz: [
      { q: 'Why does REINFORCE standardise returns?', o: ['To make them positive', 'To reduce the variance of the gradient estimate', 'To make γ = 1', 'It is required for the softmax'], a: 1, why: 'Returns of all-positive reward tasks push every sampled action up; centring/scaling lowers variance (a baseline).' },
      { q: 'Which statement is true?', o: ['REINFORCE learns Q-values', 'REINFORCE updates only after an episode ends', 'REINFORCE needs a target network', 'REINFORCE is deterministic'], a: 1, why: 'It needs full-episode returns Gₜ (Monte-Carlo), so it updates at episode end.' },
      { q: 'The policy map is the probability of…', o: ['reward', 'pushing right minus pushing left', 'the next state', 'falling'], a: 1, why: 'We plot π(right|s) − π(left|s): positive (orange) means the policy prefers pushing right.' }],
    mount(root, ctx) {
      const P = { std: true, lr: 0.01, rival: false, speed: 12, hidden: 32 };
      let R, Dq, DR, DD, running = false, frame = 0, trail = [];
      const create = () => { R = makeTrainer('rf', 5, { standardize: P.std, lr: P.lr, hidden: P.hidden }); Dq = makeTrainer('dqn', 5); DR = makeDemo(R, 7); DD = makeDemo(Dq, 7); trail = []; };
      create();
      const cvR = Ui.canvas(520, 230), cvD = Ui.canvas(520, 230), cvM = Ui.canvas(520, 280), cvC = Ui.canvas(560, 260), chart = new LineChart(cvC, { xlabel: 'episode', yMin: 0, yMax: 520 });
      const stEp = Ui.stat('REINFORCE ep'), stAvg = Ui.stat('REINFORCE avg20'), stEp2 = Ui.stat('DQN ep'), stAvg2 = Ui.stat('DQN avg20');
      const trainBtn = Ui.btn('Train', () => { running = !running; trainBtn.textContent = running ? 'Pause' : 'Train'; }, 'pri');
      const cStd = Ui.chk('standardise returns', true, v => { P.std = v; R.agent.standardize = v; });
      const cRival = Ui.chk('add DQN rival', false, v => { P.rival = v; cvD.c.parentNode.style.display = v ? '' : 'none'; });
      const sLr = Ui.slider({ label: 'learning rate', min: 0.002, max: 0.06, step: 0.002, value: 0.01, on: v => { P.lr = v; R.agent.opt.lr = v; } });
      const sH = Ui.slider({ label: 'hidden units (needs reset)', min: 8, max: 64, step: 8, value: 32, on: v => P.hidden = v });
      const sG = Ui.slider({ label: 'discount γ', min: 0.8, max: 0.999, step: 0.001, value: 0.99, on: v => { R.agent.gamma = v; } });
      const sS = Ui.slider({ label: 'train speed (ms/frame)', min: 2, max: 40, step: 1, value: 12, on: v => P.speed = v });
      const left = h('div', null,
        Ui.card('Controls', Ui.row(trainBtn, Ui.btn('Apply & reset', () => { create(); running = false; trainBtn.textContent = 'Train'; }), cStd, cRival), Ui.row(sLr, sH), Ui.row(sG, sS), Ui.row(stEp, stAvg, stEp2, stAvg2)),
        Ui.card('Learning curves (moving avg 20)', cvC.c, Ui.legend([['#d9825b', 'REINFORCE'], ['#7aa2c8', 'DQN rival'], ['#86b894', 'solved ≥ 475']])));
      const dCard = Ui.card('DQN rival — greedy policy', cvD.c); dCard.style.display = 'none';
      const right = h('div', null, Ui.card('REINFORCE — greedy policy', cvR.c), dCard, Ui.card('REINFORCE policy map  π(right) − π(left)', cvM.c));
      cvD.c.parentNode.style.display = 'none';
      root.appendChild(Ui.grid('g2', left, right));
      ctx.raf(dt => {
        if (running) { R.run(P.speed * (P.rival ? 0.5 : 1)); if (P.rival) Dq.run(P.speed * 0.5); }
        const o = DR.env.obs(); trail.push([o[2], o[3]]); if (trail.length > 40) trail.shift();
        DR.advance(dt); drawCart(cvR, DR.env.s, { label: 'REINFORCE', score: DR.score, pref: R.pref(DR.env.obs()) });
        if (P.rival) { DD.advance(dt); drawCart(cvD, DD.env.s, { label: 'DQN', score: DD.score, pref: Math.tanh(Dq.pref(DD.env.obs()) * 0.5) }); }
        if (frame % 6 === 0) policyMap(cvM, R, trail);
        if (frame++ % 5 === 0) {
          const a = curve(R), b = curve(Dq); chart.draw([{ name: 'REINFORCE', color: '#d9825b', pts: a.ma, width: 2.6 }, ...(P.rival ? [{ name: 'DQN', color: '#7aa2c8', pts: b.ma, width: 2.6 }] : [])], { hlines: [{ y: 475, color: '#86b894' }] });
          stEp.set(R.rewards.length); stAvg.set(R.avg().toFixed(0)); stEp2.set(P.rival ? Dq.rewards.length : '–'); stAvg2.set(P.rival ? Dq.avg().toFixed(0) : '–');
        }
      });
    },
  });

  /* ============================================================ B1 ====== */
  Lab.register({
    id: 'b1', unit: 1, num: 'B1', bonus: true, title: 'GridWorld: DP, Q-learning vs SARSA',
    tag: 'Syllabus extras: value/policy iteration, and the classic cliff where Q-learning and SARSA disagree.',
    star: 'Paint your own maze · watch value iteration propagate · see on- vs off-policy behaviour',
    idea: `<p>Unit 1’s syllabus lists <b>value iteration</b>, <b>policy iteration</b> and model-free control with <b>SARSA</b> and <b>Q-learning</b>. This bonus lab makes them visible. In <b>Dynamic programming</b> the model is known: sweep the Bellman optimality backup and watch value spread outward from the goal. In <b>Cliff walking</b> two learners run side by side: Q-learning (off-policy) learns the shortest path along the cliff edge, SARSA (on-policy) accounts for its own exploration and takes the safe route.</p>`,
    tryIt: ['DP tab: click <b>Sweep</b> repeatedly and watch the value wave travel from the goal (green +10). Raise slip probability: the policy detours around pits.',
      'Paint walls/pits/goal with the tool selector and click cells. Re-run value iteration.',
      'Compare <b>policy iteration</b> (few big steps) with value iteration (many small sweeps).',
      'Cliff tab: train both for ~500 episodes. Q-learning’s online return is lower (it falls off the cliff while exploring) even though its greedy path is shorter. Lower ε to 0.01 and the two converge.'],
    hood: `<pre>Value iteration : V(s) ← max_a Σ_s' P(s'|s,a) [ r(s') + γ V(s') ]
Policy iteration: evaluate π until ΔV&lt;1e-4, then π(s) ← argmax_a Σ P [ r + γ V ]
SARSA           : Q(s,a) += α [ r + γ Q(s',a')  − Q(s,a) ]   (a' = the action actually taken)
Q-learning      : Q(s,a) += α [ r + γ max_a' Q(s',a') − Q(s,a) ]</pre>
    <p>Because SARSA’s target includes the exploratory next action, states next to the cliff look dangerous to it, so its greedy path keeps a safe distance.</p>`,
    quiz: [
      { q: 'Which pair is on-policy vs off-policy?', o: ['Q-learning on, SARSA off', 'SARSA on, Q-learning off', 'both on-policy', 'both off-policy'], a: 1, why: 'SARSA evaluates the policy it follows (uses the actual next action); Q-learning evaluates the greedy policy regardless of behaviour.' },
      { q: 'Value iteration needs…', o: ['no model of the environment', 'the transition model P(s′|s,a)', 'a neural network', 'a replay buffer'], a: 1, why: 'It is model-based dynamic programming: it sweeps the Bellman optimality equation using P and r.' },
      { q: 'Increasing the slip probability makes the optimal policy…', o: ['hug dangerous cells', 'keep away from pits', 'ignore the goal', 'random'], a: 1, why: 'With stochastic moves, cells next to pits risk accidental falls, so values (and the policy) steer around them.' }],
    mount(root, ctx) {
      const tabs = Ui.seg([{ v: 'dp', label: 'Dynamic programming' }, { v: 'cliff', label: 'Cliff: Q-learning vs SARSA' }], 'dp', v => { dpBox.style.display = v === 'dp' ? '' : 'none'; clBox.style.display = v === 'cliff' ? '' : 'none'; });
      const dpBox = h('div'), clBox = h('div', { style: 'display:none' });
      root.appendChild(h('div', { style: 'margin-bottom:14px' }, tabs)); root.appendChild(dpBox); root.appendChild(clBox);

      /* ----- DP ----- */
      const W = 8, H = 6, DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]], AR = ['↑', '→', '↓', '←'];
      const cell = Array.from({ length: H }, () => new Array(W).fill(0)); cell[0][W - 1] = 3; cell[1][W - 1] = 2; cell[1][3] = 2; cell[2][3] = 2; cell[3][1] = 1; cell[3][2] = 1; cell[4][5] = 1; cell[3][5] = 1; cell[2][5] = 1;
      const start = [0, H - 1]; let V, pol, sweeps = 0, delta = 0, agent = null, tool = 1, par = { gamma: 0.95, slip: 0.2, step: -0.04 };
      const initV = () => { V = Array.from({ length: H }, () => new Array(W).fill(0)); pol = Array.from({ length: H }, () => new Array(W).fill(1)); sweeps = 0; delta = 0; agent = null; };
      initV();
      const term = (x, y) => cell[y][x] >= 2, wall = (x, y) => cell[y][x] === 1, rewardOf = (x, y) => (cell[y][x] === 3 ? 10 : cell[y][x] === 2 ? -10 : 0);
      function outcomes(x, y, a) {
        const res = []; const probs = [[a, 1 - par.slip], [(a + 1) % 4, par.slip / 2], [(a + 3) % 4, par.slip / 2]];
        for (const [d, p] of probs) { let nx = x + DIRS[d][0], ny = y + DIRS[d][1]; if (nx < 0 || ny < 0 || nx >= W || ny >= H || wall(nx, ny)) { nx = x; ny = y; } res.push([p, nx, ny]); }
        return res;
      }
      const qOf = (x, y, a, VV) => outcomes(x, y, a).reduce((s, [p, nx, ny]) => s + p * (term(nx, ny) ? rewardOf(nx, ny) : par.step + par.gamma * VV[ny][nx]), 0);
      function sweep() { const nv = V.map(r => r.slice()); delta = 0; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { if (wall(x, y) || term(x, y)) continue; let b = -Infinity, ba = 0; for (let a = 0; a < 4; a++) { const q = qOf(x, y, a, V); if (q > b) { b = q; ba = a; } } delta = Math.max(delta, Math.abs(b - V[y][x])); nv[y][x] = b; pol[y][x] = ba; } V = nv; sweeps++; }
      function policyIter() {
        let it = 0; do { delta = 0; const nv = V.map(r => r.slice()); for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { if (wall(x, y) || term(x, y)) continue; const v = qOf(x, y, pol[y][x], V); delta = Math.max(delta, Math.abs(v - V[y][x])); nv[y][x] = v; } V = nv; it++; } while (delta > 1e-4 && it < 300);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { if (wall(x, y) || term(x, y)) continue; let b = -Infinity, ba = 0; for (let a = 0; a < 4; a++) { const q = qOf(x, y, a, V); if (q > b + 1e-9) { b = q; ba = a; } } pol[y][x] = ba; } sweeps++; return it;
      }
      const cvG = Ui.canvas(640, 480); const cs = 80;
      function drawGrid() {
        const { g } = cvG; g.clearRect(0, 0, 640, 480); g.font = '12px "JetBrains Mono",monospace';
        let mx = 1; for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) mx = Math.max(mx, Math.abs(V[y][x]));
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const X = x * cs, Y = y * cs; const c = cell[y][x];
          g.fillStyle = c === 1 ? '#332e27' : c === 3 ? '#2a4a35' : c === 2 ? '#4a2427' : Colors.diverge(V[y][x] / mx); g.fillRect(X + 1, Y + 1, cs - 2, cs - 2);
          g.fillStyle = '#ece6d8'; g.textAlign = 'center';
          if (c === 1) { g.fillStyle = '#6f6859'; g.fillText('wall', X + cs / 2, Y + cs / 2 + 4); }
          else if (c >= 2) { g.font = '600 15px "JetBrains Mono"'; g.fillText(c === 3 ? 'GOAL +10' : 'PIT −10', X + cs / 2, Y + cs / 2 + 5); g.font = '12px "JetBrains Mono",monospace'; }
          else { g.fillText(V[y][x].toFixed(2), X + cs / 2, Y + cs - 10); g.font = '24px "Instrument Sans"'; g.fillStyle = '#ffffffcc'; g.fillText(sweeps ? AR[pol[y][x]] : '·', X + cs / 2, Y + cs / 2 + 4); g.font = '12px "JetBrains Mono",monospace'; }
          if (x === start[0] && y === start[1]) { g.strokeStyle = '#d7b56d'; g.lineWidth = 2; g.strokeRect(X + 3, Y + 3, cs - 6, cs - 6); g.fillStyle = '#d7b56d'; g.fillText('S', X + 10, Y + 14); }
        }
        if (agent) { g.fillStyle = '#fff'; g.beginPath(); g.arc(agent.x * cs + cs / 2, agent.y * cs + cs / 2 + 18, 9, 0, 7); g.fill(); g.strokeStyle = '#d7b56d'; g.lineWidth = 2; g.stroke(); }
      }
      cvG.c.addEventListener('click', e => { const [px, py] = cvG.xy(e); const x = Math.floor(px / cs), y = Math.floor(py / cs); if (x < 0 || y < 0 || x >= W || y >= H || (x === start[0] && y === start[1])) return; cell[y][x] = tool === 0 ? 0 : (cell[y][x] === tool ? 0 : tool); initV(); stat(); });
      const stS = Ui.stat('sweeps / iterations'), stD = Ui.stat('max |ΔV|');
      function stat() { stS.set(sweeps); stD.set(delta.toFixed(4)); }
      const toolSeg = Ui.seg([{ v: 1, label: 'wall' }, { v: 2, label: 'pit' }, { v: 3, label: 'goal' }, { v: 0, label: 'erase' }], 1, v => tool = v);
      let autoVI = false;
      dpBox.appendChild(Ui.grid('g21', Ui.card('Value function & greedy policy (click cells to paint)', cvG.c),
        h('div', null, Ui.card('Dynamic programming', Ui.row(Ui.btn('Sweep (value iter.)', () => { sweep(); stat(); }, 'pri'), Ui.btn('Policy iteration step', () => { policyIter(); stat(); }), Ui.btn('Auto-run VI', () => { autoVI = !autoVI; })), Ui.row(Ui.btn('Reset values', () => { initV(); stat(); }), Ui.btn('Send agent', () => { agent = { x: start[0], y: start[1], n: 0 }; }))),
          Ui.card('Environment', Ui.slider({ label: 'discount γ', min: 0.5, max: 0.999, step: 0.001, value: par.gamma, on: v => { par.gamma = v; initV(); } }), Ui.slider({ label: 'slip probability', min: 0, max: 0.6, step: 0.05, value: par.slip, on: v => { par.slip = v; initV(); } }), Ui.slider({ label: 'step cost', min: -0.5, max: 0, step: 0.01, value: par.step, on: v => { par.step = v; initV(); } }), h('div', { style: 'margin-top:8px' }, 'Paint tool'), toolSeg, h('div', { style: 'margin-top:8px' }, Ui.row(stS, stD))))));
      let tAgent = 0;
      ctx.raf(dt => {
        if (autoVI && delta >= 1e-5 || (autoVI && sweeps === 0)) { sweep(); stat(); } else if (autoVI) autoVI = false;
        if (agent) { tAgent += dt; if (tAgent > 0.22) { tAgent = 0; const a = pol[agent.y][agent.x], r = Math.random(); let d = a; if (r < par.slip / 2) d = (a + 1) % 4; else if (r < par.slip) d = (a + 3) % 4; const nx = agent.x + DIRS[d][0], ny = agent.y + DIRS[d][1]; if (!(nx < 0 || ny < 0 || nx >= W || ny >= H || wall(nx, ny))) { agent.x = nx; agent.y = ny; } agent.n++; if (term(agent.x, agent.y) || agent.n > 80) { setTimeout(() => { agent = null; }, 500); agent.n = 999; } } }
        drawGrid();
      });

      /* ----- Cliff ----- */
      const CW = 12, CH = 4, cfg = { eps: 0.1, alpha: 0.5, speed: 5 }, crng = new RNG(3);
      let QL, SA, retQ, retS, crun = false;
      const idx = (x, y) => y * CW + x;
      function newCliff() { QL = { Q: new Float64Array(CW * CH * 4), name: 'Q-learning' }; SA = { Q: new Float64Array(CW * CH * 4), name: 'SARSA' }; retQ = []; retS = []; }
      newCliff();
      const cstep = (x, y, a) => { let nx = x + DIRS[a][0], ny = y + DIRS[a][1]; nx = Math.max(0, Math.min(CW - 1, nx)); ny = Math.max(0, Math.min(CH - 1, ny)); if (ny === CH - 1 && nx > 0 && nx < CW - 1) return { x: 0, y: CH - 1, r: -100, done: false }; return { x: nx, y: ny, r: -1, done: nx === CW - 1 && ny === CH - 1 }; };
      const egreedy = (Q, x, y) => { if (crng.rand() < cfg.eps) return crng.int(4); const q = Q.subarray(idx(x, y) * 4, idx(x, y) * 4 + 4); let b = 0; for (let a = 1; a < 4; a++) if (q[a] > q[b] + 1e-12) b = a; return b; };
      const maxQ = (Q, x, y) => Math.max(...Q.subarray(idx(x, y) * 4, idx(x, y) * 4 + 4));
      function episode(agent, sarsa, rets) {
        let x = 0, y = CH - 1, a = egreedy(agent.Q, x, y), G = 0, steps = 0;
        while (steps++ < 300) {
          const t = cstep(x, y, a), a2 = egreedy(agent.Q, t.x, t.y); G += t.r; const i = idx(x, y) * 4 + a;
          const target = t.done ? t.r : t.r + (sarsa ? agent.Q[idx(t.x, t.y) * 4 + a2] : maxQ(agent.Q, t.x, t.y));
          agent.Q[i] += cfg.alpha * (target - agent.Q[i]); x = t.x; y = t.y; a = a2; if (t.done) break;
        }
        rets.push(G);
      }
      const greedyPath = Q => { let x = 0, y = CH - 1; const path = [[x, y]], seen = new Set(); for (let k = 0; k < 40; k++) { const q = Q.subarray(idx(x, y) * 4, idx(x, y) * 4 + 4); let b = 0; for (let a = 1; a < 4; a++) if (q[a] > q[b] + 1e-12) b = a; const t = cstep(x, y, b); if (t.r === -100) { path.push([t.x, t.y, 'fall']); break; } x = t.x; y = t.y; path.push([x, y]); if (t.done || seen.has(idx(x, y))) break; seen.add(idx(x, y)); } return path; };
      const cvQ = Ui.canvas(560, 220), cvS = Ui.canvas(560, 220), cvR = Ui.canvas(560, 250), cchart = new LineChart(cvR, { xlabel: 'episode', title: 'online return per episode (higher is better, smoothed over 25)' });
      function drawCliff(cv, ag, col) {
        const { g, w } = cv, cs = w / CW, hh = 220 / CH; g.clearRect(0, 0, w, 220);
        for (let y = 0; y < CH; y++) for (let x = 0; x < CW; x++) { const isCliff = y === CH - 1 && x > 0 && x < CW - 1; g.fillStyle = isCliff ? '#4a2427' : '#1a1713'; g.fillRect(x * cs + 1, y * hh + 1, cs - 2, hh - 2); if (isCliff) { g.fillStyle = '#cf6a76'; g.font = '10px "Instrument Sans"'; g.textAlign = 'center'; g.fillText('cliff', x * cs + cs / 2, y * hh + hh / 2 + 3); } }
        g.fillStyle = '#d7b56d'; g.font = '700 13px "Instrument Sans"'; g.textAlign = 'center'; g.fillText('S', cs / 2, (CH - 1) * hh + hh / 2 + 4); g.fillStyle = '#86b894'; g.fillText('G', (CW - 0.5) * cs, (CH - 1) * hh + hh / 2 + 4);
        const p = greedyPath(ag.Q); g.strokeStyle = col; g.lineWidth = 4; g.lineJoin = 'round'; g.beginPath(); p.forEach(([x, y], i) => { const X = x * cs + cs / 2, Y = y * hh + hh / 2; i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke();
        g.fillStyle = '#ece6d8'; g.textAlign = 'left'; g.font = '600 12px "Instrument Sans"'; g.fillText(ag.name + ' — greedy path' + (p[p.length - 1][2] ? ' (falls!)' : ' (' + (p.length - 1) + ' steps)'), 8, 14);
      }
      const cbtn = Ui.btn('Train both', () => { crun = !crun; cbtn.textContent = crun ? 'Pause' : 'Train both'; }, 'pri');
      const cEp = Ui.stat('episodes'), cQ = Ui.stat('Q-learning avg'), cS = Ui.stat('SARSA avg');
      clBox.appendChild(Ui.grid('g2', h('div', null, Ui.card('Learned greedy paths', cvQ.c, h('div', { style: 'height:8px' }), cvS.c)),
        h('div', null, Ui.card('Controls', Ui.row(cbtn, Ui.btn('Reset', newCliff)), Ui.row(Ui.slider({ label: 'exploration ε', min: 0, max: 0.4, step: 0.01, value: cfg.eps, on: v => cfg.eps = v }), Ui.slider({ label: 'step size α', min: 0.05, max: 1, step: 0.05, value: cfg.alpha, on: v => cfg.alpha = v })), Ui.row(Ui.slider({ label: 'speed (episodes/frame)', min: 1, max: 30, step: 1, value: cfg.speed, on: v => cfg.speed = v })), Ui.row(cEp, cQ, cS)), Ui.card(null, cvR.c, Ui.legend([['#7aa2c8', 'Q-learning (off-policy)'], ['#d9825b', 'SARSA (on-policy)']])))));
      const smooth = (r, k = 25) => { const o = []; for (let i = k - 1; i < r.length; i++) { let s = 0; for (let j = i - k + 1; j <= i; j++) s += r[j]; o.push([i + 1, s / k]); } return o; };
      let cf = 0;
      ctx.raf(() => {
        if (clBox.style.display === 'none') return;
        if (crun) for (let i = 0; i < cfg.speed; i++) { episode(QL, false, retQ); episode(SA, true, retS); }
        drawCliff(cvQ, QL, '#7aa2c8'); drawCliff(cvS, SA, '#d9825b');
        if (cf++ % 6 === 0) { cchart.draw([{ name: 'Q-learning', color: '#7aa2c8', pts: smooth(retQ) }, { name: 'SARSA', color: '#d9825b', pts: smooth(retS) }], { yMin: -150, yMax: 0 }); cEp.set(retQ.length); const av = r => r.length ? (r.slice(-25).reduce((a, b) => a + b, 0) / Math.min(25, r.length)).toFixed(0) : '–'; cQ.set(av(retQ)); cS.set(av(retS)); }
      });
      stat();
    },
  });
})();
