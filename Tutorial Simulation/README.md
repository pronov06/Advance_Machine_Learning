# AML Lab — interactive simulations for 21CSE425T (Advanced Machine Learning)

Every tutorial in the five unit notebooks (T1–T15) rebuilt as a **live, in-browser simulation**, plus three
bonus labs for syllabus topics the notebooks don't cover. Nothing to install, nothing to train in Colab:
all algorithms are implemented **from scratch in JavaScript** and run in the page.

## Run it
* **Double-click `index.html`** (works offline, no server), or
* double-click **`Start AML Lab.bat`** (starts `serve.js` on http://localhost:5173 if Node.js is installed), or
* `node serve.js` from this folder.

## Presenting it
Open **`Presenter Sheet.html`** and print it (Ctrl+P, A4, background graphics off): a timed 7-minute route, likely questions, and a rehearsal log.

## Design
* **Theme** — "Atelier": warm near-black, ivory type, one restrained brass accent; each unit has its own muted hue (terracotta, teal, brass, heather, steel).
* **Type** — Instrument Serif (headlines, numerals), Instrument Sans (interface and prose), JetBrains Mono (mathematics). All bundled in `fonts/` (SIL Open Font License) so it still works offline.
* **Structure** — masthead with a chapter ribbon (one tick per chapter, filled when you pass its check), a full-screen **Contents** index (button, or press `c`), editorial chapter pages with a tabbed **Briefing** (The idea · Try this · Mathematics · Check yourself) above the **Workbench**, and `[` / `]` to move between chapters.

## What's inside (and what's new compared with the notebooks)

| Task | Page | Beyond the notebook |
|---|---|---|
| **T1** Q-learning chatbot | Q-table heat-map, chat | train with **your 👍/👎** (RLHF-style), reward noise, swap answers mid-training, **teach new intents** live |
| **T2** DQN (CartPole) | physics + DQN | **ablate replay buffer / target net**, live policy map over (θ, θ̇), kick the pole, **race the agent** with arrow keys |
| **T3** REINFORCE | policy gradient | toggle return standardisation, **DQN-vs-REINFORCE head-to-head** |
| B1 GridWorld | value / policy iteration, cliff | paint mazes; **Q-learning vs SARSA** on the cliff |
| **T4** GCN (Karate Club) | message passing | **click to label nodes, rewire edges**, receptive-field inspector, how the receptive field grows with depth |
| **T5** Knowledge graph + SPARQL | own SPARQL engine | matched triples **light up**; ask in English → SPARQL; **TransE** link prediction with held-out facts |
| **T6** Frequent subgraph mining | WL hashing | click a pattern to highlight occurrences; **WL colour-refinement** animator; isomorphism-invariance test |
| B2 PageRank / label propagation | link analysis | teleport slider, seed labels by clicking |
| **T7** AdaBoost & Gradient Boosting | boosting | **dot size = live sample weight**, residual view, L1 vs L2 with outliers, why stumps can't do XOR |
| **T8** Bagging & bias–variance | decomposition | bias²/variance **measured over 40 fresh datasets**, bootstrap lab, depth sweep |
| **T9** Face recognition (ensembles) | PCA + RF/Bagging/AdaBoost | eigenfaces, reconstruction slider, probe with votes, **occlusion stress test** |
| **T10** Market-basket analysis | support/confidence/lift | build baskets by clicking, rule scatter, **live recommender** |
| **T11** Apriori | itemset lattice | **pruned region visualised**, counted-vs-brute-force statistic, correctness check |
| **T12** FP-tree for classification | FP-Growth | **animated tree construction**, conditional pattern bases, patterns-as-features classifier + flower probe |
| **T13** Optimizers (+Dropout/BN) | arena | **six optimizers racing on loss landscapes**; same-init MLP arena with live decision boundaries |
| **T14** Back-propagation | walkthrough | stage-by-stage chain rule with real numbers, **numerical gradient check**, vanishing-gradient meter |
| **T15** CNN face recognition | hand-written conv/pool/back-prop | live filters & feature maps, **occlusion-sensitivity map**, augmentation, vs T9 |
| B3 Perceptron / McCulloch–Pitts | XOR | the perceptron fails on XOR; a hidden layer fixes it |

Each page has *The idea*, *Try this*, *Under the hood* (equations/pseudocode) and a short **concept check**
(progress is stored in your browser). The T10 baskets are shared with T11 and T12.

## Honest notes
* **Datasets.** Olivetti faces, Iris, digits and the breast-cancer table need downloads, so they are replaced by
  procedural stand-ins generated in the page: a 12-person synthetic face gallery (T9, T15), Iris-like data sampled
  from the real Iris class means/standard deviations (T12), 2-D toy sets (T7, T13, T14). Karate Club (T4) and the
  textbook FP-growth example (T12) are the real data.
* **T4** uses the 2-faction ground truth by default; the "4 communities" option mirrors the PyG label set used in the notebook.
* **T8** shows the *regression* (squared-loss) decomposition measured by actually refitting on 40 datasets; the notebook uses mlxtend's 0-1-loss classification version.
* **T13** the MLP arena uses 2-D toy data (not digits); BatchNorm and Dropout are real implementations.

## Layout
```
index.html          app shell
css/style.css       design system (tokens, typography, every component)
fonts/              Instrument Serif, Instrument Sans, JetBrains Mono (woff2)
js/ml.js            pure ML engine: Net (+BatchNorm/Dropout), optimizers, CartPole, DQN, REINFORCE, GCN, CART trees, PCA, logistic
js/ui.js            DOM helpers, controls, charts, force layout
js/app.js           router, masthead/ribbon/contents, chapter pages, concept checks
js/u1.js … u5.js    the simulations (one file per unit)
serve.js            optional static server
```
The engine was checked headlessly: gradient checks of back-prop (with BatchNorm/Dropout) against finite differences,
Apriori vs FP-Growth vs brute force on 900 random databases, ESU subgraph enumeration vs brute force on 1,200 random graphs.
