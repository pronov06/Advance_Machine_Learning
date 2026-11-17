<p align="center">
  <img src="https://img.shields.io/badge/Course-21CSE425T-0D1117?style=for-the-badge&labelColor=0D1117&color=B08D57" alt="course" />
  <img src="https://img.shields.io/badge/Tutorials-T1–T15-0D1117?style=for-the-badge&labelColor=0D1117&color=B08D57" alt="tutorials" />
  <img src="https://img.shields.io/badge/Python-3.10+-0D1117?style=for-the-badge&logo=python&logoColor=B08D57&labelColor=0D1117&color=B08D57" alt="python" />
  <img src="https://img.shields.io/badge/Colab-Ready-0D1117?style=for-the-badge&logo=googlecolab&logoColor=B08D57&labelColor=0D1117&color=B08D57" alt="colab" />
  <img src="https://img.shields.io/badge/License-Academic-0D1117?style=for-the-badge&labelColor=0D1117&color=B08D57" alt="license" />
</p>

<br/>

<h1 align="center">
  <samp>ADVANCE MACHINE LEARNING</samp>
</h1>

<p align="center">
  <samp>A curated collection of 15 hands-on tutorials spanning reinforcement learning, graph neural networks,<br/>ensemble methods, association rule mining, and deep neural networks — accompanied by a fully<br/>interactive browser-based simulation lab built from scratch in vanilla JavaScript.</samp>
</p>

<br/>

<p align="center">
  <a href="#-curriculum"><samp>Curriculum</samp></a>&ensp;·&ensp;
  <a href="#-interactive-simulation-lab"><samp>Simulation Lab</samp></a>&ensp;·&ensp;
  <a href="#-quick-start"><samp>Quick Start</samp></a>&ensp;·&ensp;
  <a href="#-tech-stack"><samp>Tech Stack</samp></a>&ensp;·&ensp;
  <a href="#-repository-structure"><samp>Structure</samp></a>
</p>

<br/>

---

<br/>

## ◈ Curriculum

> Five units · Fifteen tutorials · One complete journey through advanced ML.

<br/>

<details>
<summary><b>&ensp;Unit 1 — Reinforcement Learning&emsp;</b><code>T1 · T2 · T3</code></summary>
<br/>

| Tutorial | Topic | Method | Dataset / Environment |
|:--------:|-------|--------|-----------------------|
| **T1** | RL Chatbot | Tabular Q-Learning | Custom intent corpus |
| **T2** | Deep Q-Learning | DQN with experience replay & target networks | CartPole-v1 (Gymnasium) |
| **T3** | Policy Gradient | REINFORCE with return standardisation | CartPole-v1 (Gymnasium) |

**Frameworks** &ensp; `PyTorch` · `Gymnasium` · `Stable-Baselines3`

<br/>
</details>

<details>
<summary><b>&ensp;Unit 2 — Machine Learning with Graphs&emsp;</b><code>T4 · T5 · T6</code></summary>
<br/>

| Tutorial | Topic | Method | Dataset |
|:--------:|-------|--------|-----------------------|
| **T4** | Graph Neural Networks | 2-layer GCN (Kipf & Welling) | Zachary's Karate Club |
| **T5** | Knowledge Graphs | RDF triples + SPARQL querying | Custom ontology |
| **T6** | Frequent Subgraph Mining | WL hashing & subgraph enumeration | Random graph ensembles |

**Frameworks** &ensp; `PyTorch Geometric` · `NetworkX` · `rdflib`

<br/>
</details>

<details>
<summary><b>&ensp;Unit 3 — Ensemble Learning&emsp;</b><code>T7 · T8 · T9</code></summary>
<br/>

| Tutorial | Topic | Method | Dataset |
|:--------:|-------|--------|-----------------------|
| **T7** | Boosting | AdaBoost & Gradient Boosting | Breast Cancer Wisconsin |
| **T8** | Bagging & Bias–Variance | Bootstrap aggregating with decomposition | Synthetic regression |
| **T9** | Ensemble Face Recognition | PCA + Random Forest / Bagging / AdaBoost | Olivetti Faces |

**Frameworks** &ensp; `scikit-learn` · `OpenCV` · `mlxtend`

<br/>
</details>

<details>
<summary><b>&ensp;Unit 4 — Association Rule Mining&emsp;</b><code>T10 · T11 · T12</code></summary>
<br/>

| Tutorial | Topic | Method | Dataset |
|:--------:|-------|--------|-----------------------|
| **T10** | Market Basket Analysis | Support, confidence & lift metrics | Retail transactions |
| **T11** | Apriori Algorithm | From-scratch + mlxtend implementation | Transaction database |
| **T12** | FP-Growth Classification | FP-tree construction & pattern-as-features | Iris-like synthetic data |

**Frameworks** &ensp; `mlxtend` · `pandas`

<br/>
</details>

<details>
<summary><b>&ensp;Unit 5 — Neural Networks&emsp;</b><code>T13 · T14 · T15</code></summary>
<br/>

| Tutorial | Topic | Method | Dataset |
|:--------:|-------|--------|-----------------------|
| **T13** | Optimizer Comparison | SGD vs Adam vs RMSProp vs Adagrad + Dropout & BatchNorm | sklearn Digits (8×8) |
| **T14** | Back-propagation | Chain rule walkthrough from scratch + numerical gradient check | Synthetic |
| **T15** | Deep NN Face Recognition | CNN with conv/pool/backprop, occlusion sensitivity | Faces dataset |

**Frameworks** &ensp; `PyTorch` · `torchvision` · `matplotlib` · `scikit-learn`

<br/>
</details>

<br/>

---

<br/>

## ◈ Interactive Simulation Lab

> Every tutorial rebuilt as a **live, in-browser simulation** — no installs, no Colab, no training wait.  
> All algorithms implemented **from scratch in vanilla JavaScript** and running directly in the page.

<br/>

<table>
<tr><td align="center" width="110"><b>Unit</b></td><td><b>Simulations</b></td></tr>
<tr>
  <td align="center"><samp>RL</samp></td>
  <td>Q-table heat-map with live RLHF-style 👍/👎 training · DQN with ablatable replay buffer & target net · policy map over (θ, θ̇) · race the agent with arrow keys · GridWorld with Q-learning vs SARSA</td>
</tr>
<tr>
  <td align="center"><samp>Graphs</samp></td>
  <td>Interactive GCN message passing with click-to-label & edge rewiring · SPARQL engine with matched-triple highlighting & TransE link prediction · WL colour-refinement animator · PageRank with teleport slider</td>
</tr>
<tr>
  <td align="center"><samp>Ensembles</samp></td>
  <td>Boosting with live sample-weight dots & residual view · bias²/variance measured over 40 fresh datasets · eigenface reconstruction slider with occlusion stress test</td>
</tr>
<tr>
  <td align="center"><samp>ARM</samp></td>
  <td>Click-to-build baskets with rule scatter & live recommender · animated Apriori pruning lattice · animated FP-tree construction with conditional pattern bases</td>
</tr>
<tr>
  <td align="center"><samp>Neural</samp></td>
  <td>Six-optimizer arena racing on loss landscapes · stage-by-stage chain-rule walkthrough with vanishing-gradient meter · live CNN filters & occlusion-sensitivity maps</td>
</tr>
</table>

<br/>

**Launch it:**

```bash
# Option A — just open the file (works offline)
open "Tutorial Simulation/index.html"

# Option B — with Node.js (hot-reload on localhost:5173)
cd "Tutorial Simulation"
node serve.js
```

> [!TIP]
> Press `C` for the full contents index · `[` and `]` to navigate chapters · progress is saved in your browser.

<br/>

---

<br/>

## ◈ Quick Start

### Notebooks (Google Colab)

Each notebook is **fully self-contained** — dependencies are installed in the first cell.

| Notebook | Open in Colab |
|----------|:-------------:|
| `Unit1_Reinforcement_Learning_T1-T3.ipynb` | [![Colab](https://img.shields.io/badge/Open-Colab-B08D57?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/pronov06/Advance_Machine_Learning/blob/main/Unit1_Reinforcement_Learning_T1-T3.ipynb) |
| `Unit2_Machine_Learning_with_Graphs_T4-T6.ipynb` | [![Colab](https://img.shields.io/badge/Open-Colab-B08D57?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/pronov06/Advance_Machine_Learning/blob/main/Unit2_Machine_Learning_with_Graphs_T4-T6.ipynb) |
| `Unit3_Ensemble_Learning_T7-T9.ipynb` | [![Colab](https://img.shields.io/badge/Open-Colab-B08D57?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/pronov06/Advance_Machine_Learning/blob/main/Unit3_Ensemble_Learning_T7-T9.ipynb) |
| `Unit4_Association_Rule_Mining_T10-T12.ipynb` | [![Colab](https://img.shields.io/badge/Open-Colab-B08D57?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/pronov06/Advance_Machine_Learning/blob/main/Unit4_Association_Rule_Mining_T10-T12.ipynb) |
| `Unit5_Neural_Networks_T13-T15.ipynb` | [![Colab](https://img.shields.io/badge/Open-Colab-B08D57?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/pronov06/Advance_Machine_Learning/blob/main/Unit5_Neural_Networks_T13-T15.ipynb) |

### Local Setup

```bash
git clone https://github.com/pronov06/Advance_Machine_Learning.git
cd Advance_Machine_Learning
pip install torch gymnasium stable-baselines3 torch_geometric networkx rdflib \
            scikit-learn pandas opencv-python mlxtend torchvision matplotlib
```

<br/>

---

<br/>

## ◈ Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-0D1117?style=for-the-badge&logo=pytorch&logoColor=EE4C2C" />
  <img src="https://img.shields.io/badge/PyG-0D1117?style=for-the-badge&logo=pyg&logoColor=3C2179" />
  <img src="https://img.shields.io/badge/scikit--learn-0D1117?style=for-the-badge&logo=scikitlearn&logoColor=F7931E" />
  <img src="https://img.shields.io/badge/Gymnasium-0D1117?style=for-the-badge&logo=openaigym&logoColor=0081A5" />
  <img src="https://img.shields.io/badge/NetworkX-0D1117?style=for-the-badge&logo=python&logoColor=4B8BBE" />
  <img src="https://img.shields.io/badge/OpenCV-0D1117?style=for-the-badge&logo=opencv&logoColor=5C3EE8" />
  <img src="https://img.shields.io/badge/JavaScript-0D1117?style=for-the-badge&logo=javascript&logoColor=F7DF1E" />
  <img src="https://img.shields.io/badge/Jupyter-0D1117?style=for-the-badge&logo=jupyter&logoColor=F37626" />
</p>

<br/>

---

<br/>

## ◈ Repository Structure

```
Advance_Machine_Learning/
│
├── Unit1_Reinforcement_Learning_T1-T3.ipynb      ◇ Q-Learning · DQN · REINFORCE
├── Unit2_Machine_Learning_with_Graphs_T4-T6.ipynb ◇ GCN · Knowledge Graphs · Subgraph Mining
├── Unit3_Ensemble_Learning_T7-T9.ipynb            ◇ Boosting · Bagging · Face Recognition
├── Unit4_Association_Rule_Mining_T10-T12.ipynb     ◇ Market Basket · Apriori · FP-Growth
├── Unit5_Neural_Networks_T13-T15.ipynb             ◇ Optimizers · Backprop · CNN Face Recognition
│
├── Tutorial Simulation/
│   ├── index.html              ▸ app shell
│   ├── css/style.css           ▸ design system & typography
│   ├── fonts/                  ▸ Instrument Serif · Instrument Sans · JetBrains Mono
│   ├── js/
│   │   ├── ml.js               ▸ pure ML engine (nets, optimizers, envs, trees, PCA)
│   │   ├── ui.js               ▸ DOM helpers, charts, force layout
│   │   ├── app.js              ▸ router, masthead, chapter pages
│   │   └── u1.js … u5.js       ▸ unit simulations
│   ├── serve.js                ▸ optional static server
│   ├── Start AML Lab.bat       ▸ one-click launcher
│   ├── Presenter Sheet.html    ▸ timed presentation guide
│   └── README.md               ▸ detailed simulation docs
│
├── .gitignore
└── README.md                   ◁ you are here
```

<br/>

---

<br/>

<p align="center">
  <samp><b>Pronov Mazumdar</b></samp><br/><br/>
  <a href="https://github.com/pronov06">
    <img src="https://img.shields.io/badge/GitHub-pronov06-0D1117?style=for-the-badge&logo=github&logoColor=white" />
  </a>
</p>

<p align="center">
  <samp>Built with precision · Designed for understanding</samp>
</p>

<br/>
