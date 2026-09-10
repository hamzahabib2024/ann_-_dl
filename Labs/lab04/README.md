# Lab 04 — Self-Organising Maps

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Implementing and training a Self-Organising Map (SOM), an
unsupervised competitive network.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 04.pdf` | The lab manual: concepts, solved activities, and the graded task. |
| `activities_ANN_lab_04.ipynb` | The 5 solved activities from the manual (done in class), on the Iris dataset. |
| `tasks_ANN_lab_04.ipynb` | **The graded work** — Lab Task 1 and the written report, all cells executed with outputs. |
| `banknote_authentication.csv` | Banknote Authentication (UCI) — 1372 rows, 4 features, binary label. Used by the task. |

## How to run

Open `tasks_ANN_lab_04.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). The CSV must stay in the same folder as
the notebook.

```
pip install numpy pandas matplotlib scikit-learn
```

`scikit-learn` is needed only by the activities notebook, which uses it to load
Iris and for `MinMaxScaler`. The task notebook implements the SOM and the
scaling itself and needs only NumPy, pandas and matplotlib.

Run the cells **in order** — the `SOM` class is defined once and every section
after it depends on the trained map.

The activities notebook takes about 10 seconds to train, because the manual's
update loops over all 100 neurons in Python for every sample. The task notebook
trains in a few seconds despite using a larger grid and 7x more data, because
that loop is vectorised.

## What a Self-Organising Map is

A SOM is an **unsupervised** network introduced by Teuvo Kohonen. It maps
high-dimensional data onto a low-dimensional grid — normally a 2-D lattice of
`m x n` neurons — while preserving the topology of the input space: inputs that
are close together in the data end up close together on the map.

It has one input layer and one output layer. The output layer is the grid, and
**every neuron is connected to every input**, holding its own weight vector with
one value per feature. So the weights of an `m x n` map over `dim` features form
an array of shape `(m, n, dim)`.

Unlike the perceptron of Lab 03, there are no labels and no target output. The
map organises itself.

### The learning algorithm

Each input is processed in two phases.

**1. Competition.** Every neuron measures how close its weights are to the input,
normally by Euclidean distance. The closest one wins and is called the **Best
Matching Unit (BMU)**. Only the winner fires, which is why this is called
*winner-takes-all*.

```python
distances = np.linalg.norm(self.weights - x, axis=-1)
bmu = np.unravel_index(np.argmin(distances), (self.m, self.n))
```

**2. Adaptation.** The BMU moves towards the input — and so do its **neighbours
on the grid**. This is what separates a SOM from plain clustering: neighbouring
neurons learn together, which is what makes the finished map smooth and ordered.

```python
influence = np.exp(-(grid_distance ** 2) / (2 * sigma ** 2))
weights += influence * learning_rate * (x - weights)
```

How much a neuron moves depends on three things:

| Factor | Effect |
|---|---|
| `learning_rate` (α) | The overall size of the step |
| `influence` | How close the neuron is to the BMU **on the grid**, not in the data |
| `(x - w)` | The direction — towards the input |

The manual's illustration: if the input is blue and the BMU is light blue, the
winner becomes a bit bluer; if its neighbours are yellow, they gain a little
blueness too.

### Decay

Both the learning rate and the neighbourhood radius `sigma` shrink as training
proceeds:

```python
learning_rate = self.learning_rate * np.exp(-iteration / max_iter)
sigma = self.sigma * np.exp(-iteration / max_iter)
```

Early on, a wide neighbourhood drags whole regions of the map around at once, so
the map finds its rough global arrangement. Later, updates are small and local
and only fine-tune it. Choosing `sigma` matters: too small from the start and
the map never orders itself globally, some neurons never win anything (**dead
neurons**), and the model overfits.

### Reading a trained map

| Visualisation | What it shows |
|---|---|
| **U-matrix** | Mean distance from each neuron's weights to its neighbours'. Light = inside a cluster, dark = a boundary between clusters. Needs no labels. |
| **Component plane** | One feature's weight across the whole map — which features drive the structure. |
| **Class map** | Each neuron coloured by the majority label of the samples that landed on it. Needs labels, and is only used *after* training. |

### Evaluation metrics

Two standard SOM metrics need no labels at all:

* **Quantisation error** — the mean distance from a sample to its BMU. How
  tightly the map fits the data. Lower is tighter.
* **Topographic error** — the fraction of samples whose best and second-best
  neurons are *not* adjacent on the grid. How well the topology was preserved.
  Below about 5% is normally considered well ordered.

## What each activity does

All five activities use the Iris dataset.

### Activity 1 — Data pre-processing
Iris is loaded from `sklearn` and rescaled to 0–1 with `MinMaxScaler`. Scaling
matters because the BMU is chosen by Euclidean distance, so an unscaled feature
with a wide range would dominate the map.

### Activity 2 — Implementing the SOM
The `SOM` class from the manual: `_find_bmu` for the competition phase,
`_update_weights` for the adaptation phase with both decays, and `train` to loop
over the data.

### Activity 3 — Training
A 10x10 grid over the 4 Iris features, trained for 100 iterations.

### Activity 4 — Visualization
Iris has exactly 4 features and the weights sit in 0–1, so each neuron's weight
vector is read directly as an **RGBA colour**. The result is a smooth colour
gradient across the grid, which is the topology preservation made visible.

### Activity 5 — Clustering
Each sample is assigned to its BMU, and the BMU coordinates act as the cluster
label. A short extra cell counts which species landed on each neuron: the map
never saw the species, but the neurons still come out mostly single-species.

## What the task does

### Task 1 — Build and train a SOM on a downloaded dataset
The manual asks for a dataset from Kaggle or UCI. **Banknote Authentication**
(UCI) is used — the same dataset the manual's own description works through.
1372 rows, 4 features from wavelet transforms of banknote photographs, and a
binary label (0 = genuine, 1 = forged). A copy is saved in this folder so the
notebook runs offline.

The notebook works through: loading and inspecting the data, preprocessing,
a vectorised `SOM` implementation, training, the four visualisations, and
evaluation on a held-out test set. It ends with the **written report** the
manual asks for, covering preprocessing, architecture and hyperparameter
choices, the training process and metrics, and the observations.

Headline results:

| Metric | Value |
|---|---|
| Quantisation error | 0.1720 → 0.0954 (train), 0.0932 (test) |
| Topographic error | 0.0246 |
| Neurons that won a sample | 125 of 144 |
| Mean neuron purity | 0.9912 (118 of 125 fully pure) |
| Test accuracy | 0.9891 (272 of 275) |

The map was trained **without labels**; the labels are used only afterwards to
colour the map and measure it.

## Note on where the report lives

The manual asks for a notebook *and* a separate 1–2 page report. This repository
keeps every lab in notebook form, so the report is the final markdown section of
`tasks_ANN_lab_04.ipynb`, with a heading per required point. It can be exported
on its own with **File → Download as → Markdown / PDF**.

## Notes on the implementation

The task notebook changes three things from the manual's version. The activities
notebook keeps the manual's code as it was written.

**The neighbourhood uses the squared distance.** The standard Gaussian
neighbourhood is `exp(-d² / 2σ²)`. The manual writes
`np.exp(-distance / (2 * (sigma ** 2)))`, without squaring `d`. The task
notebook uses the standard form.

**The update is vectorised.** The manual loops over every neuron in Python for
every sample. On 1097 rows that runs into minutes per training run; computing
the grid distances as one array brings it down to a few seconds. The arithmetic
is unchanged.

**The scatter colour is wrapped in a list.** Activity 4 passes a neuron's
4-value weight vector to `plt.scatter(..., c=som.weights[i, j])`. Matplotlib
cannot tell whether four numbers mean one RGBA colour or four separate values
and warns once per point — 100 warnings for a 10x10 grid. Writing
`c=[som.weights[i, j]]` states that it is a single colour and the plot is
identical.

The notebooks fix the random seed before training. SOM weights start random, so
without a fixed seed every run would give a different map and different numbers
from the ones quoted above and in the report.
