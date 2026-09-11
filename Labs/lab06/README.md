# Lab 06 — Learning Vector Quantization

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Implementing and training a Learning Vector Quantization (LVQ)
network — a supervised, prototype-based classifier.

> **Note on numbering.** The manual in this folder is filed as Lab 06 and its
> pages are headed *Lab 05*. The previous lab has the same mismatch the other
> way round: it is filed as Lab 05 and headed *Lab 06*. The two are swapped in
> the original course manual (LVQ is on pages 56–62, Keras on 63–70). The folder
> follows the file name, so this is `lab06`.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 06.pdf` | The lab manual: concepts, the solved activity, and the graded task. |
| `activities_ANN_lab_06.ipynb` | The solved activity from the manual (done in class), with its bugs identified and fixed. |
| `tasks_ANN_lab_06.ipynb` | **The graded work** — Lab Task 1 and the written report, all cells executed with outputs. |
| `seeds_dataset.csv` | Seeds (UCI) — 210 wheat kernels, 7 features, 3 varieties. Used by the task. |

## How to run

Open `tasks_ANN_lab_06.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). The CSV must stay in the same folder as
the notebook.

```
pip install numpy pandas matplotlib scikit-learn
```

`scikit-learn` is used only for the train/test split, the scaler and the
cross-validation folds. The LVQ itself, the PCA projection and both baselines
are written from scratch with NumPy. Both notebooks run in well under a minute.

## What LVQ is

LVQ is a **supervised**, prototype-based classifier trained by competitive
learning. It has two layers: an input layer with one node per feature, and an
output layer with one node per **prototype**. Each prototype is a point in the
same space as the data and carries a fixed class label.

It is the supervised counterpart of the Self-Organising Map from Lab 04:

| | SOM (Lab 04) | LVQ (this lab) |
|---|---|---|
| Learning | Unsupervised — labels never used | **Supervised** — labels drive every update |
| The winner | Always moves **towards** the input | Moves towards it **only if the labels match**, otherwise **away** |
| Neighbours | Move as well | Only the winner moves |
| Purpose | Map and visualise structure | Classify |

### The algorithm

1. **Initialise** the prototypes from training samples — the manual takes the
   first sample of each class, one prototype per class.
2. For each training example, find the **winner**: the prototype closest to it
   by Euclidean distance.
3. **Reward or punish**:

> **w<sub>j</sub>(t+1) = w<sub>j</sub>(t) ± α (x<sub>k</sub> − w<sub>j</sub>(t))**

   `+` when the winner's label matches the example (pull the prototype towards
   it), `−` when it does not (push it away).
4. Repeat for several epochs, decaying α so late updates are small and the
   prototypes settle.

A new example is labelled with the class of whichever prototype wins.

### Why scaling matters so much here

LVQ decides *everything* by Euclidean distance — which prototype wins, and so
which prototype moves. A feature with a wide numeric range dominates that
distance purely because of its units. In the seeds data `area` spans about 10.6
to 21.2 while `compactness` spans 0.81 to 0.92, so without standardisation the
model would effectively see only `area`.

## What the activity does

### Activity 1 — Implementation of LVQ in Python
The manual's `LVQ` class with its `winner` and `update` methods, and a driver
that trains on six 4-bit samples in two classes.

**The code as printed in the manual does not run, and two further lines are
wrong.** The notebook lists all four problems, builds a corrected version, and
then demonstrates each of the silent bugs:

| # | As printed | Problem | Fix |
|---|---|---|---|
| 1 | `if actual -- j:` | `--` is not a comparison and `j` is never defined (the variable is `J`). Raises `NameError`. | `if actual == J:` |
| 2 | the `if`/`else` in `update` | Mis-indented relative to each other — `IndentationError`. | Align them |
| 3 | `if D0 > D1: return 0 else: return 1` | **Returns the prototype that is further away.** If `D0 > D1` then prototype 1 is nearer, so 1 should win. | `return 0 if D0 < D1 else 1` |
| 4 | `for i in range(len(weights))` in `update` | `len(weights)` is the number of **prototypes** (2), not **features** (4). Features 2 and 3 are never updated. | `for i in range(len(sample))` |

Problems 1 and 2 stop the code running. Problems 3 and 4 let it run while doing
the wrong thing, so the notebook proves each one:

* **Bug 3** is shown with a sample that *is* prototype 1, so its distance to that
  prototype is exactly 0. The correct winner is obviously 1; the manual's rule
  returns 0.
* **Bug 4** is shown by moving a prototype from `[0,0,0,0]` halfway towards
  `[1,1,1,1]`. The result should be `[0.5, 0.5, 0.5, 0.5]`; the manual's version
  gives `[0.5, 0.5, 0.0, 0.0]`.

**Why this was easy to miss.** The manual's buggy version still answers *class 0*
for its test sample — the same answer the corrected version gives. With only two
prototypes an inverted winner often just swaps which prototype is which. The
trained weights give it away though: the manual's end with `1, 1` and `0, 0` in
the last two columns, still the integers they started as.

## What the task does

### Task 1 — Build and train LVQ on a downloaded dataset
The manual asks for a dataset from Kaggle or UCI. **Seeds (UCI)** is used: 210
wheat kernels, 7 continuous geometric features, three varieties (Kama, Rosa,
Canadian) with exactly 70 of each. It suits LVQ because the classes are
balanced, every feature is numeric with no missing values, and there are three
of them — so it exercises the multi-class behaviour the manual's two-prototype
example cannot show.

*(The task text says "train SOMs algorithm on the dataset", carried over from the
previous lab. The task title and the rest of the lab are LVQ, so LVQ is what is
implemented.)*

The notebook covers: loading and inspecting the data, preprocessing, an LVQ
written from scratch for any number of classes and prototypes, choosing the
prototype count by cross-validation, training, the learning curve, a PCA view of
where the prototypes ended up, evaluation on a held-out test set, a seed
sensitivity check, and a comparison against two baselines. It ends with the
written report the manual asks for.

### Results

| Method | Points stored | Test accuracy |
|---|---|---|
| Nearest centroid (no learning) | 3 | 0.8571 |
| **LVQ (3 prototypes)** | **3** | **0.8635** (mean of 5 seeds, sd 0.0078) |
| 1-nearest-neighbour | 147 | 0.9048 |

Cross-validation on the training set chose **one prototype per class**;
accuracy fell steadily as prototypes were added (0.9298 at k=1 down to 0.9070 at
k=4), so the simplest model won on merit.

### The main finding

**LVQ rediscovered the class centroids rather than improving on them.** Each
learned prototype ends up **0.15–0.20** away from its class centroid, while a
typical member of that class sits **1.26–1.35** away — the prototypes are six to
nine times closer to the class average than an average kernel is. That is why
LVQ (0.8635) barely beats the no-learning nearest-centroid baseline (0.8571).

The learning curve says the same thing from another direction: training accuracy
is at its **maximum on the first epoch** and then moves inside a band four
samples wide. With one prototype per class, initialised inside its own class, on
three roughly spherical well-separated clusters, there is very little for the
reward-and-punish steps to fix.

The real result is the **compression**: 0.8635 accuracy from three stored points
instead of 147, a 49x reduction, giving up about four points of accuracy against
1-NN. Those remaining errors are concentrated in Kama, which sits *between* the
other two varieties in the projection — 4 of its 21 test kernels go to Canadian.
A single prototype cannot trace a boundary that shape; 1-NN, with every training
point available, can.

## Note on where the report lives

The manual asks for a notebook *and* a 1–2 page report. This repository keeps
every lab in notebook form, so the report is the final markdown section of
`tasks_ANN_lab_06.ipynb`, with a heading per required point: data preprocessing,
model architecture and hyperparameter rationale, the training process and
evaluation metrics, and the observations. It can be exported with
**File → Download as → Markdown / PDF**.

## Notes on the implementation

**The task implements LVQ from scratch rather than reusing the activity's
class.** The manual's version is hard-wired to exactly two prototypes — `winner`
compares `D0` and `D1` by name — and LVQ is at its most useful on multi-class
problems. The task version takes any number of classes and any number of
prototypes per class.

**The prototype count is chosen by cross-validation, not on the test set.**
`k=2` would in fact have scored higher on this particular test set than the
`k=1` cross-validation selected. Switching to it would be choosing a model by
looking at the answer, and the difference involved — under two points on 63
samples, about one kernel — is well inside the noise. The test set is evaluated
once, at the end.

**Baselines are computed with NumPy rather than scikit-learn.** `NearestCentroid`
and `KNeighborsClassifier` both pull in joblib, which emits a "could not find
the number of physical cores" warning on this machine. Both baselines are three
lines of NumPy, so the notebook computes them directly and stays clean.

**Seeds are fixed** before training, since prototypes are initialised at randomly
chosen samples. Section 10 deliberately varies the seed to show how much of the
headline accuracy is method and how much is luck.
