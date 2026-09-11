# Lab 08 — Hopfield Networks

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Implementing a Hopfield network and using it to classify handwritten
digits.

> **Note on numbering.** The brief in this folder is titled *Lab 07 — Hopfield
> Network* and arrived as `Lab 07.pdf`. It is filed here as `Lab 08.pdf` to match
> the folder, since `lab07` already holds a different *Lab 07.pdf* (the RNN lab).
> Labs 05, 06 and 07 all carry similar mismatches between their file names and
> their internal headings.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 08.pdf` | The assignment brief — a single-page problem statement. |
| `activities_ANN_lab_08.ipynb` | Hopfield fundamentals, built up from scratch. The brief has no solved activities, so this is the groundwork the task needs. |
| `tasks_ANN_lab_08.ipynb` | **The graded work** — the digit classifier and the written report, all cells executed with outputs. |
| `Reduced MNIST Data/` | The dataset from Kaggle: 28x28 JPEGs, 1000 train and 200 test images per digit (12,000 files). |

The training folder is spelled `Reduced Trainging data` in the download. The
typo is left as-is so the paths work exactly as the dataset ships.

## How to run

Open `tasks_ANN_lab_08.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). The `Reduced MNIST Data` folder must stay
beside the notebook.

```
pip install numpy pandas matplotlib pillow
```

Everything is implemented directly in NumPy — there is no TensorFlow or
scikit-learn here, because a Hopfield network has nothing to train. Both
notebooks finish in well under a minute, apart from the first read of the 12,000
image files.

## What a Hopfield network is

An **associative memory**, not a classifier. Every neuron connects to every
other; there are no layers, no forward direction, and no output to train.
Patterns are written into the weights in one shot, and afterwards the network
*recalls* whichever stored pattern best matches what it is shown.

**State.** Each of the `N` neurons holds `+1` or `-1`. For a 28x28 image that is
784 neurons.

**Storage (Hebbian rule).** For patterns ξ¹ … ξᵖ:

> **W = (1/N) Σ ξᵘ (ξᵘ)ᵀ**, with **W<sub>ii</sub> = 0**

Neurons that agree across the stored patterns get a positive weight, those that
disagree a negative one. The diagonal is zeroed so no neuron drives itself.

**Recall.** Repeatedly set **s<sub>i</sub> ← sign(Σ<sub>j</sub> W<sub>ij</sub>
s<sub>j</sub>)** until nothing changes. The result is a **fixed point** — an
attractor.

**Energy.**

> **E = −½ Σ<sub>i≠j</sub> W<sub>ij</sub> s<sub>i</sub> s<sub>j</sub>**

The stored patterns sit at the bottoms of the valleys of this function. The
convergence guarantee has a condition that is easy to gloss over: energy is
non-increasing under **asynchronous** updates — one neuron at a time. Updating
**synchronously**, recomputing every neuron at once, is faster and usually
converges anyway, but carries no such guarantee and can fall into a two-state
cycle. Both are implemented in the activity notebook.

### The two limits

| Limit | What it says |
|---|---|
| **Capacity** | About **0.138N** patterns can be stored before recall collapses — roughly 108 for 784 neurons. |
| **Correlation** | That figure assumes the patterns are **random and near-orthogonal**. Correlated patterns break the network far below capacity. |

The second limit is the one that decides this lab.

## What the activity notebook does

The brief contains no solved activities, so this notebook establishes the
groundwork the task depends on.

**Storing and recalling.** Three 8x8 patterns — a cross, a hollow square and a
diagonal — stored in a 64-neuron network. All three are stable fixed points, and
the network recovers the original **exactly** from 10%, 25% and even 40% of
pixels flipped. This is the classic demonstration and it works.

**Energy only goes down.** An asynchronous recall from a 35%-damaged pattern,
with the energy measured after every single-neuron update: 128 updates, energy
−1.88 → −33.00, monotonically non-increasing at every step. The synchronous
version reaches the same place in one sweep.

**Limit 1 — capacity.** Storing `p` random patterns and checking how many remain
stable:

| Patterns stored | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|
| Still stable | 100% | 100% | 100% | 89% | 85% | 63% | 30% | 10% |

The decline begins right around the predicted 0.138 × 64 ≈ 8.8.

**Limit 2 — correlation.** Four patterns, far inside capacity, stored three
ways:

| Patterns | Mean overlap | Stable |
|---|---|---|
| Random (near-orthogonal) | 0.09 | 4 of 4 |
| Correlated, 30% different | 0.23 | 4 of 4 |
| Correlated, 15% different | 0.53 | **0 of 4** |

Same network, same number of patterns — correlation alone destroys the memory.
This is what the graded task runs into.

## What the task notebook does

**The brief:** implement a Hopfield network for the classification of
handwritten digits from MNIST, using the Reduced MNIST dataset.

**Preprocessing.** Images are binarised at a threshold of 128 (a Hopfield neuron
has no room for a grey value — 19.1% of pixels survive as ink) and flattened to
784-element vectors. Each digit is then condensed into a **single prototype** by
pixel-wise majority vote over its 1000 training images, giving ten patterns to
store.

**Two storage rules** are implemented: the classic Hebbian rule, and the
pseudo-inverse (projection) rule, which divides out the overlap between patterns
instead of simply summing them.

### Results

| Method | Test accuracy |
|---|---|
| Chance (10 balanced classes) | 0.1000 |
| Hopfield, **Hebbian** storage | **0.1000** |
| Hopfield, **pseudo-inverse** storage | **0.8000** |
| Direct prototype match, no network at all | **0.8110** |

### What the numbers mean

**The Hebbian network collapses entirely.** Accuracy is exactly chance, all 2000
test images are classified as the same digit, and across every input the network
reaches only **two distinct final states**. Not one of the ten prototypes is a
fixed point of its own network. The ten energy valleys have merged into one.

**The cause is correlation, not capacity.** Ten patterns in 784 neurons is a
tenth of the limit. But the prototypes have a **mean overlap of 0.685** (maximum
0.865, between digits 4 and 9) — every digit is a blob of ink in the middle of
the same frame. The Hebbian rule adds the patterns together, so what they share
reinforces and what distinguishes them is drowned out. The activity notebook
shows the identical collapse at an overlap of just 0.53.

**The pseudo-inverse rule repairs the memory completely.** All ten prototypes
become stable, and **94.9%** of test images settle exactly onto one of them. As
an associative memory, the network now does its job properly.

**But as a classifier it still loses to the trivial baseline.** 0.8000 against
the 0.8110 of just comparing each image to the ten prototypes — no network, no
dynamics, one matrix multiplication.

**And it does not catch up on damaged input**, which is the test most favourable
to it:

| Pixels flipped | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Direct match | 0.8110 | 0.7905 | 0.7540 | 0.6785 | 0.4580 |
| Hopfield first | 0.8000 | 0.7810 | 0.7485 | 0.6560 | 0.4200 |

Repairing corrupted patterns is exactly what a Hopfield network is for, so the
expectation was that it would pull ahead as the damage grew. It does the
opposite.

**Why it can only lose.** Direct matching compares the image with each prototype
on a continuous scale. The network forces a hard discrete decision first —
`sign()` discards how strongly each neuron was driven, and the state snaps to
one attractor. Snap to the right one and the comparison would have got there
anyway; snap to the wrong one and the evidence that would have corrected it is
already gone.

**The honest conclusion.** A Hopfield network is a content-addressable memory,
and at that job it works: after the storage rule is fixed it recalls ten stored
digits from partial input almost perfectly. Classification is a different job,
and the architecture has no mechanism for it — no output layer, no decision
boundary, nothing trainable against labels. The ~0.80 ceiling comes from
representing each digit with a single template, not from the network wrapped
around it.

## Note on where the report lives

The brief asks only for an implementation, but the other labs in this repository
each carry a written report, so the same structure is kept: the report is the
final markdown section of `tasks_ANN_lab_08.ipynb`, covering data preprocessing,
model architecture and design decisions, results and evaluation metrics, and the
observations. It can be exported with **File → Download as → Markdown / PDF**.

## Notes on the implementation

**Recall is synchronous in the task notebook.** It converges in a handful of
iterations and lets all 2000 test images be processed as a single matrix
operation. The asynchronous version — the one the energy guarantee actually
applies to — is implemented and demonstrated in the activity notebook.

**The pseudo-inverse rule is an addition, not a substitution.** The Hebbian rule
is the one in the lectures, and it is run first and reported in full precisely
because its failure is the most informative result in this lab. The
pseudo-inverse rule is then introduced to show that the failure is caused by the
storage rule rather than by Hopfield networks as such.

**`np.linalg.pinv` is used rather than `inv`.** With ten heavily correlated
patterns the overlap matrix is close to singular, and the pseudo-inverse handles
that without blowing up.
