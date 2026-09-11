# Lab 07 — Recurrent Neural Networks in Keras

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Implementing and training recurrent neural networks with Keras.

> **Note on numbering.** The manual in this folder is filed as Lab 07 and its
> pages are headed *Lab 08*. The two previous labs have the same kind of
> mismatch — Lab 05's file is headed *Lab 06* and Lab 06's is headed *Lab 05*.
> The folder follows the file name, so this is `lab07`.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 07.pdf` | The lab manual: concepts, the solved activity, and the graded task. |
| `activities_ANN_lab_07.ipynb` | The solved activity from the manual (done in class) — the SimpleRNN tutorial it links to. |
| `tasks_ANN_lab_07.ipynb` | **The graded work** — Lab Task 1 and the written report, all cells executed with outputs. |
| `monthly-sunspots.csv` | Monthly sunspot counts from 1749, 2820 observations. Used by the activity. |

The graded task uses the **IMDb Large Movie Review Dataset**, which ships with
Keras as `keras.datasets.imdb` and is downloaded and cached on first use, so
there is no CSV for it in this folder.

## How to run

Open `tasks_ANN_lab_07.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`).

```
pip install tensorflow numpy pandas matplotlib scikit-learn
```

The activities notebook needs `monthly-sunspots.csv` to stay in the same folder.
The task notebook downloads IMDb on first run and needs an internet connection
for that one step only.

**Runtime.** The activity takes under a minute. The task notebook trains
thirteen recurrent networks on 25,000 reviews and takes roughly 15 minutes on a
CPU. Neither needs a GPU.

## What an RNN is

A recurrent network processes a **sequence** one element at a time, carrying a
hidden state from each step to the next. That hidden state is its memory: at
every step it combines what it has seen so far with the new input.

> **h<sub>t</sub> = tanh(W h<sub>t−1</sub> + U x<sub>t</sub>)**
> **y<sub>t</sub> = softmax(V h<sub>t</sub>)**

The same weights are reused at **every** time step. That is what makes the
network recurrent, and what lets one model handle sequences of any length —
unrolling an RNN over a 5-word sentence gives something like a 5-layer network
in which all five layers share one set of weights.

`tanh` is the usual activation here. Its second derivative decays slowly to
zero, which keeps gradients in the useful region of the function for longer.

### What a SimpleRNN layer contains

Exactly three things:

| Weight | Shape | Role |
|---|---|---|
| kernel (`wx`) | (features, units) | applied to the input at the current step |
| recurrent kernel (`wh`) | (units, units) | applied to the previous hidden state |
| bias (`bh`) | (units,) | added at every step |

The activity verifies this by recomputing a layer's output in five lines of
NumPy and checking it against `model.predict`.

### RNN topologies

Because sequences can appear at the input, the output, or both, RNNs support
shapes that an MLP or CNN cannot:

| Topology | Example |
|---|---|
| one → one | ordinary classification (no RNN needed) |
| one → many | image captioning: one image, a sentence out |
| **many → one** | **sentiment analysis — what the graded task does** |
| many → many | machine translation |
| synced many → many | labelling every frame of a video |

### Vanishing and exploding gradients

Training an RNN uses **backpropagation through time (BPTT)**: because the
parameters are shared across time steps, the gradient at the output depends on
every earlier step too.

That chain is the problem. The influence of an early input reaches the output
only after being multiplied by the recurrent matrix once per step. If those
values are below 1 they shrink exponentially and **vanish**; above 1 they
**explode**. Either way a plain RNN struggles to connect information that is far
apart in a sequence — the long-term dependency problem.

**LSTM** was designed to fix exactly this. Instead of one `tanh` layer per step
it uses four interacting layers and a cell state that information can travel
along without being repeatedly multiplied. The graded task tests whether that
difference actually shows up.

## What the activity does

### Activity 1 — the SimpleRNN tutorial
The manual asks the student to work through the *Understanding Simple Recurrent
Neural Networks in Keras* tutorial. The notebook does it in two halves.

**Part 1: taking the layer apart.** A network with 2 hidden units and 3 time
steps, with `activation='linear'` so the arithmetic stays transparent. The three
weight matrices are printed, then the sequence `[1, 2, 3]` is pushed through the
network *and* through the equations by hand. They match exactly, which confirms
there is no hidden machinery — one input matrix, one recurrent matrix, one bias,
reapplied at each step.

It also makes the vanishing gradient concrete: `h1` depends only on `x1`, but
`h3` depends on `x1` through **two** passes of the recurrent matrix.

**Part 2: predicting sunspots.** The series is scaled to 0–1, split by time
(never shuffled — the test set has to be the future), and cut into 12-month
windows. A 3-unit `SimpleRNN` with a linear output is trained for 20 epochs.

### An honest result from the activity

The tutorial reports the RMSE and stops. The notebook adds the two baselines any
forecast should be measured against, and the picture changes:

| | train RMSE | test RMSE |
|---|---|---|
| SimpleRNN | 0.071 | 0.107 |
| Persistence — "next = latest" | 0.063 | **0.080** |
| Always predict the mean | — | 0.246 |

The network has clearly learned something: it is far better than ignoring the
input, and the plot tracks every rise and fall of the cycle. **But it does not
beat persistence**, on either split.

Two reasons, both visible in the notebook:

* **It undershoots the peaks.** Squared error is dominated by the large misses
  at the spikes, and a small network trained on MSE hedges towards the middle of
  the range — safe on average, expensive at the extremes.
* **There is very little to train on.** `get_XY` takes only every 12th value as
  a target, so 2256 monthly observations become just **187 training examples**
  for a network being asked to learn an 11-year cycle. Sunspot counts in
  adjacent months are also strongly correlated, which makes persistence a hard
  baseline.

This does not make the activity wrong — it demonstrates the mechanics of
`SimpleRNN` exactly as intended. It is a reminder that a plot which "tracks the
data" is not evidence of a useful model until it is compared with the trivial
alternative.

## What the task does

### Task 1 — a movie review classifier with a SimpleRNN
The manual asks for IMDb reviews from Stanford AI Lab's Large Movie Review
Dataset — the dataset Keras ships as `keras.datasets.imdb`. 50,000 reviews,
25,000 for training and 25,000 for testing, exactly balanced between positive
and negative, already tokenised into integer word indices.

This is the **many → one** topology: a whole review goes in, one sentiment score
comes out.

**Preprocessing.** The vocabulary is capped at the 10,000 most frequent words;
rarer ones become the unknown token, since there are too few examples to learn
an embedding for them. Reviews are then padded or truncated to a fixed 200
tokens — the median review is 178 and the 90th percentile is 467, so 200 keeps
most reviews nearly whole while keeping sequences short enough for a SimpleRNN
to propagate gradients through. Keras pads at the **front** by default, which
suits a layer that returns only its final state: the end of the review stays
closest to the output.

**Architecture.** `Embedding(10000, 32)` → `SimpleRNN(32)` → `Dense(1, sigmoid)`,
with `binary_crossentropy` and `adam`. The layers are deliberately small: the
recurrent matrix is applied 200 times over, so a large one makes the gradient
problems worse rather than better.

**The experiment.** The manual's central claim is that plain RNNs cannot hold
information across long sequences, and that LSTM exists to fix it. That is
testable, so the notebook trains the identical architecture with `SimpleRNN` and
with `LSTM` at three sequence lengths, then repeats the headline comparison with
three seeds to check the gap is larger than run-to-run noise.

### Results

| Layer | 100 tokens | 200 | 400 |
|---|---|---|---|
| SimpleRNN | 0.8173 | 0.8368 | 0.8337 |
| LSTM | 0.8316 | **0.8624** | 0.8588 |

LSTM wins at every length, and the gap widens from 1.4 points at 100 tokens to
2.6 at 200 — the direction the manual predicts.

**But the claim is only partly borne out.** If long sequences were simply harder
for a plain RNN, accuracy should keep falling as the window grows. It does not:
both layers dip slightly from 200 to 400 tokens, and the gap stops widening
(2.6 points at 200, 2.5 at 400). Going from 200 to 400 adds mostly padding for
most reviews while doubling the number of steps, so neither layer gains.

Repeating the 200-token comparison across three seeds:

| Layer | Mean | Std | Range |
|---|---|---|---|
| SimpleRNN | 0.8301 | 0.0066 | 0.8235 – 0.8368 |
| LSTM | 0.8484 | 0.0147 | 0.8331 – 0.8624 |

The +0.0183 advantage is 2.8× the SimpleRNN's run-to-run spread, so it is not
one lucky run — but the **ranges overlap**: the LSTM's worst seed (0.8331) came
in below the SimpleRNN's best (0.8368), and the LSTM is the more variable of the
two. The supportable claim is that LSTM is better *on average* by a small
margin, not that it wins every time.

### The most instructive failure

The headline model is trained for 5 epochs and scores **0.7500** on the test
set — *worse* than the 0.8368 the identical architecture and seed reach with
**3** epochs in the experiment above.

Nothing is wrong with the evaluation. `model.evaluate` uses the weights from the
**last** epoch, and epoch 5 was a bad place to stop: validation accuracy fell
from 0.8148 to 0.7438 in that single epoch while training accuracy climbed to
0.9674 and validation loss rose from 0.446 to 0.677. Two extra epochs cost
nearly nine percentage points.

`EarlyStopping(restore_best_weights=True)`, as used in Lab 05, would have kept
the epoch-4 weights. It is deliberately left out here so that the failure is
visible in the curves rather than quietly patched over — it is the clearest
demonstration of overfitting in these labs.

## Note on where the report lives

The manual asks for a notebook *and* a 1–2 page report. This repository keeps
every lab in notebook form, so the report is the final markdown section of
`tasks_ANN_lab_07.ipynb`, with a heading per required point: data preprocessing,
model architecture and hyperparameter rationale, the training process and
evaluation metrics, and the observations. It can be exported with
**File → Download as → Markdown / PDF**.

## Notes on the implementation

**`Input(shape=...)` rather than `input_shape=`.** The tutorial writes
`SimpleRNN(2, input_shape=(3,1))`. Keras 3 deprecates passing `input_shape` to a
layer and warns on every model built that way, so the notebooks use an explicit
`Input` layer. The resulting model is identical.

**No `clear_session()` in the activity.** Only two small models are built there,
so it bought nothing — and it is what prints the `tf.reset_default_graph is
deprecated` notice into the output. Removing it fixed the warning at its source
rather than suppressing it. The task notebook still calls it, where thirteen
models are trained in one kernel.

**Seeds are fixed** with `keras.utils.set_random_seed` before every model, since
weights start random. Where the question is specifically whether one layer beats
another, the notebook varies the seed deliberately and reports the spread rather
than a single run.
