# Lab 03 — The Perceptron

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Implementing and training a perceptron, the basic building block of a
neural network.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 03.pdf` | The lab manual: concepts, solved activities, and the graded tasks. |
| `activities_ANN_lab_03.ipynb` | The 2 solved activities from the manual (done in class), on the Iris dataset. |
| `tasks_ANN_lab_03.ipynb` | **The graded work** — Lab Tasks 1 and 2, all cells executed with outputs. |
| `penguins.csv` | Palmer Penguins — 344 penguins, 3 species, 4 body measurements. Used by both tasks. |
| `breastcancer.csv` | Wisconsin Diagnostic Breast Cancer (UCI) — 569 samples, 30 features. Used by Task 2. |

## How to run

Open `tasks_ANN_lab_03.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). The two CSV files must stay in the same
folder as the notebook.

```
pip install numpy pandas matplotlib
```

The activities notebook downloads the Iris dataset from the UCI repository, so
it needs an internet connection. If the download fails it falls back to the copy
of Iris bundled with `scikit-learn`, so the notebook still runs offline.

Run the cells **in order** — the Perceptron class is defined once in Task 2 and
reused by every section after it.

## What a perceptron is

A perceptron is a supervised model that maps a vector of input features to a
single binary class label. It has one input node per feature, each with a
weight, and one output node.

```
x1 ──w1──┐
x2 ──w2──┤
x3 ──w3──┼──> Σ (weighted sum) ──> step function ──> 0 or 1
 1 ──b───┘
```

The output node takes the **weighted sum** of its inputs and passes it through a
**step function**, which returns `1` if the sum is positive and `0` otherwise.
That is the whole model — which is why it can only ever draw a single straight
boundary between two classes.

### The step function

```python
def step(x):
    return np.where(x > 0, 1, 0)
```

### The training procedure and the delta rule

Training feeds the whole training set through the network over and over. One
full pass over the data is called an **epoch**. After each prediction the
weights are corrected by the **delta rule**:

> **w(t+1) = w(t) + α (d − y) x**

| Term | Meaning |
|---|---|
| `d` | the target label (what the answer should be) |
| `y` | the predicted label (what the perceptron said) |
| `d − y` | `0` when the prediction is right, otherwise ±1 giving the direction to move |
| `x` | the input vector — the correction is proportional to the input |
| `α` | the **learning rate**, controlling how big each step is |

Because `d − y` is zero whenever the prediction is correct, **only mistakes
change the weights**. A correct prediction leaves the model untouched.

The learning rate matters. Too large and the weights overshoot the boundary and
oscillate; too small and training takes an impractical number of epochs. Common
values are `0.1`, `0.01` and `0.001`.

### The bias trick

The bias is the value that shifts the boundary away from the origin. Rather than
tracking it separately, a constant column of `1`s is appended to the input data:

```python
X = np.c_[X, np.ones((X.shape[0]))]   # bias trick
```

The bias then becomes just another weight in `W` and is trained by the same
delta rule as everything else. This is why a perceptron with `N` features has
`N + 1` weights.

### Weight initialisation

```python
self.W = np.random.randn(N + 1) / np.sqrt(N)
```

The weights start as random values from a normal distribution. Dividing by
`sqrt(N)` scales them down, which leads to faster convergence.

### The convergence property

> If the dataset is **linearly separable**, the perceptron is guaranteed to find
> a set of weights that classifies every sample correctly.

The other half of that statement is the perceptron's famous limitation: if the
classes *cannot* be separated by a straight line, it never converges. This is
why a single perceptron can learn the bitwise `OR` and `AND` functions but not
`XOR`, and it is what motivates multi-layer networks.

Both halves are visible in this lab — see the two learning curves in Task 2.

## What each activity does

### Activity 1 — Load and visualise the Iris dataset
The Iris dataset has 4 features and 3 classes. The last 50 rows
(`Iris-virginica`) are stripped off so that only `Iris-setosa` and
`Iris-versicolor` are left, because those two are linearly separable. The class
column is converted to `0` / `1` and the data is plotted as a scatter to confirm
a straight line can separate the classes.

### Activity 2 — Implement the perceptron
The `perceptron()` function from the manual: weights start at zero, and for each
epoch every sample is passed through `np.dot(w, x)` and a step comparison. When
the prediction is wrong the weights are updated and the mistake is counted. The
count per epoch is then plotted as a learning curve, which drops to zero.

## What each task does

### Task 1 — Loading and visualising a dataset
The manual asks for a dataset from Kaggle or the UCI repository.
**Palmer Penguins** (`penguins.csv`) is used.

- The dataset is loaded and inspected: shape, species counts, missing values.
- Because a perceptron is a **binary** classifier, the data is reduced to two
  classes — **Adelie** and **Gentoo** — the same way the manual drops the third
  Iris class. The Chinstrap rows are left out.
- Two features are kept, `flipper_length_mm` and `bill_depth_mm`, which is
  enough to separate those two species and few enough to draw on a flat scatter
  plot. Rows with missing measurements are dropped, leaving 274 of 344.
- Labels are encoded as **Adelie = 0**, **Gentoo = 1**.
- The scatter plot shows two clearly separated clusters — Gentoo penguins have
  longer flippers and shallower bills.

### Task 2 — Implementing and training the perceptron
A `Perceptron` class with `__init__`, `step`, `fit` and `predict`, following the
structure in the manual, plus an `errors_` list that records how many samples
were misclassified in each epoch so the learning curve can be drawn.

**On the penguins** (learning rate `0.01`, 20 epochs):

- The learning curve falls from 85 misclassified samples to **zero at epoch 8**
  and stays there — the data is linearly separable, so the convergence property
  holds. Once the curve reaches zero no prediction is wrong, so no weight update
  happens and nothing changes for the remaining epochs.
- Final training accuracy is **1.0000** (274 of 274).
- The three weights are printed: `flipper_length_mm` is positive, so a longer
  flipper pushes the output towards Gentoo; `bill_depth_mm` is negative, so a
  deeper bill pushes it towards Adelie. That matches the scatter plot.
- The decision boundary is drawn on top of the data by rearranging
  `w1x1 + w2x2 + b = 0` into `x2 = -(w1x1 + b) / w2`, which shows the line the
  weights represent.

**On the breast cancer dataset** (learning rate `0.001`, 30 epochs), with the
same class unchanged:

- The `id` column is dropped because an identifier carries no information, and
  so is the empty `Unnamed: 32` column at the end of the file. The diagnosis is
  encoded as **B (benign) = 0**, **M (malignant) = 1**.
- The learning curve falls quickly and then **flattens out above zero** instead
  of reaching it. These classes are not perfectly linearly separable, so a few
  samples sit on the wrong side of any straight line, keep triggering weight
  updates, and the error count oscillates forever. It is the same limitation
  that stops a perceptron from learning XOR.
- Final training accuracy is **0.9842**.
- With 30 features there are 31 weights, so they are sorted by magnitude and the
  10 largest are shown.

## Why the features are standardised

Each weight update is `alpha * error * x`, so the correction is proportional to
the size of the input. Flipper length runs to about 230 while bill depth is
around 15 — the larger feature dominates every step and throws the boundary a
long way past where it should be.

Subtracting the mean and dividing by the standard deviation puts both features
on the same footing:

```python
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
```

The notebook shows the difference directly by training the same perceptron on
the raw measurements: accuracy drops from **1.0000** to **0.4489**, barely
better than guessing. The per-epoch error count still looks low, because it only
counts mistakes *as they happen* during the pass — but the enormous updates mean
the final weights land somewhere useless. This is the overshooting the manual
warns about, caused here by the scale of the inputs rather than by `α` itself.

## Note on the implementation

The manual writes the step function as `return 1 if x > 0 else 0`. That works
for a single value, but `predict()` calls it on a whole array of dot products at
once, and comparing an array with `>` returns an array of booleans rather than
one `True` or `False`. `np.where(x > 0, 1, 0)` behaves identically for a single
value and also handles the array case, so it is used instead.

The notebooks fix the random seed (`np.random.seed`) before training. The
starting weights are random, so without a fixed seed every run would produce
different weights, a different learning curve, and different numbers in the
text above.
