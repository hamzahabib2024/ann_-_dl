# Lab 05 — Artificial Neural Networks with Keras

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** Building, training and tuning artificial neural networks with Keras.

> **Note on numbering.** The manual in this folder is filed as Lab 05 and its
> pages are headed *Lab 06*. The folder follows the file name, so this is
> `lab05`. It is the Keras lab either way.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 05.pdf` | The lab manual: concepts, solved activities, and the graded tasks. |
| `activities_ANN_lab_05.ipynb` | The 2 solved activities from the manual (done in class), on MNIST. |
| `tasks_ANN_lab_05.ipynb` | **The graded work** — Lab Tasks 1 and 2 with both written reports, all cells executed with outputs. |
| `winequality-red.csv` | Wine Quality (UCI) — 1599 red wines. Used by the tasks. |
| `winequality-white.csv` | Wine Quality (UCI) — 4898 white wines. Used by the tasks. |
| `winequality.names` | The dataset's own documentation and citation request, as shipped by UCI. |

## How to run

Open `tasks_ANN_lab_05.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). The two CSV files must stay in the same
folder as the notebook.

```
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Run the cells **in order** — the data split and the scaler are created once in
Task 1 and reused by every experiment in Task 2.

**Runtime.** The task notebook trains about 20 small networks on 4547 rows and
finishes in a few minutes. The activities notebook is much heavier: it trains 5
networks on 60,000 MNIST images for 60 epochs in total, which takes roughly
10–15 minutes on a CPU. Neither needs a GPU.

## Keras in three steps

**Keras** is a high-level API for neural networks that runs on top of
TensorFlow. The **Sequential API** stacks layers one after another, and building
any model follows the same three steps.

### 1. Build

```python
from keras import Sequential
from keras.layers import Input, Dense

model = Sequential([
    Input(shape=(12,)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid"),
])
```

Common layer types:

| Layer | What it is for |
|---|---|
| `Dense` | Fully connected — every input connects to every neuron |
| `Flatten` | Collapses a multi-dimensional input into one dimension |
| `Dropout` | Randomly zeroes a fraction of activations during training, to reduce overfitting |
| `Conv2D` | Processes images — used in convolutional networks |
| `MaxPooling` | Down-samples feature maps by taking the maximum in each block |
| `Embedding` | Maps words or categories to dense vectors |

### 2. Compile — choose a loss and an optimizer

| Loss function | Use it for |
|---|---|
| `mean_squared_error` | Regression — predicting a number |
| `binary_crossentropy` | Two classes, with **one sigmoid** output neuron |
| `categorical_crossentropy` | More than two classes, with a **softmax** output layer |

| Optimizer | How it works |
|---|---|
| `sgd` | Plain gradient descent; one fixed learning rate for every weight |
| `adam` | Adapts the learning rate per weight from the gradient history — the usual default |
| `rmsprop` | Normalises updates by a moving average of squared gradients |

Matching the loss to the output layer is the part that is easy to get wrong:
a sigmoid output needs `binary_crossentropy`, a softmax output needs
`categorical_crossentropy`.

### 3. Fit and evaluate

```python
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
```

`fit` returns a **history** object holding the loss and metrics for every epoch,
which is what the training curves in these notebooks are drawn from.

## Key ideas this lab demonstrates

**Activation functions.** ReLU passes positive values through unchanged and
zeroes negatives. Sigmoid squashes everything into 0–1, and its gradient becomes
almost zero once a neuron saturates — so gradients shrink as they propagate
backwards through the layers, the **vanishing gradient** problem. Activity 2
tests this, and the result is worth reading carefully: with only two hidden
layers sigmoid ends up at essentially the same accuracy as ReLU. The cost shows
up in **speed** instead. Vanishing gradients need depth to bite.

**Optimizers.** Adam adapts a learning rate for each weight individually; plain
SGD uses one fixed rate for all of them. Given the same number of epochs, adam
normally gets much further.

**Overfitting.** When training accuracy keeps climbing while validation accuracy
flattens, the network is memorising the training rows rather than learning
anything that transfers. The gap between the two curves is the symptom to watch.

**Dropout** and **early stopping** are the two defences used in Task 2. Dropout
randomly switches off a fraction of neurons during training so the network
cannot lean on any single one. Early stopping watches validation accuracy and
restores the weights from the best epoch instead of whichever epoch happened to
be last.

## What each activity does

Both activities use MNIST — 70,000 handwritten digit images, 28x28 greyscale,
10 classes. Pixels are scaled to 0–1 and labels are one-hot encoded.

### Activity 1 — Implementing an ANN in Keras
`Flatten` turns each image into 784 values, two hidden ReLU layers (128 and 64)
do the work, and a 10-neuron softmax produces class probabilities. Compiled with
adam and `categorical_crossentropy`, trained for 10 epochs with batch size 32
and a 20% validation split, then evaluated on the test set.

### Activity 2 — Experiments
Four experiments, each changing exactly one thing so its effect can be read on
its own:

| Experiment | Change |
|---|---|
| 1 | More neurons — 256 and 128 instead of 128 and 64 |
| 2 | Sigmoid activation instead of ReLU |
| 3 | SGD optimizer instead of adam |
| 4 | 20 epochs instead of 10 |

A summary table and a bar chart compare all five runs at the end, including the
train-minus-validation gap so the overfitting is visible alongside the accuracy.

| Experiment | Test accuracy | vs baseline |
|---|---|---|
| Baseline (128-64, relu, adam, 10 epochs) | 0.9748 | — |
| 1. More neurons (256-128) | 0.9759 | +0.0011 |
| 2. Sigmoid activation | 0.9745 | −0.0003 |
| 3. SGD optimizer | 0.9589 | −0.0159 |
| 4. 20 epochs | 0.9756 | +0.0008 |

Only one of the four changes made a real difference, and it made things worse.
The notebook ends with a section reading each result against what the theory
predicts:

* **More neurons** changed almost nothing — an MLP on MNIST is already near its
  ceiling, so extra capacity has nothing left to buy.
* **Sigmoid** cost almost no accuracy but was clearly slower: 0.8715 training
  accuracy after one epoch against ReLU's 0.9236, needing most of the ten epochs
  to catch up. Two hidden layers is too shallow for gradients to vanish through.
* **SGD** did not fail, it **underfitted** — it finished at 0.9624 *training*
  accuracy against the baseline's 0.9934, with validation accuracy still
  climbing when the epochs ran out. Adam reaches in one epoch roughly what SGD
  reaches in four.
* **20 epochs** bought nothing: across the extra ten epochs training accuracy
  rose from 0.9934 to 0.9966 while validation accuracy stayed flat near 0.974
  and validation **loss rose** from 0.1083 to 0.1570. That is the clearest
  overfitting signal in the lab, and the argument for the early stopping used in
  Task 2.

## What each task does

### Task 1 — Building and training a neural network
The manual asks for the **Wine Quality** dataset from UCI. Both provided files
are used — 1599 red and 4898 white wines, combined into 6497 rows with a
`wine_type` feature so the colour is kept rather than discarded.

Quality is a taster's score from 3 to 9, heavily concentrated on 5 and 6, so the
task is framed as **binary classification**: a wine is *good* if it scores 6 or
above. That gives 4113 good against 2384 not good, so the majority-class
baseline to beat is **0.6331**.

The data is split 70/15/15, stratified, **before** scaling, and `StandardScaler`
is fitted on the training rows only. The model is 64-32 ReLU with a single
sigmoid output, adam, `binary_crossentropy`, 50 epochs at batch size 32.

### Task 2 — Enhancing network performance
Eight configurations covering everything the manual asks for — number of
neurons, number of layers, batch size, dropout, and the RMSprop and
SGD-with-momentum optimizers. Each changes one thing at a time, and all are
judged on the **validation** set; the test set is consulted only at the end.

The tuned model combines what the experiments showed rather than taking the
single best row: 128-64 with dropout 0.2, batch size 128, adam, and early
stopping.

Both models are then retrained with **five random seeds each**, because a single
run cannot distinguish a real improvement from the run-to-run variation caused
by random weight initialisation.

### Results

| Model | Mean test accuracy (5 seeds) | Std dev |
|---|---|---|
| Majority-class baseline | 0.6331 | — |
| Task 1 baseline network | 0.7659 | 0.0103 |
| Task 2 tuned network | 0.7778 | 0.0079 |

The tuning gain is **about 1.2 percentage points** — real, but only a little
larger than the spread between seeds, which is exactly why it was measured
across five of them. The tuned model is also more consistent.

All eight configurations landed within 1.6 percentage points of each other. When
widening, deepening, regularising and swapping the optimizer all give the same
answer, the limit is the data rather than the architecture: wine quality is a
subjective human score, and some of the error is irreducible.

## Note on where the reports live

The manual asks for a notebook *and* a 1–2 page report for each task. This
repository keeps every lab in notebook form, so both reports are markdown
sections inside `tasks_ANN_lab_05.ipynb`:

* **Report — Lab Task 1**: data preprocessing, model architecture and the
  rationale for each hyperparameter, the training process and evaluation
  metrics, and the observations.
* **Report — Lab Task 2**: initial performance, the techniques applied, the
  improved performance, and the observations.

They can be exported with **File → Download as → Markdown / PDF**.

## Notes on the implementation

**`Input(shape=...)` instead of `Flatten(input_shape=...)`.** The manual writes
`Flatten(input_shape=(28, 28))` as the first layer. Keras 3 — the version
installed here — deprecates passing `input_shape` to a layer and warns on every
model built that way. The notebooks use an explicit `Input` layer followed by a
bare `Flatten()`, which is the recommended form and produces an identical model.

**MNIST is cast to `float32`.** Dividing a `uint8` array by 255 produces
`float64`, which makes the dataset take twice the memory for no benefit. On a
machine that is short of RAM this is the difference between training normally
and swapping to disk — the first attempt at running the activities notebook
stalled for exactly this reason.

**`keras.backend.clear_session()` before each model**, and each experiment's
model is released once its history has been saved, so memory does not build up
across five sequentially trained networks.

**Seeds are fixed** with `keras.utils.set_random_seed` before every model.
Network weights start random, so without a fixed seed each run would give
different numbers from the ones quoted above and in the reports. Where the
question is specifically whether a change helped, the notebook varies the seed
deliberately and reports the mean and spread instead.
