# Lab 01 — Basics of Python

**Course:** AIC 380 – Artificial Neural Networks
**Topic:** A recap of the Python basics needed before we start implementing neural networks.

## Files in this folder

| File | What it is |
|---|---|
| `Lab 01.pdf` | The lab manual: concepts, solved activities, and the graded tasks. |
| `activities_ANN_lab_01.ipynb` | The 6 solved activities from the manual (done in class). |
| `tasks_ANN_lab_01.ipynb` | **The graded work** — Lab Tasks 1, 2 and 3, all cells executed with outputs. |
| `students.pkl`, `students_class.pkl` | Binary files created by Task 3. Not written by hand — the notebook generates them. |

## How to run

Open `tasks_ANN_lab_01.ipynb` in Jupyter, VS Code, or Google Colab and run the
cells from top to bottom (`Run All`). No extra libraries are needed — everything
uses the Python standard library, and only `pickle` is imported.

Run the cells **in order**: Task 3 reuses the `students` list built in its own
first cell, and the pickle files are recreated every time the notebook is run.

## Python concepts covered in this lab

### Quick comparison

| | List | Tuple | Dictionary | Set |
|---|---|---|---|---|
| Written as | `[1, 2, 3]` | `(1, 2, 3)` | `{"a": 1}` | `{1, 2, 3}` |
| Ordered | Yes | Yes | Yes (insertion order) | No |
| Can be changed | Yes | **No** | Yes | Yes |
| Duplicates allowed | Yes | Yes | Keys must be unique | No |
| Accessed by | Index — `x[0]` | Index — `x[0]` | Key — `x["a"]` | No indexing |

### List
An ordered, changeable collection, like a one-dimensional array — but it can
hold mixed types and even other lists. Indexing starts at `0`, and negative
indices count from the end (`-1` is the last item).

```python
colors = ["red", "green", "blue"]
colors.append("orange")   # add at the end
colors.insert(1, "pink")  # add at a position
colors.remove("blue")     # delete by value
colors.sort()             # alphabetical order
len(colors)               # how many items
```

**Slicing** takes a piece of the list with `[start:stop:step]`. It is
*inclusive–exclusive*: the `start` item is included, the `stop` item is not.

```python
colors[1:3]   # items 1 and 2
colors[-3:]   # last three items
colors[::2]   # every second item
colors[:]     # a full copy
```

### Tuple
Same idea as a list, but **immutable** — once created it cannot be changed.
Trying to assign to an element raises `TypeError`. Tuples are used for data that
should stay fixed, and they can be *unpacked* into separate variables.

```python
numbers = (10, 20, 30, 40, 50)
numbers[2]                  # 30
numbers[1] = 99             # TypeError: tuples are immutable
bigger = numbers + (60, 70) # you build a NEW tuple instead
a, b, c, d, e = numbers     # unpacking
```

### Dictionary
Stores **key : value** pairs (an associative array / hash table). You look items
up by key instead of by position, which makes the lookup fast and the code
readable.

```python
person = {"name": "John", "age": 25}
person["email"] = "j@x.com"   # add a new pair
person["age"] = 26            # update a value
del person["age"]             # remove a key
"name" in person              # check a key exists -> True
person.keys(), person.values(), person.items()
```

### Strings
Text in single or double quotes; triple quotes for multi-line text. Strings are
immutable, so every "change" returns a new string. They can be sliced just like
lists, which is how `reverse_string` works with `text[::-1]`.

```python
name = "Python"
name[::-1]                 # 'nohtyP'
f"Hello, {name}!"          # f-string (the modern way)
"Hello, {}!".format(name)  # .format()
"Hello, %s!" % name        # old % style
```

### Flow control
`if` / `elif` / `else` for decisions, `for` to walk through a collection, and
`while` to repeat while a condition holds. There is no `switch` in Python — use
`if`. Blocks are marked by **indentation**, not braces.

```python
for number in range(10):     # 0 to 9
    if number in (3, 4, 7):
        break                # leave the loop now
    else:
        continue             # jump to the next round
```

`break` stops a loop early, `continue` skips to the next iteration, and a loop
can even have its own `else`, which runs only if the loop never hit `break`.

### Functions
Reusable blocks defined with `def`. Arguments can have default values, and a
function can return several values at once as a tuple. A `lambda` is a small
one-line function.

```python
def power(base, exponent=2):     # exponent is optional
    return base ** exponent

power(5)        # 25
power(5, 3)     # 125
square = lambda x: x * x         # same as a small def
```

Note on arguments: mutable objects (lists, dictionaries) *can* be changed inside
a function and the caller sees the change; immutable ones (numbers, strings,
tuples) cannot.

### Classes
A class is a blueprint that bundles data (attributes) with behaviour (methods).
`__init__` runs when the object is created, and `self` refers to the object
itself. A class can **inherit** from another to reuse its code.

```python
class Person:
    def __init__(self, name, age):
        self.name = name        # attribute
        self.age = age

    def introduce(self):        # method
        print(f"I am {self.name}")

class Student(Person):          # inherits from Person
    def __init__(self, name, age, student_id):
        super().__init__(name, age)   # let Person do its part
        self.student_id = student_id
```

### File I/O
`open(filename, mode)` opens a file — `"r"` read, `"w"` write, `"a"` append, and
add `"b"` for binary. Using `with` is preferred because it closes the file
automatically, even if an error happens.

```python
with open("text.txt", "w") as f:
    f.write("This is a sample string")

with open("text.txt") as f:
    print(f.read())
```

Plain files store **text**. To store real Python objects, use `pickle` — that is
what Task 3 is about (explained in its section below).

## What each task does

### Task 1 — Lists, dictionaries and tuples
Creating and changing the three basic Python containers.

- **Lists:** `append`, `insert`, `remove`, `reverse`, `sort`, slicing and `len`.
- **Dictionaries:** adding a key, updating a value, deleting a key with `del`,
  checking a key with `in`, and printing the keys and values.
- **Tuples:** indexing, concatenation and unpacking. Changing an element is
  wrapped in `try` / `except TypeError`, so the notebook *prints* the error
  message instead of stopping — this is the proof that tuples are immutable.
- **Combining them:** a dictionary whose values are lists, and a list of
  `(fruit, colour)` tuples that is searched for one item.

### Task 2 — Functions and classes
- **Functions:** `greet`, `fibonacci`, `is_prime`, `factorial` and
  `reverse_string`. `fibonacci` is written as a loop (fast even for large `n`),
  `factorial` is recursive, and `is_prime` only tests divisors up to `√n`.
- **Classes:** `Person` (name, age, email) → `Student` inherits from it and adds
  `student_id` → `Course` keeps a list of `Student` objects.
  `Student` calls `super().__init__()` so the parent does the shared setup.
- **Demo:** a course "Introduction to Python" with three students added, listed,
  introduced, and one of them shown studying.

### Task 3 — File I/O with `pickle`
**Pickling** turns a Python object into bytes that can be saved in a file;
**unpickling** rebuilds the original object from those bytes. It is used because
it keeps the real Python types — a list of objects comes back as a list of
objects, not as text that has to be parsed again.

- A list of dictionaries is saved to `students.pkl` and read back by
  `load_students`.
- A custom `Student` class (name, age, grade) is saved to `students_class.pkl`
  and read back by `load_student_objects`.
- `add_student` and `remove_student` load the file, change the list, and write
  it back. The demo adds *Diana* and removes *Bob*.

Two things to remember about pickle:

1. Always open the file in **binary** mode — `"wb"` to write, `"rb"` to read.
2. Only load pickle files from a **trusted source**, because unpickling can run
   arbitrary code.

## Note on the two `Student` classes

The manual asks for a class named `Student` in both Task 2 and Task 3, but with
different attributes. Since both live in one notebook, the Task 3 definition
replaces the Task 2 one. This is intentional and is marked with a comment in the
code — just run the cells in order and each task uses the class it expects.
