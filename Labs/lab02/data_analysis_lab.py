"""AIC 380 - Artificial Neural Networks
Lab 02, Lab Task 3: analysing a small student-grade dataset with NumPy.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1. Load and Display the Dataset
# ---------------------------------------------------------------------------
# Each sublist is one student: [ID, math, science, english]
data = [
    [1, 85, 78, 92],
    [2, 76, 85, 88],
    [3, 90, 92, 95],
    [4, 68, 72, 70],
    [5, 88, 89, 94],
]

# Convert the dataset into a NumPy array
grades = np.array(data)

print("Dataset as a NumPy array (columns: ID, Math, Science, English):")
print(grades)
print("Shape:", grades.shape, "| dtype:", grades.dtype)

# ---------------------------------------------------------------------------
# 2. Extract and Analyze Specific Data Columns
# ---------------------------------------------------------------------------
# Column 0 holds the student IDs, so it is kept separate and never averaged.
student_ids = grades[:, 0]
math = grades[:, 1]
science = grades[:, 2]
english = grades[:, 3]

print("\nMath grades   :", math)
print("Science grades:", science)
print("English grades:", english)

# Calculate and display the average grade for each subject
subject_names = ["Math", "Science", "English"]
subject_averages = [math.mean(), science.mean(), english.mean()]

print("\nAverage grade per subject")
for name, average in zip(subject_names, subject_averages):
    print(f"  {name:<8}: {average:.2f}")

# ---------------------------------------------------------------------------
# 3. Perform Basic Statistical Analysis
# ---------------------------------------------------------------------------
# Columns 1 to 3 are the grades. axis=1 averages across each row (per student).
student_averages = grades[:, 1:].mean(axis=1)

print("\nOverall average per student")
for student_id, average in zip(student_ids, student_averages):
    print(f"  Student {student_id}: {average:.2f}")

# Find the student with the highest and the lowest average grade
best_index = np.argmax(student_averages)
worst_index = np.argmin(student_averages)

print(f"\nHighest average: Student {student_ids[best_index]} "
      f"with {student_averages[best_index]:.2f}")
print(f"Lowest  average: Student {student_ids[worst_index]} "
      f"with {student_averages[worst_index]:.2f}")

# ---------------------------------------------------------------------------
# 4. Manipulate and Transform the Data
# ---------------------------------------------------------------------------
# A copy is used so the original array stays intact for the comparison below.
grades_with_bonus = grades.copy()

# Add 5 bonus points to each student's math grade
grades_with_bonus[:, 1] = grades_with_bonus[:, 1] + 5

print("\nMath grades before the bonus:", grades[:, 1])
print("Math grades after  the bonus:", grades_with_bonus[:, 1])

# Calculate the new average grade for each student after adding the bonus
new_student_averages = grades_with_bonus[:, 1:].mean(axis=1)

print("\nAverage per student before and after the bonus")
for student_id, before, after in zip(student_ids, student_averages, new_student_averages):
    print(f"  Student {student_id}: {before:.2f} -> {after:.2f}  (+{after - before:.2f})")

# ---------------------------------------------------------------------------
# 5. Visualise the Subject Averages
# ---------------------------------------------------------------------------
figure, axes = plt.subplots(figsize=(6, 4))

bars = axes.bar(subject_names, subject_averages, width=0.55, color="#2a78d6")

# Label each bar directly, so the exact value is readable without the gridlines
for bar, average in zip(bars, subject_averages):
    axes.text(bar.get_x() + bar.get_width() / 2, average + 1.5,
              f"{average:.1f}", ha="center", va="bottom",
              fontsize=10, color="#0b0b0b")

# The bars start at 0 so that their heights stay proportional to the values
axes.set_ylim(0, 100)
axes.set_ylabel("Average grade", color="#52514e")
axes.set_title("Average grade per subject", color="#0b0b0b", fontsize=12, pad=12)

# Keep the axes recessive: no box around the plot, a faint grid behind the bars
axes.spines["top"].set_visible(False)
axes.spines["right"].set_visible(False)
axes.spines["left"].set_color("#d6d5d0")
axes.spines["bottom"].set_color("#d6d5d0")
axes.tick_params(colors="#52514e")
axes.yaxis.grid(True, color="#e6e5e0", linewidth=0.8)
axes.set_axisbelow(True)

plt.tight_layout()
plt.show()
