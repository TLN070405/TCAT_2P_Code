import matplotlib.pyplot as plt

"""# Define only the lines shown in the latest legend for Ew
legend_entries = [
    ("TCAT  t=1000 sec", "magenta", "--"),
    ("TCAT  t=10000 sec", "goldenrod", "--"),
    ("TCAT  t=100000 sec", "dodgerblue", "--"),
    ("TCAT  t=150000 sec", "green", "--"),
    ("RE  t=0.1 sec", "orange", "-"),
    ("RE  t=1 sec", "red", "-"),
    ("RE  t=1000 sec", "brown", "-"),
    ("RE  t=1500 sec", "red", "-"),
]"""
"""
# Define only the lines shown in the latest legend for Pw
legend_entries = [
    ("TCAT  t=0.1 sec", "dodgerblue", "--"),
    ("TCAT  t=1.0 sec", "green", "--"),
    ("TCAT  t=5 sec", "red", "--"),
    ("RE  t=0.1 sec", "orange", "-"),
    ("RE  t=1.0 sec", "red", "-"),
    ("RE  t=100 sec", "brown", "-"),
    ("RE  t=200 sec", "red", "-"),
]
"""
# Define only the lines shown in the latest legend for Jwn
legend_entries = [
    ("t=0.0001 sec", "orange", "-"),
    ("t=0.001 sec", "green", "-"),
    ("t=0.01 sec", "red", "-"),
    ("t=0.1 sec", "purple", "-"),
    ("t=1.0 sec", "brown", "-"),
    ("t=2.0 sec", "green", "-"),
]



# Plot only the legend
plt.figure(figsize=(8, 10))
for label, color, style in legend_entries:
    plt.plot([], [], linestyle=style, color=color, linewidth=3, label=label)

plt.legend(fontsize=30)
plt.axis("off")
plt.tight_layout()
plt.show()
