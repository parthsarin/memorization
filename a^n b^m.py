import numpy as np
import matplotlib.pyplot as plt

theta = 2 * np.pi * (7/8)
Wh = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
Wx = np.array([
    [0, 0],
    [0, 0]
])
h = np.array([0, -1])
hlist = [h]

hyperplane = -0.5

for i in range(50):
    y = np.array([0, 1]) if h[0] < hyperplane else np.array([1, 0])
    ylabel = "a" if y[0] == 1 else "b"

    # print(f"[step {i}] hidden = {h}, y_label = {ylabel}")
    print(ylabel, end=", ")

    h = Wh @ h + Wx @ y
    h = h.round(2)
    hlist.append(h)
print()

# add a vertical line on the y-axis
plt.axvline(x=hyperplane, color="black")

for a, b in zip(hlist, hlist[1:]):
    hlist = np.array(hlist)
    plt.plot(hlist[:, 0], hlist[:, 1], "o-")

# fix the proportions of the axes
plt.gca().set_aspect("equal", adjustable="box")

plt.show()

# number of unique hidden states
print("number of unique hidden states:", len(set([tuple(x) for x in hlist])))
