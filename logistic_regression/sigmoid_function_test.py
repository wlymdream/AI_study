import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid_func(x):
    result = []
    for x_data in x:
        result.append(1.0 / (1.0 + math.exp(-x_data)))
    return result

x = np.arange(-20, 20, 1)
y = sigmoid_func(x)

plt.plot(x, y)
plt.show()