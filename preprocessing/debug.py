import numpy as np
import matplotlib.pyplot as plt

loaded_rgb = np.load("dataset/rgb/rgb-001.npy")

rgb_normalized = (
loaded_rgb - loaded_rgb.min()
) / (
loaded_rgb.max() - loaded_rgb.min()
)

plt.imshow(rgb_normalized)
plt.show()