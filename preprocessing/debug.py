# import numpy as np
# import matplotlib.pyplot as plt

# loaded_rgb = np.load("dataset/rgb/rgb-001.npy")

# rgb_normalized = (
# loaded_rgb - loaded_rgb.min()
# ) / (
# loaded_rgb.max() - loaded_rgb.min()
# )
# print(rgb_normalized.shape)
# print(564//128)
# print(500//128)
# plt.imshow(rgb_normalized)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

lr = np.load("dataset/patches/sr_input/patch_000001.npy")
hr = np.load("dataset/patches/sr_target/patch_000001.npy")
rgb = np.load("dataset/patches/color_target/patch_000001.npy")

rgb_normalized = (rgb - rgb.min()) / (rgb.max() - rgb.min())

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(lr, cmap="hot")
plt.title("TIR 200m")

plt.subplot(1,3,2)
plt.imshow(hr, cmap="hot")
plt.title("TIR 100m")

plt.subplot(1,3,3)
plt.imshow(rgb_normalized)
plt.title("RGB 100m")

plt.show()

print("LR:", lr.shape)
print("HR:", hr.shape)
print("RGB:", rgb.shape)