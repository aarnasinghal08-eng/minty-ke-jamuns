import os
import numpy as np

PATCH_SIZE_LR = 32
UPSCALE_FACTOR = 2

PATCH_SIZE_HR = PATCH_SIZE_LR * UPSCALE_FACTOR

STRIDE = 16

SR_INPUT_DIR = "dataset/patches/sr_input"
SR_TARGET_DIR = "dataset/patches/sr_target"
COLOR_INPUT_DIR = "dataset/patches/color_input"
COLOR_TARGET_DIR = "dataset/patches/color_target"

def save_patch(folder, patch, patch_id):
    filename = f"patch_{patch_id:06d}.npy"
    np.save(os.path.join(folder, filename), patch)

def extract_patch_pairs(tir200,tir100,rgb100,patch_id):
    height = tir200.shape[0]
    width = tir200.shape[1]

    for y in range(0,height - PATCH_SIZE_LR + 1,STRIDE):
        for x in range(0,width - PATCH_SIZE_LR + 1,STRIDE):
            lr_patch = tir200[y:y + PATCH_SIZE_LR,x:x + PATCH_SIZE_LR]
            hr_x = x * UPSCALE_FACTOR
            hr_y = y * UPSCALE_FACTOR

            tir100_patch = tir100[hr_y:hr_y + PATCH_SIZE_HR,hr_x:hr_x + PATCH_SIZE_HR]
            rgb100_patch = rgb100[hr_y:hr_y + PATCH_SIZE_HR,hr_x:hr_x + PATCH_SIZE_HR]

            if (lr_patch.shape[:2] != (PATCH_SIZE_LR, PATCH_SIZE_LR) or tir100_patch.shape[:2] != (PATCH_SIZE_HR, PATCH_SIZE_HR) or rgb100_patch.shape[:2] != (PATCH_SIZE_HR, PATCH_SIZE_HR)):
                continue

            save_patch(SR_INPUT_DIR, lr_patch, patch_id)
            save_patch(SR_TARGET_DIR, tir100_patch, patch_id)
            save_patch(COLOR_INPUT_DIR, tir100_patch, patch_id)
            save_patch(COLOR_TARGET_DIR, rgb100_patch, patch_id)

            patch_id += 1

    return patch_id

def main():
    patch_id = 1
    TIR200_DIR = "dataset/tir200"
    TIR100_DIR = "dataset/tir100"
    RGB100_DIR = "dataset/rgb100"

    files = sorted(os.listdir(TIR200_DIR))

    for file in files:
        tir200 = np.load(os.path.join(TIR200_DIR, file))
        tir100 = np.load(os.path.join(TIR100_DIR, file))
        rgb_file = file.replace("_tir", "_rgb")
        rgb100 = np.load(os.path.join(RGB100_DIR, rgb_file))

        patch_id = extract_patch_pairs(tir200,tir100,rgb100,patch_id)

        print(f"Processing {file}...")

    print(f"Total patches created: {patch_id - 1}")

if __name__ == "__main__" :
    main()