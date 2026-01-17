import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function loads the image and tries to isolate only the handwriting
# by removing the horizontal notebook lines
def extract_handwriting_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary where text becomes white
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Create a wide horizontal kernel to detect notebook lines
    h, w = gray.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, 1)

    # Thicken the detected lines so we remove them properly
    line_mask = cv2.dilate(horizontal_lines, np.ones((3,3), np.uint8), 1)

    # Subtract the lines from the text to keep only handwriting
    handwriting_mask = cv2.subtract(binary, line_mask)

    return handwriting_mask, img, line_mask 

# This function converts the image to clean black & white
# and removes the horizontal rules again for safety
def bw_rule_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding for clean black-white conversion
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect horizontal lines again using adaptive threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    horizontal_dilated = cv2.dilate(horizontal, kernel2, 1)

    # Replace detected rule pixels with white
    bw_clean = bw.copy()
    bw_clean[horizontal_dilated > 0] = 255

    return bw_clean

# This function keeps only the pixels that both methods agree are text
def strict_intersection(mask1, bw_clean):
    mask2 = cv2.adaptiveThreshold(
        bw_clean, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    # AND operation to keep only common text areas
    final_mask = cv2.bitwise_and(mask1, mask2)

    # Convert result to black text on white background
    final = np.ones_like(final_mask) * 255
    final[final_mask > 0] = 0

    return final, final_mask

# Path to the input image
image_path = "kwrite13pass.jpeg"

# Step 1: Get handwriting-only mask
mask1, original, rule_zone = extract_handwriting_mask(image_path)

# Step 2: Clean the image to black & white without rules
bw_clean = bw_rule_removal(original)

# Step 3: Keep only the reliable text pixels
final_img, final_mask = strict_intersection(mask1, bw_clean)

# Some letters break where the lines were removed,
# so we use vertical dilation to reconnect them
bridge_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
bridged_final = cv2.dilate(final_mask, bridge_k, iterations=1)

# Restore only the pixels that originally belonged to handwriting
recovered_pixels = cv2.bitwise_and(bridged_final, mask1)
final_mask_restored = cv2.bitwise_or(final_mask, recovered_pixels)

# Final clean result: black handwriting on white background
restored_final_image = np.ones_like(final_mask_restored) * 255
restored_final_image[final_mask_restored > 0] = 0

# Show original, broken, and restored results side by side
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Before Restoration (Broken)")
plt.imshow(final_img, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("After Restoration (Connected)")
plt.imshow(restored_final_image, cmap="gray")
plt.axis("off")

plt.show()

# Save the final output image
cv2.imwrite("final_restored.png", restored_final_image)
