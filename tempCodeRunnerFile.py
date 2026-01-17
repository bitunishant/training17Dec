import cv2
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# FUNCTION 1: Extract Handwriting Mask (No Rules)
# --------------------------------------------------
def extract_handwriting_mask(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image path")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binary image (text = white)
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 10
    )

    # Detect horizontal rules
    h, w = gray.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, 1)

    # Refine mask
    line_mask = cv2.dilate(horizontal_lines, np.ones((3,3), np.uint8), 1)

    # Remove rules from text
    handwriting_mask = cv2.subtract(binary, line_mask)

    return handwriting_mask, img


# --------------------------------------------------
# FUNCTION 2: Convert to B/W and Remove Rules (White)
# --------------------------------------------------
def bw_rule_removal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to strict B/W
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect horizontal rules again
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    horizontal = cv2.dilate(horizontal, kernel2, 1)

    # Replace rule pixels with WHITE
    bw_clean = bw.copy()
    bw_clean[horizontal > 0] = 255

    return bw_clean


# --------------------------------------------------
# FUNCTION 3: STRICT INTERSECTION (AND)
# --------------------------------------------------
def strict_intersection(mask1, bw_clean):
    # Convert cleaned B/W to mask2 (text = white)
    mask2 = cv2.adaptiveThreshold(
        bw_clean, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15, 5
    )

    # Strict AND
    final_mask = cv2.bitwise_and(mask1, mask2)

    # Final output: black text on white
    final = np.ones_like(final_mask) * 255
    final[final_mask > 0] = 0

    return final


# --------------------------------------------------
# RUN FULL PIPELINE
# --------------------------------------------------
image_path = "kwrite7pass.jpeg"

# Step 1: Extract handwriting mask
mask1, original = extract_handwriting_mask(image_path)

# Step 2: Convert to B/W and remove rules (white)
bw_clean = bw_rule_removal(original)

# Step 3: Strict intersection restore
final = strict_intersection(mask1, bw_clean)

# Save result
cv2.imwrite("final_handwriting_bw.png", final)

# Display
plt.figure(figsize=(15,5))

plt.subplot(1,4,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,4,2)
plt.title("Handwriting Mask (Function 1)")
plt.imshow(mask1, cmap="gray")
plt.axis("off")

plt.subplot(1,4,3)
plt.title("B/W + Rules → White")
plt.imshow(bw_clean, cmap="gray")
plt.axis("off")

plt.subplot(1,4,4)
plt.title("FINAL (Strict AND)")
plt.imshow(final, cmap="gray")
plt.axis("off")

plt.show()
    