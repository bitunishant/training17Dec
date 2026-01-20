import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Shadow removal
def remove_shadows(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    background = cv2.medianBlur(cv2.dilate(gray, kernel), 21)

    diff = 255 - cv2.absdiff(gray, background)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


# 2. Detect horizontal rule pixels
def detect_horizontal_rules(gray):
    binary = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 5)

    h, w = gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return horizontal, np.sum(horizontal > 0)


# 3. Extract handwriting mask
def extract_handwriting_mask(processed_gray, remove_rules=True):
    binary = cv2.adaptiveThreshold(processed_gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    if not remove_rules:
        return binary, np.zeros_like(binary)

    h, w = processed_gray.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 15, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    line_mask = cv2.dilate(horizontal, np.ones((3, 3), np.uint8))
    handwriting = cv2.subtract(binary, line_mask)

    return handwriting, line_mask


# 4. Clean black & white image
def bw_rule_removal(processed_gray, remove_rules=True):
    _, bw = cv2.threshold(processed_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if not remove_rules:
        return bw

    thresh = cv2.adaptiveThreshold(processed_gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, 2)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    horizontal = cv2.dilate(horizontal, kernel2)

    bw[horizontal > 0] = 255
    return bw


# 5. Strict text intersection
def strict_intersection(mask1, bw_clean):
    mask2 = cv2.adaptiveThreshold(bw_clean, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5)

    final_mask = cv2.bitwise_and(mask1, mask2)

    final = np.ones_like(final_mask) * 255
    final[final_mask > 0] = 0

    return final, final_mask


# 6. Restore missing strokes
def restore_lost_pixels(final_mask, original_mask):
    h, w = final_mask.shape

    proximity = cv2.dilate(final_mask, np.ones((5, 5), np.uint8))
    lost = cv2.subtract(original_mask, final_mask)

    long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    long_lines = cv2.morphologyEx(lost, cv2.MORPH_OPEN, long_kernel)

    safe = cv2.subtract(lost, long_lines)
    restore = cv2.bitwise_and(safe, proximity)

    return cv2.bitwise_or(final_mask, restore)


# ---------------- MAIN ----------------
image_path = "photo1.jpg"
original = cv2.imread(image_path)

shadow_free_gray = remove_shadows(original)

rule_mask, rule_pixel_count = detect_horizontal_rules(shadow_free_gray)

h, w = shadow_free_gray.shape
rule_ratio = rule_pixel_count / (h * w)

REMOVE_RULES = rule_ratio > 0.01

mask1, _ = extract_handwriting_mask(shadow_free_gray, REMOVE_RULES)
bw_clean = bw_rule_removal(shadow_free_gray, REMOVE_RULES)

_, final_mask = strict_intersection(mask1, bw_clean)

for _ in range(2):
    final_mask = restore_lost_pixels(final_mask, mask1)

long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
line_cleanup = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, long_kernel)
final_mask = cv2.subtract(final_mask, line_cleanup)

final_image = np.ones_like(final_mask) * 255
final_image[final_mask > 0] = 0


# Display
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Shadow Free")
plt.imshow(shadow_free_gray, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Final Output")
plt.imshow(final_image, cmap="gray")
plt.axis("off")

plt.show()

cv2.imwrite("final_restored.png", final_image)
