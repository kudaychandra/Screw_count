import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread(r'C:\Users\Uday Chandra\OneDrive\Desktop\Screw_count\screw_test\img1.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Check the path!")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print(gray)
# cv2.imshow("Gray Image", gray)
# cv2.waitKey(0)

# Gaussian blur to reduce noise
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Blurred Image", blur)
# cv2.waitKey(0)

# Otsu’s thresholding (binary inverse) to separate dark objects on light background
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological opening to remove small artifacts
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Find external contours (each corresponds to one item)
contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
count = len(contours)

# Plot original image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Overlay contours in green
# Note: plt.contour expects a 2D mask; we draw all contours at once
plt.contour(opening, linewidths=1)

# Title with the count
plt.title(f'Count of Items: {count}', fontsize=16)
plt.show()
