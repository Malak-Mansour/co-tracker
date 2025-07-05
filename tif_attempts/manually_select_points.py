import cv2
import numpy as np

# List to store clicked points
clicked_points = []

# Colors
DOT_COLOR = (255, 0, 0)    # Blue for previous points
LATEST_DOT_COLOR = (0, 0, 255)  # Red for latest point

# Mouse callback function
def click_event(event, x, y, flags, param):
    global clicked_points, img, img_with_dots
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))

        # Print the latest clicked point in torch format
        print(f"[0., {x:.1f}, {y:.1f}],")

        # Draw all previous points as blue
        img_with_dots = img.copy()
        for pt in clicked_points[:-1]:
            cv2.circle(img_with_dots, pt, 5, DOT_COLOR, -1)
        # Draw latest point as red
        cv2.circle(img_with_dots, clicked_points[-1], 5, LATEST_DOT_COLOR, -1)
        cv2.imshow("Image", img_with_dots)

# Load image
image_path = 't01.jpg'  # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img_with_dots = img.copy()

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)
cv2.imshow("Image", img_with_dots)

cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("queries = torch.tensor([")
for pt in clicked_points:
    print(f"    [0., {pt[0]:.1f}, {pt[1]:.1f}],")
print("])")