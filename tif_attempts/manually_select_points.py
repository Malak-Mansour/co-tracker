import cv2

# List to store clicked points
clicked_points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: ({x}, {y})")
        clicked_points.append((x, y))

        # Optional: show a red circle where clicked
        img_copy = param.copy()
        cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", img_copy)

# Load image
image_path = 't01.jpg'  # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Show image and set mouse callback
# cv2.imshow("Image", img)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 1280, 720)  # Or another size that fits your screen
cv2.imshow("Image", img)

cv2.setMouseCallback("Image", click_event, img)

# Wait until any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save coordinates
# print("All clicked points:", clicked_points)
print("queries = torch.tensor([")
for pt in clicked_points:
    # print(f"    [0, {pt[0]}, {pt[1]}],")
    print(f"    [0., {pt[0]:.1f}, {pt[1]:.1f}],")
print("])")
