import cv2
import argparse

# Parse input arguments
parser = argparse.ArgumentParser(description="Canny Edge Detection with Trackbars")
parser.add_argument('--filename', required=True, help='Path to input image')
args = parser.parse_args()

# Read image
img = cv2.imread(args.filename)
if img is None:
    print(f"Error: Unable to load image '{args.filename}'")
    exit(1)

# Convert to gray
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Window for edge image + trackbars
cv2.namedWindow("Edge Image")

# Callback function (required but unused)
def nothing(x):
    pass

# Create trackbars in the same window
cv2.createTrackbar("Min Threshold", "Edge Image", 0, 255, nothing)
cv2.createTrackbar("Max Threshold", "Edge Image", 0, 255, nothing)
cv2.createTrackbar("Blur Kernel", "Edge Image", 1, 31, nothing)  # Max 31 for reasonable kernel size

# Set initial values
cv2.setTrackbarPos("Min Threshold", "Edge Image", 50)
cv2.setTrackbarPos("Max Threshold", "Edge Image", 150)
cv2.setTrackbarPos("Blur Kernel", "Edge Image", 7)

while True:
    # Get current trackbar positions
    tmin = cv2.getTrackbarPos("Min Threshold", "Edge Image")
    tmax = cv2.getTrackbarPos("Max Threshold", "Edge Image")
    ksize = cv2.getTrackbarPos("Blur Kernel", "Edge Image")
    
    # Kernel size must be odd and >= 1
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 1:
        ksize = 1

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(gray_img, (ksize, ksize), 0)

    # Apply Canny
    edged_img = cv2.Canny(blurred_img, tmin, tmax)

    # Show edge image with trackbars
    cv2.imshow("Edge Image", edged_img)

    # Exit if any key pressed
    if cv2.waitKey(1) != -1:
        break

cv2.destroyAllWindows()
