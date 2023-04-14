import cv2
from PIL import Image
import numpy as np
import ctypes

trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get screen size
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

def resize(image):

    image = Image.open(image)

    # Calculate the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)

    # Calculate the maximum width and height of the resized image
    max_width = screen_width
    max_height = screen_height

    # Check if the aspect ratio of the image is greater than the aspect ratio of the screen
    if aspect_ratio > float(screen_width) / float(screen_height):
        max_height = int(screen_width / aspect_ratio)
    else:
        max_width = int(screen_height * aspect_ratio)

    # Resize the image
    image = image.resize((max_width, max_height), Image.ANTIALIAS)

    # Convert the image to a NumPy array
    image_np = np.array(image)

    return image_np

# Load the original image
original_image = cv2.imread("image5.jpg")

# Resize the image
resized_image = resize("image5.jpg")

# Convert image to grayscale
grayScaleImg = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Detect faces, detected objects are returned as a list of rectangles   
faceCoordinates = trainedFaceData.detectMultiScale(grayScaleImg)
try: 
    for (x,y,w,h) in faceCoordinates:
        cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw rectangles around detected faces
except IndexError:
    # Handle the case where no faces are detected
    pass


# Convert color space from RGB to BGR
resized_image_bgr = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

# Display the resized image
cv2.imshow("Resized Image", resized_image_bgr)

# Save the image with the rectangle
cv2.imwrite("image5WithRect.jpg", resized_image_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()
