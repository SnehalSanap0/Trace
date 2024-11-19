import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random

# Load the model
model = hub.load("https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1").signatures["default"]
colorcodes = {}

def drawbox(image, ymin, xmin, ymax, xmax, namewithscore, color):
    im_height, im_width, _ = image.shape
    left, top, right, bottom = int(xmin * im_width), int(ymin * im_height), int(xmax * im_width), int(ymax * im_height)

    # Draw a thin rectangle for the bounding box
    cv2.rectangle(image, (left, top), (right, bottom), color=color, thickness=2)  # Thin outline

    # Add a subtle shadow effect for text
    shadow_color = (0, 0, 0)  # Black shadow
    cv2.putText(image, namewithscore, (left + 1, top - 1),  # Slight offset for shadow
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=shadow_color)  # Shadow color

    # Draw the actual text
    cv2.putText(image, namewithscore, (left, top),  # No offset for actual text
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                thickness=1,
                color=(255, 255, 255))  # White text

def draw(image, boxes, classnames, scores):
    boxesidx = tf.image.non_max_suppression(boxes, scores, max_output_size=20, score_threshold=0.1)
    boxesidx = boxesidx.numpy()  # Convert to numpy array
    for i in boxesidx:
        ymin, xmin, ymax, xmax = tuple(boxes[i])
        classname = classnames[i].decode("utf-8")  # Use utf-8 to decode
        if classname in colorcodes.keys():
            color = colorcodes[classname]
        else:
            # Use a lighter color for the bounding box
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
            colorcodes[classname] = color
        drawbox(image, ymin, xmin, ymax, xmax, f"{classname}: {scores[i]:.2f}", color)

def resize_with_aspect_ratio(image, max_width, max_height):
    h, w, _ = image.shape
    aspect_ratio = w / h
    if w > max_width or h > max_height:
        if aspect_ratio > 1:  # Wider than tall
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:  # Taller than wide
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

# Load and process the image
image_path = "image3.jpg"  # Replace with your image path
image = cv2.imread(image_path)
if image is None:
    print("Error: Unable to load the image. Check the path.")
else:
    # Resize the image to fit within screen dimensions
    screen_width, screen_height = 1200, 800  # Adjust these dimensions as needed
    image = resize_with_aspect_ratio(image, screen_width, screen_height)

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]

    # Perform detection
    result = model(img_tensor)
    boxes = result["detection_boxes"].numpy()
    classnames = result["detection_class_entities"].numpy()
    scores = result["detection_scores"].numpy()

    draw(image, boxes, classnames, scores)

    # Display the resized image with detections
    cv2.imshow("detection", image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()
