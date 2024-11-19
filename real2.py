import tensorflow as tf
import cv2
import numpy as np
import tensorflow_hub as hub
import random
import screeninfo

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

# Open the webcam (0 is the default camera)
video = cv2.VideoCapture(0)

# Optional: Set video frame width and height
video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, img = video.read()
    if not ret:
        print("Failed to capture frame.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]

    result = model(img_tensor)
    boxes = result["detection_boxes"].numpy()
    classnames = result["detection_class_entities"].numpy()
    scores = result["detection_scores"].numpy()

    draw(img_rgb, boxes, classnames, scores)

    cv2.imshow("Webcam Object Detection", img_rgb)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()