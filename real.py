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

# video = cv2.VideoCapture("video.mp4")
video = cv2.VideoCapture("classroom.mp4")

# Get video dimensions
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a window with the same size as the video
cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("detection", frame_width, frame_height)

# Get screen dimensions
screen_width = screeninfo.get_monitors()[0].width
screen_height = screeninfo.get_monitors()[0].height

# Calculate the center position
center_x = (screen_width - frame_width) // 2
center_y = (screen_height - frame_height) // 2

# Move the window to the center of the screen
cv2.moveWindow("detection", center_x, center_y)

while True:
    ret, img = video.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = tf.image.convert_image_dtype(img_rgb, tf.float32)[tf.newaxis, ...]

    result = model(img_tensor)
    boxes = result["detection_boxes"].numpy()
    classnames = result["detection_class_entities"].numpy()
    scores = result["detection_scores"].numpy()

    draw(img, boxes, classnames, scores)

    cv2.imshow("detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()