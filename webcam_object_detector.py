from IPython import display

import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#%matplotlib inline

import cv2 as cv2

#keras.summary

def draw_box(image, label,score, x1, y1, x2, y2):
    h = image.shape[0]
    w = image.shape[1]     
    x1 = int(x1*w)
    x2 = int(x2*w)
    y1 = int(y1*h)
    y2 = int(y2*h)
    label = label + " " + score.astype('str')
    cv2.rectangle(image, (x1, y2), (x2, y1), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    return image

def draw_boxes(result, image, top_predictions=100):
    w = image.shape[0]
    h = image.shape[1]

    for i in range(top_predictions):
        score = result["detection_scores"][i]
        label = result["detection_class_entities"][i].decode("ascii")
        if score > 0.3:
            y1,x1,y2,x2 = tuple(result["detection_boxes"][i])
            #print(tuple(result["detection_boxes"][i]))
            image = draw_box(image, label, score, x1, y1, x2, y2)

    #x and  y the wrong way round... x is thinner
    #imgshow = plt.imshow(image)
    return image

def formatImage(image):
    #imageResized = cv2.resize(image, (height,width), interpolation=cv2.INTER_LINEAR)
    mangledImage = np.reshape(image, [1,image.shape[0],image.shape[1],3])
    img_tensor = tf.convert_to_tensor(mangledImage, dtype=tf.float32)
    return img_tensor



def detect(image_tensor, detector):
    result = detector(image_tensor)
    result = {key:value.numpy() for key,value in result.items()}
    return result


mobile_net = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1" #224,224

res_net = "https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1"

#inception = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(mobile_net).signatures['default']
cap = cv2.VideoCapture(0)
print("made it here")

while cap.isOpened():
    ret,frame = cap.read()
    if ret:
        frame = frame.astype(float) / 255
        #img = cv2.imread(frame)
        imgFormat = formatImage(frame)
        result = detect(imgFormat, detector)
        detected_webcam = draw_boxes(result, frame)


        cv2.namedWindow("object detection", cv2.WINDOW_NORMAL) 
        cv2.imshow("object detection", detected_webcam)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
print("made it here3")
