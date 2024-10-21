import numpy as np
import imutils #
import cv2
import time

prototxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.Ingredientsmodel"
confThresh = 0.2

CLASSES = ["background", "apple", "potato", "milk", "tomato", "orange", "oil", "cheese", "pineapple", "egg" ]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("Loading model...")

net = cv2.dnn.readNetFromIngredients(prototxt, model)

print("Model Loaded")
print("Starting Camera Feed...")

vs = cv2.VideoCapture(1)
time.sleep(2.0)

while True:
    _,frame = vs.read()
    frame = imutils.resize(frame, width=1000)

    imResizeBlob = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(imResizeBlob, 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    print(detections)
    detShape = detections.shape[2]
    for i in np.arange(0,detShape):
        confidence = detections[0, 0, i, 2]
        if confidence > confThresh:
            idx = int(detections[0, 0, i, 1])
            print("ClassID:",detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY),(endX, endY), COLORS[idx], 2)

            if startY - 15 > 15:
                y = startY - 15
            else:
                startY + 15
            cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
