import cv2
import numpy as np

model = {
    'coco': 'coco.names',
    'cfg': 'cfg/yolov3-tiny.cfg',
    'weight': 'weights/yolov3-tiny.weights',
    'conf_threshold': 0.5,
    'nms_threshold': 0.3
}

CLASSES = []
with open(model['coco'], 'r') as f:
    CLASSES = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromDarknet(model['cfg'], model["weight"])
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def draw_box(img, indices, boxes, class_ids, confidences):
    # returned data format [[0]]
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        target = CLASSES[class_ids[i]].upper()
        color = COLORS[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, f"{target} {int(confidences[i] * 100)}%", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def find_objects(outputs, img):
    """
                |  0       1      2       3           4          5       6       ...     79
    box number  | cx      cy      w       h       confidence  person      bicycle       toothbrush
    0           | 0.51    0.64    0.58   0.36     0.93        0           0               0
    ...         |
    299         |
    """

    height, width, channel = img.shape
    boxes = []
    class_ids = []
    confidences = []

    for output in outputs:
        for detection in output:
            # remove not class data
            scores = detection[5:]
            # get max value index
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])
            if confidence > model['conf_threshold']:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(detection[0] * width - w /2), int(detection[1] * height - h / 2)
                boxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(confidence)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, model['conf_threshold'], model['nms_threshold'])
    draw_box(img, indexes, boxes, class_ids, confidences)


def get_layers(nt):
    layer_names = nt.getLayerNames()
    out_layers = [layer_names[i[0] - 1] for i in nt.getUnconnectedOutLayers()]
    return out_layers


cap = cv2.VideoCapture(0)
cap.set(3, 1920) # width
cap.set(4, 1080) # height

while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], crop=False)
    net.setInput(blob)

    outputs = net.forward(get_layers(net))
    find_objects(outputs, img)

    cv2.imshow('frame', img)
    k = cv2.waitKey(1) & 0xff
    # press 'ESC' to quit
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()