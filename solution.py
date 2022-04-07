import time
#from detecto import core, utils, visualize
import numpy as np
import tensorflow as tf
from detecto import core, utils
import torch
import torchvision.models as models
import cv2

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
coords = []
thresh = 0.2
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Device: {device}")
num_classes = 2

model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained_backbone=False, num_classes=num_classes)
model.load_state_dict(torch.load('model_weights.pth', map_location=device))
model.eval()
model.to(device)

indexer = 0

COLORS = np.random.uniform(0, 255, size=(2, 3))

def GetLocation(move_type, env, current_frame):
    global indexer
    #time.sleep(1) #artificial one second processing time
    
    #Use relative coordinates to the current position of the "gun", defined as an integer below
    if move_type == "relative":
        """
        North = 0
        North-East = 1
        East = 2
        South-East = 3
        South = 4
        South-West = 5
        West = 6
        North-West = 7
        NOOP = 8
        """
        coordinate = env.action_space.sample() 
    #Use absolute coordinates for the position of the "gun", coordinate space are defined below
    else:
        """
        (x,y) coordinates
        Upper left = (0,0)
        Bottom right = (W, H) 
        """
        #img = utils.read_image()
        img = current_frame.copy()
        img = preprocess(img)
        img = img.to(device)
        print("Beginning detection")

        returns = []

        detections = model(img)[0]

        print("Finished detection")

        fileName = f'detections/{indexer}.jpg'
        indexer+=1
        save_detection(detections, current_frame, None)
        

        idx_scores = np.where(detections['scores'].detach().cpu().numpy() > thresh)

        boxes = detections["boxes"][idx_scores].detach().cpu().numpy()
        for box in boxes:
            x, y = centerOfBox(box)
            returns.append({'coordinate' : [x,y], 'move_type' : move_type})

        print("Detection complete")

    return returns


def centerOfBox(box):
    assert box.shape == (4, )

    y = (box[2] + box[0]) / 2
    x = (box[3] + box[1]) / 2

    return x, y

#https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/
def preprocess(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    return image



def save_detection(detections, orig, fname = None):
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > thresh:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format('Duck', confidence * 100)
            print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    
    if fname is not None:
        cv2.imwrite(fname, orig)
