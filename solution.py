import time
from detecto import core, utils, visualize
import numpy as np
import torch
from numba import jit, cuda
import tensorflow as tf

"""
Replace following with your own algorithm logic

Two random coordinate generator has been provided for testing purposes.
Manual mode where you can use your mouse as also been added for testing purposes.
"""
coords = []
thresh = 0.6
model = core.Model.load('model_weights.pth', ['DUCK'])

def GetLocation(move_type, env, current_frame):
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
        print("RUNNING MODEL")

        """pred = model.predict(current_frame)
        labels, boxes, scores = pred

        filtered_idx = np.where(scores > thresh)
        max_idx = np.argmax(scores)
        filtered_scores=scores[filtered_idx]
        filtered_boxes=boxes[filtered_idx]
        num_list = filtered_idx[0].tolist()
        filtered_labels = [labels[i] for i in num_list]

        for box in filtered_boxes:
            x, y = centerOfBox(box)
            coords.append(np.array([x,y]))

        #visualize.show_labeled_image(current_frame, filtered_boxes, filtered_labels)

        coordinate = coords.pop(0)"""
        with tf.device('/device:GPU:0'):
            labels, boxes, scores  = model.predict_top(current_frame)
            x, y = centerOfBox(boxes[0])

            coordinate = env.action_space_abs.sample()
        
            coordinate[0] = x
            coordinate[1] = y

    return [{'coordinate' : coordinate, 'move_type' : move_type}]


def centerOfBox(box):
    assert box.shape == (4, )

    y = (box[2] + box[0]) / 2
    x = (box[3] + box[1]) / 2

    return x, y
