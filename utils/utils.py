import matplotlib.pyplot as plt
import cv2
import json
import os


def load_image(image_path,gray_scale=False):
    image = cv2.imread(image_path)
    if gray_scale:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        return image
    image = image[..., ::-1]
    return image


def plot_image(image,name,binary=False):
    if not binary:
        plt.imshow(image)
    else:
        plt.imshow(image,cmap='binary')
    plt.title(name)
    plt.show()


def parse_json(json_path):
    with open(json_path,'r') as f:
        config = json.load(f)
        return config
