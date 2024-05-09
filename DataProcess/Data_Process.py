'''
Author: AKiy github:Akiy071
Date: 2024-01-11 08:47:54
LastEditors: AKiy
LastEditTime: 2024-04-08 22:52:00
Description: 
'''
import base64
import os
import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from labelme import utils

"""
Split DataSet and Transform Label Image to .Png file.
"""

class DataFileTypeTo():
    def __init__(self, args):
        self.args = args
        self.imgpath = args.image_dir
        self.outdir = args.out_path

    def json2png(self):

        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)

        for dir_name in (os.listdir(self.imgpath)):

            label_dir = self.outdir + "/"+dir_name+"/Label/"

            if not os.path.exists(label_dir):
                os.mkdir(self.outdir + "/"+dir_name)
                os.mkdir(label_dir)

            img_path = self.imgpath+"/"+dir_name
            count = 0
            count += len(os.listdir(img_path))
            print("Find that %d images in %s" % (count//2, img_path))

            if count != 0:
                pbar = tqdm(total=count//2)
                pbar.set_description("Image Translate")

                for file_name in os.listdir(img_path):
                    if file_name.endswith(".json"):

                        json_path = os.path.join(img_path, file_name)

                        if os.path.isfile(json_path):
                            json_data = json.load(open(json_path))
                            imageData = json_data.get("imageData")

                            if not imageData:
                                imagePath = json_data.get("imagePath")
                                # Please Make Sure Your json File and Image in Same directory.
                                imagePath = os.path.join(img_path, imagePath)

                                with open(imagePath, "rb") as f:
                                    imageData = f.read()
                                    imageData = base64.b64encode(imageData).decode(
                                        "utf-8")  # decode & encode

                        img = utils.img_b64_to_arr(imageData)  # Image Data
                        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)

                        label_name2value = {"_background_": 0}
                        for shape in sorted(json_data["shapes"], key=lambda x: x["label"]):
                            label_name = shape["label"]

                            if label_name in label_name2value:
                                label_value = label_name2value[label_name]
                            else:
                                label_value = len(label_name2value)
                                label_name2value[label_name] = label_value

                        label, _ = utils.shapes_to_label(
                            img.shape, json_data["shapes"], label_name2value)
                        label_names = [None]*(max(label_name2value.values())+1)
                        for name, value in label_name2value.items():
                            label_names[value] = name

                        save_path = os.path.join(
                            self.outdir+"/"+dir_name, file_name.split(".")[0]+".jpg")
                        cv2.imwrite(save_path, img)
                        utils.lblsave(os.path.join(label_dir, "%s.png" %
                                      file_name.split(".")[0]), label)

                        pbar.update(1)
                pbar.close()
        print("Image split Done from json to png.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--image_dir", type=str, help="image directory path",
#                         default="Data\BSData-main\Data_Split")
#     parser.add_argument("--out_path", type=str,
#                         help="output directory path", default="./output/")
#     args = parser.parse_args()

#     DataTransformer = DataFileTypeTo(args)
#     DataTransformer.json2png()
