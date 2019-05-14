'''
Created on Aug 25, 2017

@author: busta
'''

import cv2
import numpy as np

from nms import get_boxes

from models import ModelResNetSep2
import net_utils

from ocr_utils import ocr_image
from data_gen import draw_box_points
import torch

import argparse

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

f = open('codec.txt', 'r', encoding='utf-8')
codec = f.readlines()[0]
f.close()

def resize_image(im, max_size = 1585152, scale_up=True):

  if scale_up:
    image_size = [im.shape[1] * 3 // 32 * 32, im.shape[0] * 3 // 32 * 32]
  else:
    image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
  while image_size[0] * image_size[1] > max_size:
    image_size[0] /= 1.2
    image_size[1] /= 1.2
    image_size[0] = int(image_size[0] // 32) * 32
    image_size[1] = int(image_size[1] // 32) * 32


  resize_h = int(image_size[1])
  resize_w = int(image_size[0])


  scaled = cv2.resize(im, dsize=(resize_w, resize_h))
  return scaled, (resize_h, resize_w)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='e2e-mlt.h5')
  parser.add_argument('-segm_thresh', default=0.5)

  font2 = ImageFont.truetype("Arial-Unicode-Regular.ttf", 18)

  args = parser.parse_args()

  net = ModelResNetSep2(attention=True)
  net_utils.load_net(args.model, net)
  net = net.eval()

  if args.cuda:
    print('Using cuda ...')
    net = net.cuda()

  cap = cv2.VideoCapture(0)
  cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
  ret, im = cap.read()

  frame_no = 0
  with torch.no_grad():
    while ret:
      ret, im = cap.read()

      if ret==True:
        im_resized, (ratio_h, ratio_w) = resize_image(im, scale_up=False)
        images = np.asarray([im_resized], dtype=np.float)
        images /= 128
        images -= 1
        im_data = net_utils.np_to_variable(images, is_cuda=args.cuda).permute(0, 3, 1, 2)
        seg_pred, rboxs, angle_pred, features = net(im_data)

        rbox = rboxs[0].data.cpu()[0].numpy()
        rbox = rbox.swapaxes(0, 1)
        rbox = rbox.swapaxes(1, 2)

        angle_pred = angle_pred[0].data.cpu()[0].numpy()


        segm = seg_pred[0].data.cpu()[0].numpy()
        segm = segm.squeeze(0)

        draw2 = np.copy(im_resized)
        boxes =  get_boxes(segm, rbox, angle_pred, args.segm_thresh)

        img = Image.fromarray(draw2)
        draw = ImageDraw.Draw(img)

        #if len(boxes) > 10:
        #  boxes = boxes[0:10]

        out_boxes = []
        for box in boxes:

          pts  = box[0:8]
          pts = pts.reshape(4, -1)

          det_text, conf, dec_s = ocr_image(net, codec, im_data, box)
          if len(det_text) == 0:
            continue

          width, height = draw.textsize(det_text, font=font2)
          center =  [box[0], box[1]]
          draw.text((center[0], center[1]), det_text, fill = (0,255,0),font=font2)
          out_boxes.append(box)
          print(det_text)

        im = np.array(img)
        for box in out_boxes:
          pts  = box[0:8]
          pts = pts.reshape(4, -1)
          draw_box_points(im, pts, color=(0, 255, 0), thickness=1)

        cv2.imshow('img', im)
        cv2.waitKey(10)


