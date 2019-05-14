'''
Created on Sep 3, 2017

@author: Michal.Busta at gmail.com
'''
import os, sys
sys.path.append('./build')


import numpy as np
import torch
import cv2

import net_utils
import argparse
import math

from data_gen import draw_box_points
from models import ModelMLTRCTW
from ocr_utils import print_seq_ext

from demo import resize_image

from nms import get_boxes
import torch.nn.functional as F

import csv
import unicodedata as ud
import editdistance

f = open('codec_rctw.txt', 'r')
codec = f.readlines()[0]
f.close()
print(len(codec))


maping_lang = {}
maping_lang[''] = 'Symbols'
maping_lang['LATIN'] = 'Latin'
maping_lang['DIGIT'] = 'Latin'
maping_lang['ARABIC'] = 'Arabic'
maping_lang['BENGALI'] = 'Bangla'
maping_lang['HANGUL'] = 'Korean'
maping_lang['CJK'] = 'Chinese'
maping_lang['HIRAGANA'] = 'Japanese'
maping_lang['KATAKANA'] = 'Japanese'

scripts = ['', 'DIGIT', 'LATIN', 'ARABIC', 'BENGALI', 'HANGUL', 'CJK', 'HIRAGANA', 'KATAKANA']

eval_text_length = 3

def load_detections(p):
  '''
  load annotation from the text file
  :param p:
  :return:
  '''
  text_polys = []
  if not os.path.exists(p):
    return np.array(text_polys, dtype=np.float32)
  with open(p, 'r') as f:
    reader = csv.reader(f,  delimiter=',',  quotechar='"')
    for line in reader:
      # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
      line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
      
      x2, y2, x3, y3, x4, y4, x1, y1, conf = list(map(float, line[:9]))
      #cls = 0 
      text_polys.append([x1, y1, x2, y2, x3, y3, x4, y4, conf])
     
    return np.array(text_polys, dtype=np.float)
  
def load_gt(p, is_icdar = False):
  '''
  load annotation from the text file, 
  :param p:
  :return:
  '''
  text_polys = []
  text_gts = []
  if not os.path.exists(p):
    return np.array(text_polys, dtype=np.float32), text_gts
  with open(p, 'r') as f:
    reader = csv.reader(f,  delimiter=',',  quotechar='"')
    for line in reader:
      # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
      line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
      
      x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
      #cls = 0
      gt_txt = '' 
      delim = ''
      start_idx = 9
      if is_icdar:
        start_idx = 8
      
      for idx in range(start_idx, len(line)):
        gt_txt += delim + line[idx]
        delim = ','
        
      text_polys.append([x4, y4, x1, y1, x2, y2, x3, y3])
      text_line = gt_txt.strip()
        
      
      text_gts.append(text_line) 
     
    return np.array(text_polys, dtype=np.float), text_gts

def draw_detections(img, boxes, color = (255, 0, 0)):
  
  draw2 = np.copy(img)
  if len(boxes) == 0:
    return draw2
  for i in range(0, boxes.shape[0]):
    pts = boxes[i]
    pts  = pts[0:8]
    pts = pts.reshape(4, -1)
    pts = np.asarray(pts, dtype=np.int)
    draw_box_points(draw2, pts, color=color, thickness=2)
    
  #cv2.imshow('nms', draw2)
  
  return draw2

def intersect(a, b):
  '''Determine the intersection of two rectangles'''
  rect = (0,0,0,0)
  r0 = max(a[0],b[0])
  c0 = max(a[1],b[1])
  r1 = min(a[2],b[2])
  c1 = min(a[3],b[3])
  # Do we have a valid intersection?
  if r1 > r0 and  c1 > c0: 
      rect = (r0,c0,r1,c1)
  return rect

def union(a, b):
  r0 = min(a[0],b[0])
  c0 = min(a[1],b[1])
  r1 = max(a[2],b[2])
  c1 = max(a[3],b[3])
  return (r0,c0,r1,c1)

def area(a):
  '''Computes rectangle area'''
  width = a[2] - a[0]
  height = a[3] - a[1]
  return abs(width * height)

def evaluate_image(img, detections, gt_rect, gt_txts, iou_th=0.5, iou_th_vis=0.5, iou_th_eval=0.5, eval_text_length = 3):
    
  '''
  Summary : Returns end-to-end true-positives, detection true-positives, number of GT to be considered for eval (len > 2).
  Description : For each predicted bounding-box, comparision is made with each GT entry. Values of number of end-to-end true
                positives, number of detection true positives, number of GT entries to be considered for evaluation are computed.
  
  Parameters
  ----------
  iou_th_eval : float
      Threshold value of intersection-over-union used for evaluation of predicted bounding-boxes
  iou_th_vis : float
      Threshold value of intersection-over-union used for visualization when transciption is true but IoU is lesser.
  iou_th : float
      Threshold value of intersection-over-union between GT and prediction.
  word_gto : list of lists
      List of ground-truth bounding boxes along with transcription.
  batch : list of lists
      List containing data (input image, image file name, ground truth).
  detections : tuple of tuples
      Tuple of predicted bounding boxes along with transcriptions and text/no-text score.
  
  Returns
  -------
  tp : int
      Number of predicted bounding-boxes having IoU with GT greater than iou_th_eval.
  tp_e2e : int
      Number of predicted bounding-boxes having same transciption as GT and len > 2.
  gt_e2e : int
      Number of GT entries for which transcription len > 2.
  '''
  
  gt_to_detection = {}
  detection_to_gt = {}
  tp = 0
  tp_e2e = 0
  tp_e2e_ed1 = 0
  gt_e2e = 0
  
  gt_matches = np.zeros(gt_rect.shape[0])
  gt_matches_ed1 = np.zeros(gt_rect.shape[0])
  
  for i in range(0, len(detections)):
      
    det = detections[i]
    box =  det[0] # Predicted bounding-box parameters
    box = np.array(box, dtype="int") # Convert predicted bounding-box to numpy array
    box = box[0:8].reshape(4, 2)
    bbox = cv2.boundingRect(box)
    
    bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
    bbox[2] += bbox[0] # Convert width to right-coordinate
    bbox[3] += bbox[1] # Convert height to bottom-coordinate
    
    det_text = det[1] # Predicted transcription for bounding-box
    
    for gt_no in range(len(gt_rect)):
        
      gtbox = gt_rect[gt_no]
      txt = gt_txts[gt_no] # GT transcription for given GT bounding-box
      gtbox = np.array(gtbox, dtype="int")
      gtbox = gtbox[0:8].reshape(4, 2)
      rect_gt = cv2.boundingRect(gtbox)
      
      
      rect_gt = [rect_gt[0], rect_gt[1], rect_gt[2], rect_gt[3]]
      rect_gt[2] += rect_gt[0] # Convert GT width to right-coordinate
      rect_gt[3] += rect_gt[1] # Convert GT height to bottom-coordinate 

      inter = intersect(bbox, rect_gt) # Intersection of predicted and GT bounding-boxes
      uni = union(bbox, rect_gt) # Union of predicted and GT bounding-boxes
      ratio = area(inter) / float(area(uni)) # IoU measure between predicted and GT bounding-boxes
      
      # 1). Visualize the predicted-bounding box if IoU with GT is higher than IoU threshold (iou_th) (Always required)
      # 2). Visualize the predicted-bounding box if transcription matches the GT and condition 1. holds
      # 3). Visualize the predicted-bounding box if transcription matches and IoU with GT is less than iou_th_vis and 1. and 2. hold
      if ratio > iou_th:
        if not gt_no in gt_to_detection:
          gt_to_detection[gt_no] = [0, 0]
          
        edit_dist = editdistance.eval(det_text.lower(), txt.lower())
        if edit_dist <= 1:
          gt_matches_ed1[gt_no] = 1
          draw_box_points(img, box, color = (0, 128, 0), thickness=2)
            
        if edit_dist == 0: #det_text.lower().find(txt.lower()) != -1:
          draw_box_points(img, box, color = (0, 255, 0), thickness=2)
          gt_matches[gt_no] = 1 # Change this parameter to 1 when predicted transcription is correct.
          
          if ratio < iou_th_vis:
              #draw_box_points(draw, box, color = (255, 255, 255), thickness=2)
              #cv2.imshow('draw', draw) 
              #cv2.waitKey(0)
              pass
                
        tupl = gt_to_detection[gt_no] 
        if tupl[0] < ratio:
          tupl[0] = ratio 
          tupl[1] = i 
          detection_to_gt[i] = [gt_no, ratio, edit_dist]  
                  
  # Count the number of end-to-end and detection true-positives
  for gt_no in range(gt_matches.shape[0]):
    gt = gt_matches[gt_no]
    gt_ed1 = gt_matches_ed1[gt_no]
    txt = gt_txts[gt_no]
    
    gtbox = gt_rect[gt_no]
    gtbox = np.array(gtbox, dtype="int")
    gtbox = gtbox[0:8].reshape(4, 2)
    
    if len(txt) >= eval_text_length and not txt.startswith('##'):
      gt_e2e += 1
      if gt == 1:
        tp_e2e += 1
      if gt_ed1 == 1:
        tp_e2e_ed1 += 1
        
            
    if gt_no in gt_to_detection:
      tupl = gt_to_detection[gt_no] 
      if tupl[0] > iou_th_eval: # Increment detection true-positive, if IoU is greater than iou_th_eval
        if len(txt) >= eval_text_length and not txt.startswith('##'):
          tp += 1   
      #else:
      #  draw_box_points(img, gtbox, color = (255, 255, 255), thickness=2)
        
  for i in range(0, len(detections)):  
    det = detections[i]
    box =  det[0] # Predicted bounding-box parameters
    box = np.array(box, dtype="int") # Convert predicted bounding-box to numpy array
    box = box[0:8].reshape(4, 2)
    
    if not i in detection_to_gt:
      draw_box_points(img, box, color = (0, 0, 255), thickness=2)
    else:
      [gt_no, ratio, edit_dist] = detection_to_gt[i]
      if edit_dist > 0:
        draw_box_points(img, box, color = (255, 0, 0), thickness=2)
            
  #cv2.imshow('draw', draw)             
  return tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt 
  
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 

import glob

def process_splits(trans, word_splits, conf, splits, start, ctc_f, rot_mat, angle, box_points, w, h, draw, is_dict, debug = False):
  '''
  Summary : Split the transciption and corresponding bounding-box based on spaces predicted by recognizer FCN.
  Description : 

  Parameters
  ----------
  trans : string
      String containing the predicted transcription for the corresponding predicted bounding-box.
  conf : list
      List containing sum of confidence for all the character by recognizer FCN, start and end position in bounding-box for generated transciption.
  splits :  list
      List containing index of position of predicted spaces by the recognizer FCN.
  norm2 : matrix
      Matrix containing the cropped bounding-box predicted by localization FCN in the originial image.
  ctc_f : matrix
      Matrix containing output of recognizer FCN for the given input bounding-box.
  rot_mat : matrix
      Rotation matrix returned by get_normalized_image function.
  boxt : tuple of tuples
      Tuple of tuples containing parametes of predicted bounding-box by localization FCN.
  draw : matrix
      Matrix containing input image.
  is_dict : 
  debug : boolean
      Boolean parameter representing debug mode, if it is True visualization boxes are generated.

  Returns
  -------
  boxes_out : list of tuples
      List of tuples containing predicted bounding-box parameters, predicted transcription and mean confidence score from the recognizer.
  '''
  spl = word_splits
  boxout = np.copy(box_points)
  #draw_box_points(draw, boxout, color = (0, 255, 0), thickness=2)
  start_f = start[0,0]
  mean_conf = conf[0, 0] / max(1, len(trans)) # Overall confidence of recognizer FCN
  boxes_out = []
  y = 0
  for s in range(len(spl)):
    text = spl[s]
    end_f = splits[0, s]
    if s < len(spl) - 1:
      try:
        if splits[0, s] > start_f:
            end_f = splits[0, s] # New ending point of bounding-box transcription
      except IndexError:
        pass
    scalex = w / float(ctc_f.shape[1])
    poss = start_f * scalex
    pose = (end_f + 2) * scalex
    rect = [[poss, h], [poss, y], [pose, y], [pose, h]]
    rect = np.array(rect)
    int_t = rot_mat
    dst_rect = np.copy(rect)
    dst_rect[:,0]  = int_t[0,0]*rect[:,0] + int_t[0,1]*rect[:, 1] + int_t[0,2]
    dst_rect[:,1]  = int_t[1,0]*rect[:,0] + int_t[1,1]*rect[:, 1] + int_t[1,2]
    dst_rect[:,0] += boxout[1, 0] 
    dst_rect[:,1] += boxout[1, 1] 
    
    if debug:
      draw_box_points(draw, dst_rect, color = (0, 255, 0))
      cv2.imshow('draw', draw)
      cv2.waitKey(0)
    
    boxes_out.append( (dst_rect, [text, mean_conf, is_dict] ) )
    start_f = end_f + 1
  return boxes_out 
  

if __name__ == '__main__': 
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-cuda', type=int, default=1)
  parser.add_argument('-model', default='e2e-mltrctw.h5')
  parser.add_argument('-images_dir', default='/home/busta/data/ch8_validation_e2e')
  parser.add_argument('-debug', type=int, default=0)
  parser.add_argument('-segm_thresh', default=0.9)
  parser.add_argument('-evaluate', type=int, default=1)
  parser.add_argument('-out_dir', default='eval')
  parser.add_argument('-eval_text_length', type=int, default=3)
  
  args = parser.parse_args()
  
  net = ModelMLTRCTW(attention=True)
  model_name = 'SemanticTexte2e'
  print("Using {0}".format(model_name))

  net_utils.load_net(args.model, net)
  net = net.eval()
    
  if args.cuda:
    print('Using cuda ...')
    net = net.cuda()
  
  images = glob.glob( os.path.join(args.images_dir, '*.jpg') )
  png = glob.glob( os.path.join(args.images_dir, '*.png') )
  images.extend(png)
  
  #cmp_trie.load_dict('/home/busta/data/icdar2013-Test/GenericVocabulary.txt')
  #cmp_trie.load_codec('codec.txt')
  
  tp_all = 0  
  gt_all = 0
  tp_e2e_all = 0
  gt_e2e_all = 0
  tp_e2e_ed1_all = 0
  detecitons_all = 0
  
  im_no = 0
  min_height = 8
    
  
  if not os.path.exists('eval'):
    os.mkdir(args.out_dir)
  if not os.path.exists('preview'):
    os.mkdir('preview')
    
  eval_text_length = args.eval_text_length
  
  nums = []
  image_no = 0
  
  with torch.no_grad():
  
    for img_name in sorted(images):
      try:
        num = int( os.path.basename(img_name).replace('ts_', "").replace("img_", "").replace('.jpg', ""))
        nums.append( (num,  img_name) )
      except:
        nums.append( (image_no,  img_name) )
        image_no += 1
      
    for tp in sorted(nums, key=lambda images: images[0]):
      num = tp[0]
      img_name = tp[1]
      base_nam = os.path.basename(img_name)
      
      if args.evaluate == 1:
        res_gt = base_nam.replace(".jpg", '.txt').replace(".png", '.txt')
        res_gt = '{0}/gt_{1}'.format(args.images_dir, res_gt)
        if not os.path.exists(res_gt):
          res_gt = base_nam.replace(".jpg", '.txt').replace("_", "")
          res_gt = '{0}/gt_{1}'.format(args.images_dir, res_gt)
          if not os.path.exists(res_gt):
            print('missing! {0}'.format(res_gt))
            gt_rect, gt_txts = [], []
            #continue
        gt_rect, gt_txts  = load_gt(res_gt)
      
      print(img_name)
      
      img = cv2.imread(img_name)
      
      #font = cv2.FONT_HERSHEY_SIMPLEX
      #cv2.putText(img,'cs',(10,img.shape[0] -40), font, 0.8,(255,255,255),2,cv2.LINE_AA)
      
      im_resized, (ratio_h, ratio_w) = resize_image(img, max_size=1848*1024, scale_up=True) #1348*1024 #1848*1024
      #im_resized = im_resized[:, :, ::-1]
      images = np.asarray([im_resized], dtype=np.float)
      images /= 128
      images -= 1
      im_data = net_utils.np_to_variable(images, is_cuda=args.cuda).permute(0, 3, 1, 2)
      
      [iou_pred, iou_pred1], rboxs, angle_pred, features = net(im_data) 
      iou = iou_pred.data.cpu()[0].numpy()
      iou = iou.squeeze(0)
      
      iou_pred1 = iou_pred1.data.cpu()[0].numpy()
      iou_pred1 = iou_pred1.squeeze(0)
      
      #ioud = segm_predd.data.cpu()[0].numpy()
      #ioud = ioud.squeeze(0)

      rbox = rboxs[0].data.cpu()[0].numpy()
      rbox = rbox.swapaxes(0, 1)
      rbox = rbox.swapaxes(1, 2)
      
      #rboxd = rboxd.data.cpu()[0].numpy()
      #rboxd = rboxd.swapaxes(0, 1)
      #rboxd = rboxd.swapaxes(1, 2)
      #rboxd = rboxd
      
      #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)) 
      #iou = cv2.erode(iou, kernel,iterations = 1)
      #iou = cv2.blur(iou, (3, 3))
      if args.debug == 1:
        cv2.imshow('iou', iou)
        #cv2.imshow('ioud', ioud)
        cv2.imshow('iou_pred1', iou_pred1)
      
      
      size = 3
      import  scipy.ndimage as ndimage
      image_max = ndimage.maximum_filter(iou, size=size, mode='constant')
      mask = (iou == image_max)
      iou2 = iou * mask
      
        
      if args.debug == 1:
        cv2.imshow('iou2', iou2)
      
      detections = get_boxes(iou, rbox, angle_pred[0].data.cpu()[0].numpy(), args.segm_thresh)
      #detectionsd = get_boxes(iou_pred1, rboxd, angle_pred[1].data.cpu()[0].numpy(), args.segm_thresh, iou_thresh=0.2)
      
      im_scalex = im_resized.shape[1] / img.shape[1]
      im_scaley = im_resized.shape[0] / img.shape[0]
      
      detectionso = np.copy(detections)
      if len(detections) > 0:
        detections[:, 0] /= im_scalex
        detections[:, 2] /= im_scalex
        detections[:, 4] /= im_scalex
        detections[:, 6] /= im_scalex
        
        detections[:, 1] /= im_scaley
        detections[:, 3] /= im_scaley
        detections[:, 5] /= im_scaley
        detections[:, 7] /= im_scaley
      
      
      draw = np.copy(img)
      
      detetcions_out = []
      
      pil_img = Image.fromarray(draw)
      pil_draw = ImageDraw.Draw(pil_img)
      
      font = ImageFont.truetype("Arial-Unicode-Regular.ttf", 16)
      box_no = 0
        
    
      res_file = os.path.join(args.out_dir,  'res_img_{num:05d}.txt'.format(num=num))
      res_file = open(res_file, 'w')
      
      for bid, box in enumerate(detections):      
      
        boxo = detectionso[bid]
        score = boxo[8]
        boxr = boxo[0:8].reshape(-1, 2)
        box_area = area( boxr.reshape(8) )
        
        conf_factor = score / box_area
        #if conf_factor < 0.1:
        #  continue
        
        boxr2 = box[0:8].reshape(-1, 2)
        boxr2[boxr2 < 0] = 0
        if boxr2[:, 0].max() > img.shape[1]:
          continue
        if boxr2[:, 1].max() > img.shape[0]:
          continue
        
        center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4
        
        dw = boxr[2, :] - boxr[1, :]
        dw2 = boxr[0, :] - boxr[3, :]
        dh =  boxr[1, :] - boxr[0, :]
        dh2 =  boxr[3, :] - boxr[2, :]
    
        h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) + 1
        h2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1]) + 1
        h = (h + h2) / 2
        w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
        w2 = math.sqrt(dw2[0] * dw2[0] + dw2[1] * dw2[1])
        w = (w + w2) / 2
        
        if (( h - 1 ) / im_scaley ) < min_height:
          print('too small detection')
          continue
        
        input_W = im_data.size(3)
        input_H = im_data.size(2)
        target_h = 44  
          
        scale = target_h / h
        target_gw = int(w * scale + target_h / 4)  
        target_gw = max(8, int(round(target_gw / 4)) * 4) 
        xc = center[0] 
        yc = center[1] 
        w2 = w 
        h2 = h 
        
        angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
        angle2 = math.atan2((boxr[3][1] - boxr[0][1]), boxr[3][0] - boxr[0][0])
        angle = (angle + angle2) / 2 
        
        #show pooled image in image layer
        scalex = (w2 + h2 / 4) / input_W 
        scaley = h2 / input_H 
      
        th11 =  scalex * math.cos(angle)
        th12 = -math.sin(angle) * scaley * input_H/input_W
        th13 =  (2 * xc - input_W - 1) / (input_W - 1)
        
        th21 = math.sin(angle) * scalex * input_W/input_H
        th22 =  scaley * math.cos(angle)  
        th23 =  (2 * yc - input_H - 1) / (input_H - 1)
                  
        t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
        t = torch.from_numpy(t).type(torch.FloatTensor)
        t = t.cuda()
        theta = t.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
        x = F.grid_sample(im_data, grid)
        
        h2 = 2 * h2
        scalex =  (w2 + int( 2 *  h2)) / input_W
        scaley = h2 / input_H
  
        th11 =  scalex * math.cos(angle)
        th12 = -math.sin(angle) * scaley
        th13 =  (2 * xc - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
  
        th21 = math.sin(angle) * scalex
        th22 =  scaley * math.cos(angle)
        th23 =  (2 * yc - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
  
  
        t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
        t = torch.from_numpy(t).type(torch.FloatTensor)
        t = t.cuda()
        theta = t.view(-1, 2, 3)
        
        grid2 = F.affine_grid(theta, torch.Size((1, 3, int( 2 * target_h), int(target_gw + 2 * target_h ))))
        x2 = F.grid_sample(im_data, grid2)
        
        im = x.data.cpu().numpy()
        im = im.squeeze(0)
        im = im.swapaxes(0, 2)
        im = im.swapaxes(0, 1)
        
        features = net.forward_features(x)
        labels_pred = net.forward_ocr(features)
        
        features2 = net.forward_features(x2)
        offset = (features2.size(2) - features.size(2)) // 2
        offset2 = (features2.size(3) - features.size(3)) // 2
        features2 = features2[:, :, offset:(features.size(2) + offset), offset2:-offset2]
        labels_pred2 = net.forward_ocr(features2)
        
        ctc_f = labels_pred.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)
    
        labels = ctc_f.argmax(2)
        
        ind = np.unravel_index(labels, ctc_f.shape)
        conf = np.mean( np.exp(ctc_f.max(2)[labels > 3]))
        #if conf < 0.4:
        #  print('Too low conf!')
        #  continue 
        
        conf_raw = np.exp(ctc_f[ind])
        
        det_text, conf2, dec_s, word_splits = print_seq_ext(labels[0, :], codec)  
        det_text = det_text.strip()
        
        if args.debug:   
          im += 1
          im *= 128
          cv2.imshow('im', im.astype(np.uint8))
          cv2.waitKey(0)
          
        if args.debug:
          print(det_text)
          
        
        if conf < 0.01 and len(det_text) == 3:
          print('Too low conf short: {0} {1}'.format(det_text, conf))
          continue 
        
        try:
          if len(det_text) > 0 and 'ARABIC' in ud.name(det_text[0]):
            det_text = det_text[::-1]
        except:
          pass
        
        has_long = False
        if len(det_text) > 0:
          rot_mat = cv2.getRotationMatrix2D( (0, 0), -angle * 180 / math.pi, 1 )
          splits_raw = process_splits(det_text, word_splits, conf_raw, dec_s, conf2, ctc_f, rot_mat, angle, boxr, w, h, im_resized, 0) # Process the split and improve the localization
          for spl in splits_raw:
      
            spl[1][0] = spl[1][0].strip()
            
            if len(spl[1][0]) >= eval_text_length:
              has_long = True
              boxw = spl[0]
              boxw[:, 0] /= im_scalex
              boxw[:, 1] /= im_scaley
              draw_box_points(img, boxw, color = (0, 255, 0))
              #cv2.imshow('img', img)
              #cv2.waitKey()
              
              #print('{0} - {1}'.format(spl[1][0], conf_factor))
              #if conf_factor < 0.01:
              #  print('Skipping {0} - {1}'.format(spl[1][0], conf_factor))
              #  continue
              print('{0} - {1}'.format(spl[1][0], conf_factor))
              boxw = boxw.reshape(8)
              detetcions_out.append([boxw, spl[1][0]])
                  
            
        
       
             
      pix = img
      
      if args.evaluate == 1:
        tp, tp_e2e, gt_e2e, tp_e2e_ed1, detection_to_gt  = evaluate_image(pix, detetcions_out, gt_rect, gt_txts, eval_text_length=eval_text_length)
        tp_all += tp 
        gt_all += len(gt_txts)
        tp_e2e_all += tp_e2e
        gt_e2e_all += gt_e2e
        tp_e2e_ed1_all += tp_e2e_ed1
        detecitons_all += len(detetcions_out)
        
        print("  E2E recall {0:.3f} / {1:.3f} / {2:.3f}, precision: {3:.3f}".format( 
          tp_e2e_all / float( max(1, gt_e2e_all) ), 
          tp_all / float( max(1, gt_e2e_all )), 
          tp_e2e_ed1_all / float( max(1, gt_e2e_all )), 
          tp_all /  float( max(1, detecitons_all)) ))
        
      pil_img = Image.fromarray(pix)
      pil_draw = ImageDraw.Draw(pil_img)
      
      det_no = 0
      for box, det_text in detetcions_out:
        
        width, height = pil_draw.textsize(det_text, font=font)
        box = box.reshape(8)
        center =  [box[2] + 3, box[3] - height - 2]
        
        draw_text = det_text
        try:
          if len(det_text) > 0 and 'ARABIC' in ud.name(det_text[0]):
            draw_text = det_text[::-1]
        except:
          pass
        
        pil_draw.text((center[0], center[1]), draw_text, fill = (0,255,0),font=font)
        if args.evaluate == 1 and det_no in detection_to_gt:
          [gt_no, ratio, edit_dist] = detection_to_gt[det_no]
          if edit_dist > 0:
            center[0] += width + 5
            gt_text = gt_txts[gt_no]
            #pil_draw.text((center[0], center[1]), gt_text, fill = (255,0,0),font=font)
  
        det_no += 1
      pix = np.array(pil_img)
              
      cv2.imwrite('preview/{0}'.format(base_nam), pix)
      
      res_file.close()
        
      #if im_no > 100:
      #  break
      im_no += 1
      if args.debug == 1:
        cv2.imshow('pix', pix)
        cv2.waitKey(0)
      
    
    
    
    
    
    
  
