# coding:utf-8
import csv
import cv2
import time
import os
import numpy as np
import math
import random
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from data_util import GeneratorEnqueuer

import PIL
import torchvision.transforms as transforms

use_pyblur = 0

if use_pyblur == 1:
  from pyblur import RandomizedBlur


def get_images(data_path):

  base_dir = os.path.dirname(data_path)
  with open(data_path) as f:
    files = f.readlines()
  files = [x.strip() for x in files]
  files_out = []
  for x in files:
    if len(x) == 0:
      continue
    if not x[0] == '/':
      x = '{0}/{1}'.format(base_dir, x)
    files_out.append(x)
  return files_out

def load_annoataion(p, im):
  '''
  load annotation from the text file
  :param p:
  :return:
  '''
  text_polys = []
  text_tags = []
  labels = []
  if not os.path.exists(p):
    return np.array(text_polys, dtype=np.float), np.array(text_tags, dtype=np.bool), labels

  norm = math.sqrt(im.shape[0] * im.shape[0] + im.shape[1] * im.shape[1])
  with open(p, 'r') as f:
    for line in f.readlines():
      # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
      line = line.replace('\ufeff', '')
      #line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
      line = line.strip()
      splits = line.split(" ")

      cls, x, y, w, h, angle = list(map(float, splits[:6]))
      if angle < -50:
        print("Min angle")
        angle = 0
      rect = ( (x * im.shape[1], y * im.shape[0]), (w * norm, h * norm), angle * 180 / math.pi )
      pts = cv2.boxPoints(rect)

      label = ''
      delim = ''
      for t in splits[6:]:
        label += delim +  t
        delim = ' '

      pts = pts.reshape(-1, 2)
      text_polys.append(pts)
      labels.append(label.strip())

      if label == '*' or label.startswith('###'): # or (w < h):
          text_tags.append(True)
      else:
          text_tags.append(False)

    return np.array(text_polys, dtype=np.float), np.array(text_tags, dtype=np.bool), labels



def load_gt_annoataion(p, is_icdar):
  '''
  load annotation from the text file
  :param p:
  :return:
  '''
  text_polys = []
  text_tags = []
  labels = []
  if not os.path.exists(p):
    return np.array(text_polys, dtype=np.float), np.array(text_tags, dtype=np.bool), labels

  delim = ','
  with open(p, 'r') as f:
    for line in f.readlines():
      line = line.replace('\ufeff', '')
      splits = line.split(delim)

      annotation = splits
      text = ''
      delim = ''
      rs = 9
      if is_icdar:
        rs = 8
      for i in range(rs, len(splits)):
        text += delim + splits[i]
        delim = ','

      text = text.strip()

      annotation = annotation[0:9]
      annotation.append(text.strip())
      pts = list(map(float, annotation[:8]))

      # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
      pts = np.asarray(pts)
      if is_icdar:
        pts = np.roll(pts,2)
      pts = pts.reshape(-1, 2)

      text_polys.append(pts)
      labels.append(text)

      if text == '*' or text.startswith('###'):
          text_tags.append(True)
      else:
          text_tags.append(False)

    return np.array(text_polys, dtype=np.float), np.array(text_tags, dtype=np.bool), labels

def draw_box_points(img, points, color = (0, 255, 0), thickness = 1):
  try:
    cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[1][0]), int(points[1][1])), (255, 0, 0), thickness)
    cv2.circle(img, (int(points[1][0]), int(points[1][1])), 10, (0, 255, 0), -1)
    cv2.line(img, (int(points[2][0]), int(points[2][1])), (int(points[1][0]), int(points[1][1])), (0, 0, 255), thickness)
    cv2.line(img, (int(points[2][0]), int(points[2][1])), (int(points[3][0]), int(points[3][1])), color, thickness)
    cv2.line(img, (int(points[0][0]), int(points[0][1])), (int(points[3][0]), int(points[3][1])), color, thickness)
  except:
    import sys, traceback
    traceback.print_exc(file=sys.stdout)
    pass




def random_rotation(img, word_gto):


  center = (img.shape[1] / 2, img.shape[0] / 2)
  angle =  random.uniform(-1900, 1900) / 10
  M  = cv2.getRotationMatrix2D(center, angle, 1)
  dst_size = (img.shape[1], img.shape[0])
  dst = cv2.warpAffine(img, M, dst_size)

  angle_rad = - angle * math.pi / 180

  wor = np.copy(word_gto)

  word_gto[:, 0, 0] = ((wor[:, 0, 0] - center[0]) * math.cos(angle_rad)) - ((wor[:, 0, 1] - center[1]) * math.sin(angle_rad)) + center[0]
  word_gto[:, 0, 1] = ((wor[:, 0, 0] - center[0]) * math.sin(angle_rad)) + ((wor[:, 0, 1] - center[1]) * math.cos(angle_rad)) + center[1]

  word_gto[:, 1, 0] = ((wor[:, 1, 0] - center[0]) * math.cos(angle_rad)) - ((wor[:, 1, 1] - center[1]) * math.sin(angle_rad)) + center[0]
  word_gto[:, 1, 1] = ((wor[:, 1, 0] - center[0]) * math.sin(angle_rad)) + ((wor[:, 1, 1] - center[1]) * math.cos(angle_rad)) + center[1]

  word_gto[:, 2, 0] = ((wor[:, 2, 0] - center[0]) * math.cos(angle_rad)) - ((wor[:, 2, 1] - center[1]) * math.sin(angle_rad)) + center[0]
  word_gto[:, 2, 1] = ((wor[:, 2, 0] - center[0]) * math.sin(angle_rad)) + ((wor[:, 2, 1] - center[1]) * math.cos(angle_rad)) + center[1]

  word_gto[:, 3, 0] = ((wor[:, 3, 0] - center[0]) * math.cos(angle_rad)) - ((wor[:, 3, 1] - center[1]) * math.sin(angle_rad)) + center[0]
  word_gto[:, 3, 1] = ((wor[:, 3, 0] - center[0]) * math.sin(angle_rad)) + ((wor[:, 3, 1] - center[1]) * math.cos(angle_rad)) + center[1]

  '''
  for i in range(0, len(word_gto) - 1):
    draw_box_points(dst, word_gto[i])
  cv2.imshow('dst', dst)
  cv2.waitKey(0)
  '''
  return dst

def random_perspective(img, word_gto):


  pts1 = np.float32([[0,0],[img.shape[1],0],[img.shape[1],img.shape[0]],[0,img.shape[1]]])
  pts2 = np.float32([[0,0],[img.shape[1],0],[img.shape[1],img.shape[0]],[0,img.shape[1]]])
  M  = cv2.getPerspectiveTransform(pts1, pts2)
  dst_size = (img.shape[1], img.shape[0])
  M[0, 1] = random.uniform(-0.2, 0.2)
  dst = cv2.warpPerspective(img, M, dst_size)
  M_inv = M #np.linalg.inv(M)

  word_gto[:, :, 0] = word_gto[:, :, 0] *  M_inv[0, 0] + word_gto[:, :, 1] *  M_inv[0, 1] +  M_inv[0, 2]
  word_gto[:, :, 1] = word_gto[:, :, 0] *  M_inv[1, 0] + word_gto[:, :, 1] *  M_inv[1, 1] +  M_inv[1, 2]

  return dst


def cut_image(img, new_size, word_gto):

  if len(word_gto) > 0:
    rep = True
    cnt = 0
    while rep:

      if cnt > 30:
        return img

      text_poly = word_gto[random.randint(0, len(word_gto) - 1)]

      center = text_poly.sum(0) / 4

      xs = int(center[0] - random.uniform(-100, 100) - new_size[1] / 2)
      xs = max(xs, 1)
      ys = int(center[1] - random.uniform(-100, 100) - new_size[0] / 2)
      ys = max(ys, 1)

      crop_rect = (xs, ys, xs + new_size[1], ys + new_size[0])
      crop_img = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

      if crop_img.shape[0] == crop_img.shape[1]:
        rep = False
      else:
        cnt += 1


  else:
    xs = int(random.uniform(0, img.shape[1]))
    ys = int(random.uniform(0, img.shape[0]))
    crop_rect = (xs, ys, xs + new_size[1], ys + new_size[0])
    crop_img = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

  if len(word_gto) > 0:
    word_gto[:, :, 0] -= xs
    word_gto[:, :, 1] -= ys

  return crop_img


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    cross = np.linalg.norm(np.cross(p2 - p1, p1 - p3))
    norm = np.linalg.norm(p2 - p1)
    if norm > 0.5:
      return cross / norm
    return cross

def generate_rbox2(im, im_size, polys, tags, labels, vis=False):

  h, w = im_size
  scale_factor = 4

  hs = int(h / scale_factor)
  ws = int(w / scale_factor)

  poly_mask = np.zeros((hs, ws), dtype=np.uint8)
  poly_full = np.zeros((hs, ws), dtype=np.uint8)
  score_map = np.zeros((hs, ws), dtype=np.float32)
  geo_map = np.zeros((hs, ws, 5), dtype=np.float32)

  # mask used during traning, to ignore some hard areas
  training_mask = np.ones((hs, ws), dtype=np.uint8)
  gt_idx = np.ones((hs, ws), dtype=np.int)
  gt_idx *= -1

  labels_out = []
  gt_out = []

  for poly_idx, poly_tag in enumerate(zip(polys, tags)):

    txt = labels[poly_idx]
    pts_orig = poly_tag[0]
    angle = ( math.atan2((pts_orig[2][1] - pts_orig[1][1]), pts_orig[2][0] - pts_orig[1][0]) + math.atan2((pts_orig[3][1] - pts_orig[0][1]), pts_orig[3][0] - pts_orig[0][0]) ) / 2

    tag = poly_tag[1]

    dh1 = pts_orig[1] - pts_orig[0]
    dh2 = pts_orig[2] - pts_orig[3]
    dh1 = math.sqrt(dh1[0] * dh1[0] + dh1[1] * dh1[1])
    dh2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1])

    poly_h = int((dh1 + dh1) / 2)

    dhw = pts_orig[1] - pts_orig[2]
    poly_w = math.sqrt(dhw[0] * dhw[0] + dhw[1] * dhw[1])

    pts = pts_orig / scale_factor
    pts2 = np.copy(pts)

    c1 = ( pts[0] + pts[1] ) / 2
    dh1 = (pts[0] - c1) / 2
    pts[0] = c1 + dh1
    dh2 = (pts[1] - c1) / 2
    pts[1] = c1 + dh2

    c1 = ( pts[2] + pts[3] ) / 2
    dh1 = (pts[2] - c1) / 2
    pts[2] = c1 + dh1
    dh2 = (pts[3] - c1) / 2
    pts[3] = c1 + dh2


    dh1 = pts2[1] - pts2[0]
    dh2 = pts2[2] - pts2[3]

    dh1 = math.sqrt(dh1[0] * dh1[0] + dh1[1] * dh1[1])
    dh2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1])

    if tag == True or poly_h < 6 or poly_w < 6 or np.sum(pts < 0) != 0 or pts_orig[:, 0].max() > im.shape[1] or pts_orig[:, 1].max() > im.shape[1] or (poly_w < poly_h and len(txt) > 3):
      cv2.fillPoly(training_mask, np.asarray([pts2.round()], np.int32), 0)
      continue

    isLine = False

    if txt.find(" ") != -1:

      pts_line = np.copy(pts2)

      c1 = ( pts[1] + pts[2] ) / 2
      dw1 = (pts[2] - c1) / 1.5
      pts_line[2] = c1 + dw1
      dw2 = (pts[1] - c1) / 1.5
      pts_line[1] = c1 + dw2


      c1 = ( pts[0] + pts[3] ) / 2
      dw1 = (pts[3] - c1) / 1.5
      pts_line[3] = c1 + dw1
      dw2 = (pts[0] - c1) / 1.5
      pts_line[0] = c1 + dw2


      cv2.fillPoly(training_mask, np.asarray([pts_line.round()], np.int32), 0)
      isLine = True

    cv2.fillPoly(poly_mask, np.asarray([pts.round()], np.int32), poly_idx + 1)
    cv2.fillPoly(poly_full, np.asarray([pts2.round()], np.int32), poly_idx + 1)

    # TODO filter small
    xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))
    xy_in_polyf = np.argwhere(poly_full == (poly_idx + 1))

    if vis:
      scaled = cv2.resize(im, dsize=(int(im.shape[1] / scale_factor), int(im.shape[0]/ scale_factor)))
      draw_box_points(scaled, pts, (0, 255, 0), 2)
      cv2.imshow('im', scaled)

      pts_o = pts * scale_factor
      draw_box_points(im, pts_o, (255, 0, 0), 2)
      cv2.imshow('orig', im)
      cv2.waitKey(0)


    for y, x in xy_in_poly:
      point = np.array([x, y], dtype=np.float32)
      if score_map[y, x] != 0:
        training_mask[y, x] = 0
        continue


      same_y = xy_in_polyf[xy_in_polyf[:, 0] == point[1]]
      min_x = same_y[:, 1].min()
      max_x = same_y[:, 1].max()
      same_x = xy_in_polyf[xy_in_polyf[:, 1] == point[0]]
      min_y = same_x[:, 0].min()
      max_y = same_x[:, 0].max()

      d1 = point[1] - min_y
      d2 = max_y - point[1]

      dw1 = point[0] -  min_x
      dw2 =  max_x - point[0]

      geo_map[y, x, 0] = d1
      geo_map[y, x, 1] = d2
      geo_map[y, x, 2] = dw1
      if pts_orig[0, 0] > im.shape[1] or pts_orig[1, 0] > im.shape[1] or pts_orig[0, 0] < 0 or pts_orig[1, 0] < 0:
        geo_map[y, x, 2] = -1
      geo_map[y, x, 3] = dw2
      if pts_orig[2, 0] > im.shape[1] or pts_orig[3, 0] > im.shape[1] or pts_orig[2, 0] < 0 or pts_orig[3, 0] < 0:
        geo_map[y, x, 3] = -1

      gt_idx[y, x] = len(gt_out)

      if dw1 < 0.5 or dw2 < 0.5:
        #score_map[y, x] = 0
        training_mask[y, x] = 0

      if isLine:
        if dw1 > dw2:
          geo_map[y, x, 2] = -1
        else:
          geo_map[y, x, 3] = -1


      geo_map[y, x, 4] = angle

    cv2.fillPoly(score_map, np.asarray([pts], np.int32), 1)
    gt_out.append(pts_orig)
    labels_out.append(txt)

  score_map[training_mask == 0] = 0
  score_map = cv2.blur(score_map,(3,3))

  return score_map, geo_map, training_mask, gt_idx, gt_out, labels_out


def generate_rbox(im, im_size, polys, tags, labels, vis=False):

  h, w = im_size
  scale_factor = 4

  hs = int(h / scale_factor)
  ws = int(w / scale_factor)

  poly_mask = np.zeros((hs, ws), dtype=np.uint8)
  score_map = np.zeros((hs, ws), dtype=np.float32)
  geo_map = np.zeros((hs, ws, 5), dtype=np.float32)

  # mask used during traning, to ignore some hard areas
  training_mask = np.ones((hs, ws), dtype=np.uint8)
  gt_idx = np.ones((hs, ws), dtype=np.int)
  gt_idx *= -1

  labels_out = []
  gt_out = []

  for poly_idx, poly_tag in enumerate(zip(polys, tags)):

    txt = labels[poly_idx]
    pts_orig = poly_tag[0]
    angle = ( math.atan2((pts_orig[2][1] - pts_orig[1][1]), pts_orig[2][0] - pts_orig[1][0]) + math.atan2((pts_orig[3][1] - pts_orig[0][1]), pts_orig[3][0] - pts_orig[0][0]) ) / 2

    tag = poly_tag[1]

    dh1 = pts_orig[1] - pts_orig[0]
    dh2 = pts_orig[2] - pts_orig[3]
    dh1 = math.sqrt(dh1[0] * dh1[0] + dh1[1] * dh1[1])
    dh2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1])

    poly_h = int((dh1 + dh1) / 2)

    dhw = pts_orig[1] - pts_orig[2]
    poly_w = math.sqrt(dhw[0] * dhw[0] + dhw[1] * dhw[1])

    pts = pts_orig / scale_factor
    pts2 = np.copy(pts)

    c1 = ( pts[0] + pts[1] ) / 2
    dh1 = (pts[0] - c1) / 1.5
    pts[0] = c1 + dh1
    dh2 = (pts[1] - c1) / 1.5
    pts[1] = c1 + dh2

    c1 = ( pts[2] + pts[3] ) / 2
    dh1 = (pts[2] - c1) / 1.5
    pts[2] = c1 + dh1
    dh2 = (pts[3] - c1) / 1.5
    pts[3] = c1 + dh2


    dh1 = pts2[1] - pts2[0]
    dh2 = pts2[2] - pts2[3]

    dh1 = math.sqrt(dh1[0] * dh1[0] + dh1[1] * dh1[1])
    dh2 = math.sqrt(dh2[0] * dh2[0] + dh2[1] * dh2[1])



    if tag == True or poly_h < 6 or poly_w < 6 or np.sum(pts < 0) != 0 or pts_orig[:, 0].max() > im.shape[1] or pts_orig[:, 1].max() > im.shape[1] or (poly_w < poly_h and len(txt) > 3):
      cv2.fillPoly(training_mask, np.asarray([pts2.round()], np.int32), 0)
      continue

    isLine = False

    if txt.find(" ") != -1:

      pts_line = np.copy(pts2)

      c1 = ( pts[1] + pts[2] ) / 2
      dw1 = (pts[2] - c1) / 1.2
      pts_line[2] = c1 + dw1
      dw2 = (pts[1] - c1) / 1.2
      pts_line[1] = c1 + dw2


      c1 = ( pts[0] + pts[3] ) / 2
      dw1 = (pts[3] - c1) / 1.2
      pts_line[3] = c1 + dw1
      dw2 = (pts[0] - c1) / 1.2
      pts_line[0] = c1 + dw2


      cv2.fillPoly(training_mask, np.asarray([pts_line.round()], np.int32), 0)
      isLine = True

    cv2.fillPoly(poly_mask, np.asarray([pts.round()], np.int32), poly_idx + 1)
    # TODO filter small
    xy_in_poly = np.argwhere(poly_mask == (poly_idx + 1))

    if vis:
      scaled = cv2.resize(im, dsize=(int(im.shape[1] / scale_factor), int(im.shape[0]/ scale_factor)))
      draw_box_points(scaled, pts, (0, 255, 0), 2)
      cv2.imshow('im', scaled)

      pts_o = pts * scale_factor
      draw_box_points(im, pts_o, (255, 0, 0), 2)
      cv2.imshow('orig', im)
      cv2.waitKey(0)


    for y, x in xy_in_poly:
      point = np.array([x, y], dtype=np.float32)
      if score_map[y, x] != 0:
        training_mask[y, x] = 0
        continue

      d1 = point_dist_to_line(pts2[1], pts2[2], point)
      d2 = point_dist_to_line(pts2[0], pts2[3], point)
      dw1 = point_dist_to_line(pts2[0], pts2[1], point)
      dw2 = point_dist_to_line(pts2[2], pts2[3], point)

      geo_map[y, x, 0] = d1
      geo_map[y, x, 1] = d2
      geo_map[y, x, 2] = dw1
      if pts_orig[0, 0] > im.shape[1] or pts_orig[1, 0] > im.shape[1] or pts_orig[0, 0] < 0 or pts_orig[1, 0] < 0:
        geo_map[y, x, 2] = -1
      geo_map[y, x, 3] = dw2
      if pts_orig[2, 0] > im.shape[1] or pts_orig[3, 0] > im.shape[1] or pts_orig[2, 0] < 0 or pts_orig[3, 0] < 0:
        geo_map[y, x, 3] = -1

      gt_idx[y, x] = len(gt_out)

      if dw1 < 1 or dw2 < 1:
        score_map[y, x] = 0

      if isLine:
        if dw1 > dw2:
          geo_map[y, x, 2] = -1
        else:
          geo_map[y, x, 3] = -1


      geo_map[y, x, 4] = angle

    cv2.fillPoly(score_map, np.asarray([pts.round()], np.int32), 1)
    gt_out.append(pts_orig)
    labels_out.append(txt)

  score_map[training_mask == 0] = 0
  #score_map = cv2.blur(score_map,(3,3))

  return score_map, geo_map, training_mask, gt_idx, gt_out, labels_out



def generator(input_size=512, batch_size=4, train_list='/home/klara/klara/home/DeepSemanticText/resources/ims2.txt', vis=False, in_train=True, geo_type = 0):
  image_list = np.array(get_images(train_list))
  print('{} training images in {}'.format(image_list.shape[0], train_list))
  index = np.arange(0, image_list.shape[0])

  allow_empty = False
  if not in_train:
    allow_empty = True

  transform = transforms.Compose([
            transforms.ColorJitter(.3,.3,.3,.3),
            transforms.RandomGrayscale(p=0.1)
        ])

  while True:
    if in_train:
      np.random.shuffle(index)
    images = []
    image_fns = []
    score_maps = []
    geo_maps = []
    training_masks = []
    gtso = []
    lbso = []
    gt_idxs = []
    im_id = 0
    for i in index:
      try:
        im_name = image_list[i]

        if in_train:
          if random.uniform(0, 100) < 80:
            im_name = image_list[int(random.uniform(0, min(19000,image_list.shape[0] - 1)))] #use with synthetic data

        if not os.path.exists(im_name):
          continue

        im = cv2.imread(im_name)
        if im is None:
          continue

        allow_empty = False
        
        print(im_name)
        name = os.path.basename(im_name)
        name = name[:-4]

        # print im_fn
        h, w, _ = im.shape
        txt_fn = im_name.replace(os.path.basename(im_name).split('.')[1], 'txt')
        base_name = os.path.basename(txt_fn)
        txt_fn_gt = '{0}/gt_{1}'.format(os.path.dirname(im_name), base_name)
        if ( not ( os.path.exists(txt_fn) or os.path.exists(txt_fn_gt)  ) )  and not allow_empty:
          continue

        allow_empty = random.randint(0, 100) < 40

        if os.path.exists(txt_fn_gt) and (txt_fn_gt.find('/done/') != -1 or txt_fn_gt.find('/icdar-2015-Ch4/') != -1):
          text_polys, text_tags, labels_txt = load_gt_annoataion(txt_fn_gt, txt_fn_gt.find('/icdar-2015-Ch4/') != -1)
        elif os.path.exists(txt_fn) and (txt_fn.find('/Latin/') != -1 or txt_fn.find('/Arabic/') != -1 or txt_fn.find('/Chinese/') != -1 or txt_fn.find('/Japanese/') != -1 or txt_fn.find('/Bangla/') != -1):
          try:
            text_polys, text_tags, labels_txt = load_annoataion(txt_fn, im)
          except:
            print(txt_fn)
            import traceback
            traceback.print_exc()
            os.remove( im_name )
            os.remove( txt_fn )
            continue

        else:
          text_polys, text_tags, labels_txt = load_annoataion(txt_fn, im)
        if in_train:

          if random.uniform(0, 100) < 50 or im.shape[0] < 600 or im.shape[1] < 600:
            top = int(random.uniform(300, 500))
            bottom = int(random.uniform(300, 500))
            left = int(random.uniform(300, 500))
            right = int(random.uniform(300, 500))
            im = cv2.copyMakeBorder(im, top , bottom, left, right, cv2.BORDER_CONSTANT)
            if len(text_polys) > 0:
              text_polys[:, :, 0] += left
              text_polys[:, :, 1] += top

          if random.uniform(0, 100) < 30 and False:
            im = random_rotation(im, text_polys)
          if random.uniform(0, 100) < 30:
            im = random_perspective(im, text_polys)

          #im = random_crop(im, text_polys, vis=False)

          scalex = random.uniform(0.5, 2)
          scaley = scalex * random.uniform(0.8, 1.2)
          im = cv2.resize(im, dsize=(int(im.shape[1] * scalex), int(im.shape[0] * scaley)))
          text_polys[:, :, 0] *= scalex
          text_polys[:, :, 1] *= scaley

          if random.randint(0, 100) < 10:
            im = np.invert(im)

        new_h, new_w, _ = im.shape
        resize_h = input_size
        resize_w = input_size
        if input_size == -1:
          image_size = [im.shape[1] // 32 * 32, im.shape[0] // 32 * 32]
          while image_size[0] * image_size[1] > 1024 * 1024:
            image_size[0] /= 1.2
            image_size[1] /= 1.2
            image_size[0] = int(image_size[0] // 32) * 32
            image_size[1] = int(image_size[1] // 32) * 32

          resize_h = int(image_size[1])
          resize_w = int(image_size[0])


        scaled = cut_image(im,  (resize_w, resize_w), text_polys)
        if scaled.shape[0] == 0 or scaled.shape[1] == 0:
          continue

        #transform_boxes(im, scaled, text_polys, text_tags, vis=False)

        if scaled.shape[1] != resize_w or  scaled.shape[0] != resize_h:

          #continue
          scalex = scaled.shape[1] / resize_w
          scaley = scaled.shape[0] / resize_h

          if scalex < 0.5 or scaley < 0.5:
            continue
          scaled = cv2.resize(scaled, dsize=(int(resize_w), int(resize_h)))

          if len(text_polys) > 0:
            text_polys[:, :, 0] /= scalex
            text_polys[:, :, 1] /= scaley


        im = scaled
        new_h, new_w, _ = im.shape


        pim = PIL.Image.fromarray(np.uint8(im))
        pim = transform(pim)

        if use_pyblur == 1 and random.uniform(0, 100) < 30:
          pim = RandomizedBlur(pim)

        im = np.array(pim)
        if geo_type == 0:
          score_map, geo_map, training_mask, gt_idx, gt_out, labels_out = generate_rbox(im, (new_h, new_w), text_polys, text_tags, labels_txt, vis=vis)
        else:
          score_map, geo_map, training_mask, gt_idx, gt_out, labels_out = generate_rbox2(im, (new_h, new_w), text_polys, text_tags, labels_txt, vis=vis)

        if score_map.sum() == 0 and (not allow_empty):
          #print('empty image')
          continue

        image_fns.append(im_name)
        images.append(im[:, :, :].astype(np.float32))
        gtso.append(gt_out)
        lbso.append(labels_out)
        training_masks.append(training_mask)
        score_maps.append(score_map)
        gt_idxs.append(gt_idx)
        geo_maps.append(geo_map)

        im_id+=1


        if len(images) == batch_size:
          images = np.asarray(images, dtype=np.float)
          images /= 128
          images -= 1

          training_masks = np.asarray(training_masks, dtype=np.uint8)
          score_maps = np.asarray(score_maps, dtype=np.uint8)
          geo_maps = np.asarray(geo_maps, dtype=np.float)
          gt_idxs = np.asarray(gt_idxs, dtype=np.int)

          yield images, image_fns, score_maps, geo_maps, training_masks, gtso, lbso, gt_idxs
          images = []
          image_fns = []
          geo_maps = []
          score_maps = []
          geo_maps = []
          training_masks = []
          gtso = []
          lbso = []
          gt_idxs = []
          im_id = 0
      except Exception as e:
        import traceback
        traceback.print_exc()
        continue

    if not in_train:
      print("finish")
      yield None
      break


def get_batch(num_workers, **kwargs):
  try:
    enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
    enqueuer.start(max_queue_size=24, workers=num_workers)
    generator_output = None
    while True:
      while enqueuer.is_running():
        if not enqueuer.queue.empty():
          generator_output = enqueuer.queue.get()
          break
        else:
          time.sleep(0.01)
      yield generator_output
      generator_output = None
  finally:
    if enqueuer is not None:
      enqueuer.stop()

if __name__ == '__main__':

  data_generator = get_batch(num_workers=1,
           input_size=544, batch_size=1,
           train_list='/mnt/textspotter/tmp/SynthText/trainArabic.txt', vis=True)
  while True:
    images, image_fns, score_maps, geo_maps, training_masks, gtso, lbso, gt_idxs = next(data_generator)
    print(image_fns)

