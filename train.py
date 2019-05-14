'''
Created on Sep 3, 2017

@author: Michal.Busta at gmail.com
'''

import torch, os
import numpy as np
import cv2

import net_utils
import data_gen
from data_gen import draw_box_points
import timeit

import math
import random

from models import ModelResNetSep2
import torch.autograd as autograd
import torch.nn.functional as F

from torch_baidu_ctc import ctc_loss, CTCLoss
#from warpctc_pytorch import CTCLoss
from ocr_test_utils import print_seq_ext


import unicodedata as ud
import ocr_gen
from torch import optim

lr_decay = 0.99
momentum = 0.9
weight_decay = 0
batch_per_epoch = 1000
disp_interval = 100

norm_height = 44

f = open('codec.txt', 'r')
codec = f.readlines()[0]
#codec = u' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_abcdefghijklmnopqrstuvwxyz{|}~£ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýž'
codec_rev = {}
index = 4
for i in range(0, len(codec)):
  codec_rev[codec[i]] = index
  index += 1
f.close()

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
  return width * height
  
def process_boxes(images, im_data, iou_pred, roi_pred, angle_pred, score_maps, gt_idxs, gtso, lbso, features, net, ctc_loss, opts, debug = False):
  
  ctc_loss_count = 0
  loss = torch.from_numpy(np.asarray([0])).type(torch.FloatTensor).cuda()
  
  for bid in range(iou_pred.size(0)):
    
    gts = gtso[bid]
    lbs = lbso[bid]
    
    gt_proc = 0
    gt_good = 0
    
    gts_count = {}

    iou_pred_np = iou_pred[bid].data.cpu().numpy()
    iou_map = score_maps[bid]
    to_walk = iou_pred_np.squeeze(0) * iou_map * (iou_pred_np.squeeze(0) > 0.5)
    
    roi_p_bid = roi_pred[bid].data.cpu().numpy()
    gt_idx = gt_idxs[bid]
    
    if debug:
      img = images[bid]
      img += 1
      img *= 128
      img = np.asarray(img, dtype=np.uint8)
    
    xy_text = np.argwhere(to_walk > 0)
    random.shuffle(xy_text)
    xy_text = xy_text[0:min(xy_text.shape[0], 100)]
    
    
    for i in range(0, xy_text.shape[0]):
      if opts.geo_type == 1:
        break
      pos = xy_text[i, :]
      
      gt_id = gt_idx[pos[0], pos[1]]
      
      if not gt_id in gts_count:
        gts_count[gt_id] = 0
        
      if gts_count[gt_id] > 2:
        continue
      
      gt = gts[gt_id]
      gt_txt = lbs[gt_id]
      if gt_txt.startswith('##'):
        continue
      
      angle_sin = angle_pred[bid, 0, pos[0], pos[1]] 
      angle_cos = angle_pred[bid, 1, pos[0], pos[1]] 
      
      angle = math.atan2(angle_sin, angle_cos)
      
      angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
      
      if math.fabs(angle_gt - angle) > math.pi / 16:
        continue
      
      offset = roi_p_bid[:, pos[0], pos[1]]
      posp = pos + 0.25
      pos_g = np.array([(posp[1] - offset[0] * math.sin(angle)) * 4, (posp[0] - offset[0] * math.cos(angle)) * 4 ])
      pos_g2 = np.array([ (posp[1] + offset[1] * math.sin(angle)) * 4, (posp[0] + offset[1] * math.cos(angle)) * 4 ])
    
      pos_r = np.array([(posp[1] - offset[2] * math.cos(angle)) * 4, (posp[0] - offset[2] * math.sin(angle)) * 4 ])
      pos_r2 = np.array([(posp[1] + offset[3] * math.cos(angle)) * 4, (posp[0] + offset[3] * math.sin(angle)) * 4 ])
      
      center = (pos_g + pos_g2 + pos_r + pos_r2) / 2 - [4*pos[1], 4*pos[0]]    
      #center = (pos_g + pos_g2 + pos_r + pos_r2) / 4
      dw = pos_r - pos_r2
      dh =  pos_g - pos_g2
    
      w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
      h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1]) 
      
      dhgt =  gt[1] - gt[0]
      
      h_gt = math.sqrt(dhgt[0] * dhgt[0] + dhgt[1] * dhgt[1])
      if h_gt < 10:
        continue
    
      rect = ( (center[0], center[1]), (w, h), angle * 180 / math.pi )
      pts = cv2.boxPoints(rect)
      
      pred_bbox = cv2.boundingRect(pts)
      pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]]
      pred_bbox[2] += pred_bbox[0]
      pred_bbox[3] += pred_bbox[1]
      
      if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(3):
        continue 
      
      gt_bbox = [gt[:, 0].min(), gt[:, 1].min(), gt[:, 0].max(), gt[:, 1].max()]
      inter = intersect(pred_bbox, gt_bbox)
      
      uni = union(pred_bbox, gt_bbox)
      ratio = area(inter) / float(area(uni)) 
      
      if ratio < 0.90:
        continue
      
      hratio = min(h, h_gt) / max(h, h_gt)
      if hratio < 0.5:
        continue
        
      input_W = im_data.size(3)
      input_H = im_data.size(2)
      target_h = norm_height  
        
      scale = target_h / h 
      target_gw = (int(w * scale) + target_h // 2)
      target_gw = max(8, int(round(target_gw / 4)) * 4) 
      
      #show pooled image in image layer
    
      scalex = (w + h // 2) / input_W 
      scaley = h / input_H 

    
      th11 =  scalex * math.cos(angle)
      th12 = -math.sin(angle) * scaley
      th13 =  (2 * center[0] - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
      
      th21 = math.sin(angle) * scalex 
      th22 =  scaley * math.cos(angle)  
      th23 =  (2 * center[1] - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
      
      
      t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
      t = torch.from_numpy(t).type(torch.FloatTensor).cuda()
      
      #t = torch.stack((th11, th12, th13, th21, th22, th23), dim=1)
      theta = t.view(-1, 2, 3)
      
      grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
      
      x = F.grid_sample(im_data[bid].unsqueeze(0), grid)

      h2 = 2 * h
      scalex =  (w + int(h2)) / input_W
      scaley = h2 / input_H

      th11 =  scalex * math.cos(angle_gt)
      th12 = -math.sin(angle_gt) * scaley
      th13 =  (2 * center[0] - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)

      th21 = math.sin(angle_gt) * scalex
      th22 =  scaley * math.cos(angle_gt)
      th23 =  (2 * center[1] - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)


      t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
      t = torch.from_numpy(t).type(torch.FloatTensor)
      t = t.cuda()
      theta = t.view(-1, 2, 3)
      
      grid2 = F.affine_grid(theta, torch.Size((1, 3, int( 2 * target_h), int(target_gw + target_h ))))
      x2 = F.grid_sample(im_data[bid].unsqueeze(0), grid2)
      
      if debug:
        x_c = x.data.cpu().numpy()[0]
        x_data_draw = x_c.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)
      
        x_data_draw += 1
        x_data_draw *= 128
        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        x_data_draw = x_data_draw[:, :, ::-1]
      
        cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 0))      
        cv2.imshow('im_data', x_data_draw)
      
        draw_box_points(img, pts)
        draw_box_points(img, gt, color=(0, 0, 255))
      
        cv2.imshow('img', img)
        cv2.waitKey(100)
      
      gt_labels = []
      gt_labels.append( codec_rev[' '] )
      for k in range(len(gt_txt)):
        if gt_txt[k] in codec_rev:                
          gt_labels.append( codec_rev[gt_txt[k]] )
        else:
          print('Unknown char: {0}'.format(gt_txt[k]) )
          gt_labels.append( 3 )
          
      if 'ARABIC' in ud.name(gt_txt[0]):
          gt_labels = gt_labels[::-1]
      gt_labels.append( codec_rev[' '] )
      
      
      features = net.forward_features(x)
      labels_pred = net.forward_ocr(features)

      fs2 = net.forward_features(x2)
      offset = (fs2.size(2) - features.size(2)) // 2
      offset2 = (fs2.size(3) - features.size(3)) // 2
      fs2 = fs2[:, :, offset:(features.size(2) + offset), offset2:-offset2]
      labels_pred2 = net.forward_ocr(fs2)
      
      label_length = []
      label_length.append(len(gt_labels))
      probs_sizes =  autograd.Variable(torch.IntTensor( [(labels_pred.permute(2,0,1).size()[0])] * (labels_pred.permute(2,0,1).size()[1]) ))
      label_sizes = autograd.Variable(torch.IntTensor( torch.from_numpy(np.array(label_length)).int() ))
      labels = autograd.Variable(torch.IntTensor( torch.from_numpy(np.array(gt_labels)).int() ))    
      
      loss = loss + ctc_loss(labels_pred.permute(2,0,1), labels, probs_sizes, label_sizes).cuda()
      loss = loss + ctc_loss(labels_pred2.permute(2,0,1), labels, probs_sizes, label_sizes).cuda()
      ctc_loss_count += 1
      
      if debug:
        ctc_f = labels_pred.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)
    
        labels = ctc_f.argmax(2)
        det_text, conf, dec_s, splits = print_seq_ext(labels[0, :], codec)  
        
        print('{0} \t {1}'.format(det_text, gt_txt))
        
      gts_count[gt_id] += 1
        
      if ctc_loss_count > 64 or debug:
        break
    
    for gt_id in range(0, len(gts)):
      
      gt = gts[gt_id]
      gt_txt = lbs[gt_id]
      
      gt_txt_low = gt_txt.lower()
      if gt_txt.startswith('##'):
        continue
      
      if gt[:, 0].max() > im_data.size(3) or gt[:, 1].max() > im_data.size(3) :
        continue 
      
      if gt.min() < 0:
        continue
      
      center = (gt[0, :] + gt[1, :] + gt[2, :] + gt[3, :]) / 4
      dw = gt[2, :] - gt[1, :]
      dh =  gt[1, :] - gt[0, :] 
      
      w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
      h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])  + random.randint(-2, 2)
      
      if h < 8:
        #print('too small h!')
        continue
      
      angle_gt = ( math.atan2((gt[2][1] - gt[1][1]), gt[2][0] - gt[1][0]) + math.atan2((gt[3][1] - gt[0][1]), gt[3][0] - gt[0][0]) ) / 2
             
      input_W = im_data.size(3)
      input_H = im_data.size(2)
      target_h = norm_height  
        
      scale = target_h / h 
      target_gw = int(w * scale) + random.randint(0, int(target_h)) 
      target_gw = max(8, int(round(target_gw / 4)) * 4) 
        
      xc = center[0] 
      yc = center[1] 
      w2 = w 
      h2 = h 
      
      #show pooled image in image layer
    
      scalex =  (w2 + random.randint(0, int(h2))) / input_W 
      scaley = h2 / input_H 
    
      th11 =  scalex * math.cos(angle_gt)
      th12 = -math.sin(angle_gt) * scaley
      th13 =  (2 * xc - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
      
      th21 = math.sin(angle_gt) * scalex 
      th22 =  scaley * math.cos(angle_gt)  
      th23 =  (2 * yc - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
        
      
      t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=np.float)
      t = torch.from_numpy(t).type(torch.FloatTensor)
      t = t.cuda()
      theta = t.view(-1, 2, 3)
      
      grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h ), int(target_gw))))
      x = F.grid_sample(im_data[bid].unsqueeze(0), grid)
      
      #score_sampled = F.grid_sample(iou_pred[bid].unsqueeze(0), grid)
      
      gt_labels = []
      gt_labels.append(codec_rev[' '])
      for k in range(len(gt_txt)):
        if gt_txt[k] in codec_rev:                
          gt_labels.append( codec_rev[gt_txt[k]] )
        else:
          print('Unknown char: {0}'.format(gt_txt[k]) )
          gt_labels.append( 3 )
      gt_labels.append(codec_rev[' '])
          
      if 'ARABIC' in ud.name(gt_txt[0]):
          gt_labels = gt_labels[::-1]
      
      features = net.forward_features(x)
      labels_pred = net.forward_ocr(features)
      
      label_length = []
      label_length.append(len(gt_labels))
      probs_sizes =  torch.IntTensor( [(labels_pred.permute(2,0,1).size()[0])] * (labels_pred.permute(2,0,1).size()[1]) )
      label_sizes = torch.IntTensor( torch.from_numpy(np.array(label_length)).int() )
      labels = torch.IntTensor( torch.from_numpy(np.array(gt_labels)).int() )
      
      loss = loss + ctc_loss(labels_pred.permute(2,0,1), labels, probs_sizes, label_sizes).cuda()
      ctc_loss_count += 1
      
      if debug:
        x_d = x.data.cpu().numpy()[0]
        x_data_draw = x_d.swapaxes(0, 2)
        x_data_draw = x_data_draw.swapaxes(0, 1)
      
        x_data_draw += 1
        x_data_draw *= 128
        x_data_draw = np.asarray(x_data_draw, dtype=np.uint8)
        x_data_draw = x_data_draw[:, :, ::-1]
        cv2.imshow('im_data_gt', x_data_draw)
        cv2.waitKey(100)
      
      gt_proc += 1
      if True:
        ctc_f = labels_pred.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)
    
        labels = ctc_f.argmax(2)
        det_text, conf, dec_s, splits = print_seq_ext(labels[0, :], codec)  
        if debug:
          print('{0} \t {1}'.format(det_text, gt_txt))
        if det_text.lower() == gt_txt.lower():
          gt_good += 1
        
      if ctc_loss_count > 128 or debug:
        break      
    
  if ctc_loss_count > 0:
    loss /= ctc_loss_count
    
  return loss, gt_good , gt_proc
  
     
def main(opts):
  
  model_name = 'E2E-MLT'
  net = ModelResNetSep2(attention=True)
  print("Using {0}".format(model_name))
  
  learning_rate = opts.base_lr
  if opts.cuda:
    net.cuda()
  optimizer = torch.optim.Adam(net.parameters(), lr=opts.base_lr, weight_decay=weight_decay)
  step_start = 0  
  if os.path.exists(opts.model):
    print('loading model from %s' % args.model)
    step_start, learning_rate = net_utils.load_net(args.model, net, optimizer)
  step_start = 0
  if opts.cuda:
    net.cuda()
    
  net.train()

  data_generator = data_gen.get_batch(num_workers=opts.num_readers, 
           input_size=opts.input_size, batch_size=opts.batch_size, 
           train_list=opts.train_list, geo_type=opts.geo_type)
  
  dg_ocr = ocr_gen.get_batch(num_workers=2,
          batch_size=opts.ocr_batch_size, 
          train_list=opts.ocr_feed_list, in_train=True, norm_height=norm_height, rgb=True)
  
  train_loss = 0
  bbox_loss, seg_loss, angle_loss = 0., 0., 0.
  cnt = 0
  ctc_loss = CTCLoss()
  
  ctc_loss_val = 0
  ctc_loss_val2 = 0
  box_loss_val = 0
  good_all = 0
  gt_all = 0
  
  
  for step in range(step_start, opts.max_iters):
    
    # batch
    images, image_fns, score_maps, geo_maps, training_masks, gtso, lbso, gt_idxs = next(data_generator)
    im_data = net_utils.np_to_variable(images, is_cuda=opts.cuda).permute(0, 3, 1, 2)
    start = timeit.timeit()
    try:
      seg_pred, roi_pred, angle_pred, features = net(im_data)
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      continue
    end = timeit.timeit()
    
    # backward

    smaps_var = net_utils.np_to_variable(score_maps, is_cuda=opts.cuda)
    training_mask_var = net_utils.np_to_variable(training_masks, is_cuda=opts.cuda)
    angle_gt = net_utils.np_to_variable(geo_maps[:, :, :, 4], is_cuda=opts.cuda)
    geo_gt = net_utils.np_to_variable(geo_maps[:, :, :, [0, 1, 2, 3]], is_cuda=opts.cuda)
    
    try:
      loss = net.loss(seg_pred, smaps_var, training_mask_var, angle_pred, angle_gt, roi_pred, geo_gt)
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      continue
      
    bbox_loss += net.box_loss_value.data.cpu().numpy() 
    seg_loss += net.segm_loss_value.data.cpu().numpy()
    angle_loss += net.angle_loss_value.data.cpu().numpy()  
      
    train_loss += loss.data.cpu().numpy()
    optimizer.zero_grad()
       
    try:
      
      if step > 10000 or True: #this is just extra augumentation step ... in early stage just slows down training
        ctcl, gt_b_good, gt_b_all = process_boxes(images, im_data, seg_pred[0], roi_pred[0], angle_pred[0], score_maps, gt_idxs, gtso, lbso, features, net, ctc_loss, opts, debug=opts.debug)
        ctc_loss_val += ctcl.data.cpu().numpy()[0]
        loss = loss + ctcl
        gt_all += gt_b_all
        good_all += gt_b_good 
      
      imageso, labels, label_length = next(dg_ocr)
      im_data_ocr = net_utils.np_to_variable(imageso, is_cuda=opts.cuda).permute(0, 3, 1, 2)
      features = net.forward_features(im_data_ocr)
      labels_pred = net.forward_ocr(features)
    
      probs_sizes =  torch.IntTensor( [(labels_pred.permute(2,0,1).size()[0])] * (labels_pred.permute(2,0,1).size()[1]) )
      label_sizes = torch.IntTensor( torch.from_numpy(np.array(label_length)).int() )
      labels = torch.IntTensor( torch.from_numpy(np.array(labels)).int() )
      loss_ocr = ctc_loss(labels_pred.permute(2,0,1), labels, probs_sizes, label_sizes) / im_data_ocr.size(0) * 0.5
      
      loss_ocr.backward()
      ctc_loss_val2 += loss_ocr.item()
      loss.backward()
      
      optimizer.step()
    except:
      import sys, traceback
      traceback.print_exc(file=sys.stdout)
      pass
    cnt += 1
    if step % disp_interval == 0:
      
      if opts.debug:
        
        segm = seg_pred[0].data.cpu()[0].numpy()
        segm = segm.squeeze(0)
        cv2.imshow('segm_map', segm)
        
        segm_res = cv2.resize(score_maps[0], (images.shape[2], images.shape[1]))
        mask = np.argwhere(segm_res > 0)
        
        x_data = im_data.data.cpu().numpy()[0]
        x_data = x_data.swapaxes(0, 2)
        x_data = x_data.swapaxes(0, 1)
        
        x_data += 1
        x_data *= 128
        x_data = np.asarray(x_data, dtype=np.uint8)
        x_data = x_data[:, :, ::-1]
        
        im_show = x_data
        try:
          im_show[mask[:, 0], mask[:, 1], 1] = 255 
          im_show[mask[:, 0], mask[:, 1], 0] = 0 
          im_show[mask[:, 0], mask[:, 1], 2] = 0
        except:
          pass
        
        cv2.imshow('img0', im_show) 
        cv2.imshow('score_maps', score_maps[0] * 255)
        cv2.imshow('train_mask', training_masks[0] * 255)
        cv2.waitKey(10)
      
      train_loss /= cnt
      bbox_loss /= cnt
      seg_loss /= cnt
      angle_loss /= cnt
      ctc_loss_val /= cnt
      ctc_loss_val2 /= cnt
      box_loss_val /= cnt
      try:
        print('epoch %d[%d], loss: %.3f, bbox_loss: %.3f, seg_loss: %.3f, ang_loss: %.3f, ctc_loss: %.3f, rec: %.5f lv2 %.3f' % (
          step / batch_per_epoch, step, train_loss, bbox_loss, seg_loss, angle_loss, ctc_loss_val, good_all / max(1, gt_all), ctc_loss_val2))
      except:
        import sys, traceback
        traceback.print_exc(file=sys.stdout)
        pass
      

      train_loss = 0
      bbox_loss, seg_loss, angle_loss = 0., 0., 0.
      cnt = 0
      ctc_loss_val = 0
      good_all = 0
      gt_all = 0
      box_loss_val = 0
      
    #if step % valid_interval == 0:
    #  validate(opts.valid_list, net)
    if step > step_start and (step % batch_per_epoch == 0):
      save_name = os.path.join(opts.save_path, '{}_{}.h5'.format(model_name, step))
      state = {'step': step,
               'learning_rate': learning_rate,
              'state_dict': net.state_dict(),
              'optimizer': optimizer.state_dict()}
      torch.save(state, save_name)
      print('save model: {}'.format(save_name))


import argparse

if __name__ == '__main__': 
  
  parser = argparse.ArgumentParser()
  parser.add_argument('-train_list', default='sample_train_data/MLT/trainMLT.txt')
  parser.add_argument('-ocr_feed_list', default='sample_train_data/MLT_CROPS/gt.txt')
  parser.add_argument('-save_path', default='backup')
  parser.add_argument('-model', default='e2e-mlt.h5')
  parser.add_argument('-debug', type=int, default=1)
  parser.add_argument('-batch_size', type=int, default=2)
  parser.add_argument('-ocr_batch_size', type=int, default=1)
  parser.add_argument('-num_readers', type=int, default=1)
  parser.add_argument('-cuda', type=bool, default=True)
  parser.add_argument('-input_size', type=int, default=256)
  parser.add_argument('-geo_type', type=int, default=0)
  parser.add_argument('-base_lr', type=float, default=0.0001)
  parser.add_argument('-max_iters', type=int, default=300000)
  
  args = parser.parse_args()  
  main(args)
  
