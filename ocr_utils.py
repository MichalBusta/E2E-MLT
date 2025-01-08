'''
Created on Oct 25, 2018

@author: Michal.Busta at gmail.com
'''

import math
import numpy as np

import torch
import torch.nn.functional as F

def print_seq_ext(wf, codec):
  prev = 0
  word = ''
  current_word = ''
  start_pos = 0
  end_pos = 0
  dec_splits = []
  splits = []
  hasLetter = False
  for cx in range(0, wf.shape[0]):
    c = wf[cx]
    if prev == c:
      if c > 2:
          end_pos = cx
      continue
    if c > 3 and c < (len(codec)+4):
      ordv = codec[c - 4]
      char = ordv
      if char == ' ' or char == '.' or char == ',' or char == ':':
        if hasLetter:
          if char != ' ':
            current_word += char
          splits.append(current_word)
          dec_splits.append(cx + 1)
          word += char
          current_word = ''
      else:
        hasLetter = True
        word += char
        current_word += char
      end_pos = cx
    elif c > 0:
      if hasLetter:
        dec_splits.append(cx + 1)
        word += ' '
        end_pos = cx
        splits.append(current_word)
        current_word = ''
      
    
    if len(word) == 0:
      start_pos = cx
    prev = c    
    
  dec_splits.append(end_pos + 1)
  conf2 = [start_pos, end_pos + 1]
  
  return word.strip(), np.array([conf2]), np.array([dec_splits]), splits

def ocr_image(net, codec, im_data, detection):
  
  boxo = detection
  boxr = boxo[0:8].reshape(-1, 2)
  
  center = (boxr[0, :] + boxr[1, :] + boxr[2, :] + boxr[3, :]) / 4
  
  dw = boxr[2, :] - boxr[1, :]
  dh =  boxr[1, :] - boxr[0, :]

  w = math.sqrt(dw[0] * dw[0] + dw[1] * dw[1])
  h = math.sqrt(dh[0] * dh[0] + dh[1] * dh[1])
  
  input_W = im_data.size(3)
  input_H = im_data.size(2)
  target_h = 40  
    
  scale = target_h / max(1, h) 
  target_gw = int(w * scale) + target_h 
  target_gw = max(2, target_gw // 32) * 32      
    
  xc = center[0] 
  yc = center[1] 
  w2 = w 
  h2 = h 
  
  angle = math.atan2((boxr[2][1] - boxr[1][1]), boxr[2][0] - boxr[1][0])
  
  #show pooled image in image layer

  scalex = (w2 + h2) / input_W * 1.2
  scaley = h2 / input_H * 1.3

  th11 =  scalex * math.cos(angle)
  th12 = -math.sin(angle) * scaley
  th13 =  (2 * xc - input_W - 1) / (input_W - 1) #* torch.cos(angle_var) - (2 * yc - input_H - 1) / (input_H - 1) * torch.sin(angle_var)
  
  th21 = math.sin(angle) * scalex 
  th22 =  scaley * math.cos(angle)  
  th23 =  (2 * yc - input_H - 1) / (input_H - 1) #* torch.cos(angle_var) + (2 * xc - input_W - 1) / (input_W - 1) * torch.sin(angle_var)
            
  t = np.asarray([th11, th12, th13, th21, th22, th23], dtype=float)
  t = torch.from_numpy(t).type(torch.FloatTensor)
  t = t.cuda()
  theta = t.view(-1, 2, 3)
  
  grid = F.affine_grid(theta, torch.Size((1, 3, int(target_h), int(target_gw))))
  
  
  x = F.grid_sample(im_data, grid)
  
  features = net.forward_features(x)
  labels_pred = net.forward_ocr(features)
  
  ctc_f = labels_pred.data.cpu().numpy()
  ctc_f = ctc_f.swapaxes(1, 2)

  labels = ctc_f.argmax(2)
  
  ind = np.unravel_index(labels, ctc_f.shape)
  conf = np.mean( np.exp(ctc_f[ind]) )
  
  det_text, conf2, dec_s, splits = print_seq_ext(labels[0, :], codec)  
  
  return det_text, conf2, dec_s