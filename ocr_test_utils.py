'''
Created on Nov 3, 2017

@author: Michal.Busta at gmail.com
'''

import os

import numpy as np
import cv2

import net_utils

import unicodedata as ud

from Levenshtein import distance
import pandas as pd

from ocr_utils import print_seq_ext

buckets = []
for i in range(1, 100):
  buckets.append(8  + 8 * i)


def test(net, codec, args,  list_file = '/home/busta/data/icdar_ch8_validation/ocr_valid.txt', norm_height=32, max_samples=1000000):
  
  
  codec_rev = {}
  index = 4
  for i in range(0, len(codec)):
    codec_rev[codec[i]] = index
    index += 1
  
  
  net = net.eval()
  #list_file = '/mnt/textspotter/tmp/90kDICT32px/train_list.txt'
  #list_file = '/home/busta/data/Challenge2_Test_Task3_Images/gt.txt'
  #list_file = '/home/busta/data/90kDICT32px/train_icdar_ch8.txt'
  fout = open('/tmp/ch8_valid.txt', 'w')
  fout_ocr = open('/tmp/ocr_valid.txt', 'w')
  
  
  dir_name = os.path.dirname(list_file)
  images = []
  with open(list_file, "r") as ins:
      for line in ins:
        images.append(line.strip())
        #if len(images) > 1000:
        #  break
  
  scripts = ['', 'DIGIT', 'LATIN', 'ARABIC', 'BENGALI', 'HANGUL', 'CJK', 'HIRAGANA', 'KATAKANA']
  
  conf_matrix = np.zeros((len(scripts), len(scripts)), dtype = np.int)
  
  gt_script = {}
  ed_script = {}
  correct_ed1_script = {}
  correct_script = {}
  count_script = {}
  for scr in scripts:
    gt_script[scr] = 0
    ed_script[scr] = 0
    correct_script[scr] = 0
    correct_ed1_script[scr] = 0
    count_script[scr] = 0
  
  it = 0
  it2 = 0
  correct = 0
  correct_ed1 = 0
  ted = 0
  gt_all = 0
  images_count = 0
  bad_words = []
  
  for img in images:
      
    imageNo = it2
    #imageNo = random.randint(0, len(images) - 1)
    if imageNo >= len(images) or imageNo > max_samples:
      break

    
    image_name = img
    
    spl = image_name.split(",")
    delim = ","
    if len(spl) == 1:
      spl = image_name.split(" ")
      delim = " "
    image_name = spl[0].strip()
    gt_txt = ''
    if len(spl) > 1:
      gt_txt = spl[1].strip()
      if len(spl) > 2:
        gt_txt += delim + spl[2]
      
      if len(gt_txt) > 1 and gt_txt[0] == '"' and gt_txt[-1] == '"':
        gt_txt = gt_txt[1:len(gt_txt) - 1]
    
    it2 += 1
    if len(gt_txt) == 0:
      print(images[imageNo])
      continue
            
    if image_name[-1] == ',':
      image_name = image_name[0:-1]
    
    img_nameo = image_name
    image_name = '{0}/{1}'.format(dir_name, image_name)
    img = cv2.imread(image_name)
    
    if img is None:
      print(image_name)
      continue
        
    scale = norm_height / float(img.shape[0])
    width = int(img.shape[1] * scale)
    width = max(8, int(round(width / 4)) * 4) 
    
    scaled = cv2.resize(img, (int(width), norm_height))  
    #scaled = scaled[:, :, ::-1]
    scaled = np.expand_dims(scaled, axis=0)
    
    scaled = np.asarray(scaled, dtype=np.float)
    scaled /= 128
    scaled -= 1
    
    try:
      scaled_var = net_utils.np_to_variable(scaled, is_cuda=args.cuda).permute(0, 3, 1, 2)
      x = net.forward_features(scaled_var)
      ctc_f = net.forward_ocr(x )
      ctc_f = ctc_f.data.cpu().numpy()
      ctc_f = ctc_f.swapaxes(1, 2)
      
      labels = ctc_f.argmax(2)
      det_text, conf, dec_s, _ = print_seq_ext(labels[0, :], codec)
    except:
      print('bad image')
      det_text = ''
        
    det_text = det_text.strip()
    gt_txt = gt_txt.strip()  
    
    try:
      if 'ARABIC' in ud.name(gt_txt[0]):
        #gt_txt = gt_txt[::-1]
        det_text = det_text[::-1]
    except:
      continue
    
    it += 1 
    
    scr_count =  [0, 0, 0,  0, 0, 0, 0, 0,  0]
    scr_count = np.array(scr_count)
    
    for c_char in gt_txt:
      assigned = False 
      for idx, scr in enumerate(scripts):
        if idx == 0:
          continue
        symbol_name = ud.name(c_char)
        if scr in symbol_name:
          scr_count[idx] += 1
          assigned = True
          break
      if not assigned:
        scr_count[0] += 1
        
        
    maximum_indices = np.where(scr_count==np.max(scr_count))      
    script = scripts[maximum_indices[0][0]]
    
    det_count =  [0, 0, 0,  0, 0, 0, 0, 0,  0]
    det_count = np.array(det_count)    
    for c_char in det_text:
      assigned = False 
      for idx, scr in enumerate(scripts):
        if idx == 0:
          continue
        try:
          symbol_name = ud.name(c_char)
          if scr in symbol_name:
            det_count[idx] += 1
            assigned = True
            break
        except:
          pass
      if not assigned:
        det_count[0] += 1
    
    maximum_indices_det = np.where(det_count==np.max(det_count))      
    script_det = scripts[maximum_indices_det[0][0]]    
          
    
    conf_matrix[maximum_indices[0][0], maximum_indices_det[0][0]] += 1
    
    edit_dist = distance(det_text.lower(), gt_txt.lower()) 
    ted += edit_dist
    gt_all += len(gt_txt)
    
    gt_script[script] += len(gt_txt)
    ed_script[script] += edit_dist
    images_count += 1
    
    fout_ocr.write('{0}, "{1}"\n'.format(os.path.basename(image_name), det_text.strip()))
    
    if det_text.lower() == gt_txt.lower():
      correct += 1
      correct_ed1 += 1
      correct_script[script] += 1
      correct_ed1_script[script] += 1      
    else:
      if edit_dist == 1:
        correct_ed1 += 1
        correct_ed1_script[script] += 1  
      image_prev = "<img src=\"{0}\" height=\"32\" />".format(img_nameo)
      bad_words.append((gt_txt, det_text, edit_dist, image_prev, img_nameo))
      print('{0} - {1} / {2:.2f} - {3:.2f}'.format(det_text, gt_txt, correct / float(it), ted / 3.0 )) 
    
    count_script[script] += 1
    fout.write('{0}|{1}|{2}|{3}\n'.format(os.path.basename(image_name), gt_txt, det_text, edit_dist))  
    
  print('Test accuracy: {0:.3f}, {1:.2f}, {2:.3f}'.format(correct / float(images_count), ted / 3.0, ted / float(gt_all) ))  
  
  
  itf = open("per_script_accuracy.csv", "w")
  itf.write('Script & Accuracy & Edit Distance & ed1 & Ch instances & Im Instances \\\\\n')
  for scr in scripts:
    correct_scr = correct_script[scr]
    correct_scr_ed1 = correct_ed1_script[scr]
    all = count_script[scr]
    ted_scr = ed_script[scr]
    gt_all_scr = gt_script[scr]
    print(' Script:{3} Acc : {0:.3f}, {1:.2f}, {2:.3f}, {4}'.format(correct_scr / float(max(all, 1)), ted_scr / 3.0, ted_scr / float(max(gt_all_scr, 1)), scr, gt_all_scr ))  
    
    itf.write('{0} & {1:.3f} & {5:.3f} &  {2:.3f} & {3} & {4} \\\\\n'.format(
      scr.title(), correct_scr / float(max(all, 1)), ted_scr / float(max(gt_all_scr, 1)), gt_all_scr, all, correct_scr_ed1 / float(max(all, 1))))  
  
  itf.write('{0} & {1:.3f} & {5:.3f} &  {2:.3f} & {3} & {4} \\\\\n'.format(
      'Total', correct / float(max(images_count, 1)), ted / float(max(gt_all, 1)), gt_all, images_count, correct_ed1 / float(max(images_count, 1)) ))      
  itf.close()    
      
  print(conf_matrix)
  np.savetxt("conf_matrix.csv", conf_matrix, delimiter=' & ', fmt='%d', newline=' \\\\\n')  
  
  itf = open("conf_matrix_out.csv", "w")
  itf.write( ' & ' )
  delim = ""
  for scr in scripts:
    itf.write( delim )
    itf.write( scr.title() )
    delim = " & "
  itf.write( '\\\\\n' ) 
  
  script_no = 0
  with open("conf_matrix.csv", "r") as ins:
    for line in ins:
      line = scripts[script_no].title() + " & " + line
      itf.write(line)
      script_no +=1 
      if script_no >= len(scripts):
        break  
   
  fout.close() 
  fout_ocr.close()
  net.train()
  
  pd.options.display.max_rows = 9999
  #pd.options.display.max_cols = 9999
  
  if len(bad_words) > 0:   
    wworst =  sorted(bad_words, key=lambda x: x[2])
       
    ww = np.asarray(wworst, np.object)
    ww = ww[0:1500, :]
    df2 = pd.DataFrame({ 'gt' : ww[:, 0], 'pred' : ww[:, 1], 'ed' : ww[:, 2], 'image': ww[:, 3]})
    
    html = df2.to_html(escape=False)
    report = open('{0}/ocr_bad.html'.format(dir_name), 'w')
    report.write(html)
    report.close()  
    
    wworst =  sorted(bad_words, key=lambda x: x[2], reverse=True)
       
    ww = np.asarray(wworst, np.object)
    ww = ww[0:1500, :]
    df2 = pd.DataFrame({ 'gt' : ww[:, 0], 'pred' : ww[:, 1], 'ed' : ww[:, 2], 'image': ww[:, 3]})
    
    html = df2.to_html(escape=False)
    report = open('{0}/ocr_not_sobad.html'.format(dir_name), 'w')
    report.write(html)
    report.close()      
  
  return correct / float(images_count), ted

  
