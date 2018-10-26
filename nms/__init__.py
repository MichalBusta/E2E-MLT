import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile nms: {}'.format(BASE_DIR))

  
def do_nms(segm_map, geo_map, angle_pred, poly_map, thres=0.3, thres2=0.2, segm_thresh=0.5):
    precision=10000
    from .adaptor import do_nms as nms_impl
    ret = np.array(nms_impl(segm_map, geo_map, angle_pred, poly_map, thres, thres2, segm_thresh), dtype='float32')
    if len(ret) > 0:
      ret[:,:8] /= precision
    return ret
  
  
def get_boxes(iou_map, rbox, angle_pred, segm_thresh=0.5):
  
  angle_pred = angle_pred.swapaxes(0, 1)
  angle_pred = angle_pred.swapaxes(1, 2)
  
  poly_map = np.zeros((iou_map.shape[0], iou_map.shape[1]), dtype = np.int32)
  poly_map.fill(-1);
  
  boxes = do_nms( iou_map, rbox, angle_pred, poly_map, 0.4, 0.2, segm_thresh)
  return boxes   

