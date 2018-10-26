'''
Created on Aug 31, 2017

@author: Michal.Busta at gmail.com
'''
import numpy as np
import torch
from torch.autograd import Variable

def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
  v = torch.from_numpy(x).type(dtype)
  if is_cuda:
      v = v.cuda()
  return v

def load_net(fname, net, optimizer=None):
  sp = torch.load(fname) 
  step = sp['step']
  try:
    learning_rate = sp['learning_rate']
  except:
    import traceback
    traceback.print_exc()
    learning_rate = 0.001
  opt_state = sp['optimizer']
  sp = sp['state_dict']
  for k, v in net.state_dict().items():
    try:
      param = sp[k]
      v.copy_(param)
    except:
      import traceback
      traceback.print_exc()
  
  if optimizer is not None:  
    try:
      optimizer.load_state_dict(opt_state)
    except:
      import traceback
      traceback.print_exc()
  
  print(fname)
  return step, learning_rate 
  