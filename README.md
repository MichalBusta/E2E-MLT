# E2E-MLT
 E2E-MLT - an Unconstrained End-to-End Method for Multi-Language Scene Text
 
code base for:  https://arxiv.org/abs/1801.09919

## Requirements
  - python3.x with opencv-python, pytorch 0.4, warp-ctc (https://github.com/SeanNaren/warp-ctc/)  
  
  
## Data 

 - [ICDAR MLT Dataset](http://rrc.cvc.uab.es/?ch=8&com=introduction)
 - ICDAR 2015 Dataset (http://rrc.cvc.uab.es/?ch=4&com=introduction)
 - RCTW-17 (http://mclab.eic.hust.edu.cn/icdar2017chinese/)
 - Synthetic MLT Data ([Arabic](http://ptak.felk.cvut.cz/public_datasets/SyntText/Arabic.zip), [Bangla](http://ptak.felk.cvut.cz/public_datasets/SyntText/Bangla.zip), [Chinese](http://ptak.felk.cvut.cz/public_datasets/SyntText/Chinese.zip), [Japanese](http://ptak.felk.cvut.cz/public_datasets/SyntText/Japanese.zip), [Korean](http://ptak.felk.cvut.cz/public_datasets/SyntText/Korean.zip)  )

![MLT SynthSet](images/synth.png)

What we have found useful:
 - for generating Arabic Scene Text: https://github.com/mpcabd/python-arabic-reshaper 
 - for generating Bangla Scene Text: PyQt4


TODO
 - update arxiv with current paper status
 - finalize code 
