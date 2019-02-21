# pytorch-soundnet

A PyTorch port of [SoundNet](https://github.com/cvondrick/soundnet).


Requirements
============

 * PyTorch >=1.0


Usage
=====


```python
import torch
from soundnet import SoundNet

model = SoundNet()
model.load_state_dict(torch.load('soundnet8_final.pth'))
```


References
==========

If you use SoundNet in your research, please cite the paper:

    SoundNet: Learning Sound Representations from Unlabeled Video 
    Yusuf Aytar, Carl Vondrick, Antonio Torralba
    NIPS 2016

