import torch.nn as nn
from MORAN_v2.models.morn import MORN
from MORAN_v2.models.asrn_res import ASRN

class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
    	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)

    def rectify(self, x, test=False):
        return self.MORN(x, test)

    def forward(self, x_rectified, length, text, text_rev, test=False):
        return self.ASRN(x_rectified, length, text, text_rev, test)
