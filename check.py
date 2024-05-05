import torch

x = torch.load("osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth", map_location=lambda storage, loc: storage)

print(x)