import os
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from models.experimental import attempt_load

def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(
        ), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (
                batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)

    return torch.device('cuda:0' if cuda else 'cpu')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_weights', type=str,
                        default='weights/last.pt', help='initial weights path')
    parser.add_argument('--out_weights', type=str,
                        default='quaed_model.pt', help='output weights path')
    parser.add_argument('--device', type=str, default='0', help='device')
    opt = parser.parse_args()

    device = select_device(opt.device)
    # Load model
    model = attempt_load(opt.in_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    print(model)
    model.half()

    torch.save(model, opt.out_weights)
    print('done.')

    print('-[INFO] before: {} kb, after: {} kb'.format(
        os.path.getsize(opt.in_weights), os.path.getsize(opt.out_weights)))
