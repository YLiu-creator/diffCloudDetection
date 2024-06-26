# import os
# import torch
# # select the GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, "1"))

import os
import argparse
import sys
import random

from datasets import L8_Biome_cls_WEAK, L8_Biome_cls_FULL

sys.path.append(".")
import numpy as np
import torch
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


from guided_diffusion import dist_util, logger
from datasets.gf1_dataset import GF1_cls_FULL, GF1_cls_WEAK
import multiprocessing

from collections import OrderedDict

seed=10
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def create_argparser():
    defaults = dict(
        data_name = 'GF1',
        data_dir="/home/FAKEDATA/GF1_datasets/datasets_321/data",

        clip_denoised=True,
        num_samples=2,
        batch_size=2,
        use_ddim=True,
        diffusion_steps=20,
        dpm_solver=False,

        model_path="./results_GF1_multiBatch_times2/savedmodel030000.pt",         #path to pretrain model
        num_ensemble=3,      #number of samples in the ensemble
        gpu_dev = "5",
        out_dir='/home/liuyang/CloudDiffusion/MedSegDiff_v1/GF1_times2_3w/',
        multi_gpu = False, #"0,1,2"
        debug = True
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":

    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir=args.out_dir)

    args.diffusion_steps = 20
    args.use_ddim = True

    os.makedirs(args.out_dir, exist_ok=True)

    if args.data_name == 'GF1':
        print('Traing GF-1 WFV dataset ...')
        print('GF1 image_size set to: ' + str((args.image_size, args.image_size)))
        train_dst = GF1_cls_FULL(root=args.data_dir, image_set='test')
        args.in_ch = 5

    datal = torch.utils.data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=False)
    data = iter(datal)
    print("The number of the GF1 dataset samples:" + str(len(data)))

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.to(dist_util.dev())

    print("Model restored from %s" % args.model_path)

    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()

    from tqdm import tqdm

    for _ in tqdm(range(len(data))):

        b, cond, name = next(data)

        c = torch.randn_like(b[:, :1, ...])
        img = torch.cat((b, c), dim=1)  # add a noise channel$
        if args.data_name == 'GF1':
            slice_ID = name[0].split("/")[-1].split('.')[0]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()

            if not args.use_ddim:
                print('No using DDIM')
                sample_fn = diffusion.p_sample_loop_known

                # sample is [1,1,320,320], x_noisy is [1,5,320,320], org is [1,5,320,320]
                # cal is [1,1,320,320], cal_out is [1,1,320,320]
                sample, x_noisy, org, cal, cal_out, diff_oup = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img,
                    step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )
            else:
                print('Test using DDIM')
                sample_fn = diffusion.ddim_sample_loop_known
                # img is [1,5,320,320]
                # sample is [1,1,320,320], x_noisy is [1,5,320,320], org is [1,5,320,320]
                # cal is [1,1,320,320], cal_out is [1,1,320,320]
                sample, x_noisy, org, cal, cal_out = sample_fn(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    img,
                    step=args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                )

            end.record()
            torch.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample

            # sample = sample.detach().cpu().numpy()
            # x_noisy = x_noisy.detach().cpu().numpy()
            # org  = org.detach().cpu().numpy()
            # cal = cal.detach().cpu().numpy()
            cal_out = cal_out.detach().cpu().numpy()

            for b_th in range(args.batch_size):
                save_name = name[b_th].split('/')[-1].replace('.npy', '')

                # np.save(args.out_dir + save_name+'_smp'+str(i)+'.npy', sample)
                # np.save(args.out_dir + save_name+'_xno'+str(i)+'.npy', x_noisy)
                # np.save(args.out_dir + save_name+'_ori'+str(i)+'.npy', org)
                # np.save(args.out_dir + save_name + '_cal' + str(i) + '.npy', cal[b_th, ...])
                np.save(args.out_dir + save_name + '_cop' + str(i) + '.npy', cal_out[b_th, ...])
