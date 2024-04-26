
import sys
import argparse
sys.path.append("../")
sys.path.append("./")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms

from datasets.gf1_dataset import GF1_WEAK
from datasets.l8_dataset import L8Biome_WEAK
from datasets.s2_dataset import S2CMC_WEAK


def create_argparser():
    defaults = dict(
        data_name='GF1',
        data_dir="./GF1_datasets/data",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=5000,
        resume_checkpoint=None, #"/results/pretrainedmodel.pt"
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = "4",
        multi_gpu = None, #"0,1,2"
        out_dir='./results_MedSegDiff/'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")

    if args.data_name == 'GF1':
        print('Traing GF-1 WFV dataset ...')
        print('GF1 image_size set to: ' + str((args.image_size, args.image_size)))
        train_dst = GF1_WEAK(root=args.data_dir, image_set='train')
        args.in_ch = 5

    elif args.data_name == 'GF1':
        print('Traing Landsat-8 Biome dataset ...')
        print('GF1 image_size set to: ' + str((args.image_size, args.image_size)))
        train_dst = L8Biome_WEAK(root=args.data_dir, image_set='train')
        args.in_ch = 8

    elif args.data_name == 'Sentinel':
        print('Traing Sentinel-2 CMC dataset ...')
        print('GF1 image_size set to: ' + str((args.image_size, args.image_size)))
        train_dst = S2CMC_WEAK(root=args.data_dir, image_set='train')
        args.in_ch = 13

    datal = th.utils.data.DataLoader(train_dst,batch_size=args.batch_size,shuffle=True)
    data = iter(datal)
    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        model.to(device = th.device('cuda', int(args.gpu_dev)))
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()





if __name__ == "__main__":
    main()
