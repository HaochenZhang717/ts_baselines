import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.multiprocessing
import logging
import torch.nn.functional as F

from utils.loggers import NeptuneLogger, PrintLogger, CompositeLogger
from models.model import MultimodalImagenTime
from models.sampler import MultiModalDiffusionProcess
from utils.utils import save_checkpoint, restore_state, create_model_name_and_dir, print_model_params, \
    log_config_and_tags, get_x_and_mask
from utils.utils_data import gen_dataloader
from utils.utils_args import parse_args_cond
import wandb
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    # model name and directory
    name = create_model_name_and_dir(args)

    # log args
    logging.info(args)
    # set-up neptune logger. switch to your desired logger
    with CompositeLogger([NeptuneLogger()]) if args.neptune \
            else PrintLogger() as logger:

        # log config and tags
        log_config_and_tags(args, logger, name)

        # --- set-up data and device ---
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_loader, test_loader = gen_dataloader(args)
        logging.info(args.dataset + ' dataset is ready.')

        model = MultimodalImagenTime(args=args, device=args.device).to(args.device)

        # optimizer

        state = dict(model=model, epoch=800)

        # restore checkpoint
        if args.resume:
            ema_model = model.model_ema if args.ema else None # load ema model if available
            init_epoch = restore_state(args, state, ema_model=ema_model)

        # print model parameters
        print_model_params(logger, model)

        # --- evaluation loop ---
        mse = 0
        mae = 0
        model.eval()
        with torch.no_grad():
            with model.ema_scope():
                process = MultiModalDiffusionProcess(
                    args, model.net,
                    (args.input_channels, args.img_resolution, args.img_resolution)
                )
                for idx, (x_ts, mask_ts, context_ts) in enumerate(test_loader, 1):
                    x_ts = x_ts.to(args.device)
                    mask_ts = mask_ts.to(args.device)
                    context_ts = context_ts.to(args.device)

                    # transform to image
                    x_ts_img = model.ts_to_img(x_ts)
                    mask_ts_img = model.ts_to_img(mask_ts, pad_val=1)

                    # sample from the model
                    # and impute, both interpolation and extrapolation are similar just the mask is different
                    x_img_sampled = process.interpolate(x_ts_img, mask_ts_img, torch.randn_like(x_ts_img), context_ts).to(x_ts_img.device)
                    x_ts_sampled = model.img_to_ts(x_img_sampled)

                    # task evaluation
                    mse_mean = F.mse_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                    mae_mean = F.l1_loss(x_ts[mask_ts == 0].to(x_ts.device), x_ts_sampled[mask_ts == 0])
                    mse += mse_mean.item()
                    mae += mae_mean.item()

                    to_save = {
                        'orig': x_ts.cpu(),
                        'predicted': x_ts_sampled.cpu(),
                    }

                    torch.save(to_save, f"{args.log_dir}/samples_epoch{state['epoch']}.pt")
                    break


if __name__ == '__main__':
    args = parse_args_cond()  # parse unconditional generation specific args
    torch.random.manual_seed(args.seed)
    np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main(args)



