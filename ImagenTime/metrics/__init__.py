from metrics.metrics_long_range import compute_all_metrics, setup_optimizer
import numpy as np
import torch

from metrics.metrics_long_range import FIDVAE, extract_embeddings, compute_fid_given_embeds


def evaluate_model_uncond(real_sig,gen_sig,args):
    """
    Args:
        real_sig: real signal
        gen_sig: generated signal
        args: args
    Returns:
        marginal score if long-term dataset, discrimin
    this function evaluates the model based on the dataset used:
    for short-term datasets(eg. sine, stock) it uses discriminative_torch.py and predictive_torch.py
    for long-term datasets(eg. fred_md) it uses metrics_long_range.py


    """

    if args.dataset in ['stock','sine','mujoco','energy']:
        # proceed with short term evaluation
        metric_iteration = 10
        from metrics.discriminative_torch import discriminative_score_metrics
        ## for deterministic results
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        disc_res = []
        for _ in range(metric_iteration):
            dsc = discriminative_score_metrics(real_sig, gen_sig, args)
            disc_res.append(dsc)
        disc_mean, disc_std = np.round(np.mean(disc_res), 4), np.round(np.std(disc_res), 4)
        from metrics.predictive_metrics import predictive_score_metrics
        predictive_score = list()
        for _ in range(metric_iteration):
            temp_pred = predictive_score_metrics(real_sig, gen_sig)
            predictive_score.append(temp_pred)
        pred_mean, pred_std = np.round(np.mean(predictive_score), 4), np.round(np.std(predictive_score), 4)
        return {'disc_mean':disc_mean,'disc_std':disc_std,'pred_mean':pred_mean,'pred_std':pred_std}

    else:
        # proceed with long term evaluation
        # conversion to meet benchmark requirements:
        real_sig,gen_sig = torch.Tensor(real_sig).float(),torch.Tensor(gen_sig).float()
        # scores = compute_all_metrics(real_sig, gen_sig, setup_optimizer,
        #                              torch.nn.Sigmoid() if args.dataset == 'temperature_rain' else torch.nn.Identity(),
        #                              args.device)

        scores = compute_fid(real_sig, gen_sig)
        return scores


def compute_fid(x_real, gens, ckpt_path):
    x_real = x_real.permute(0, 2, 1)
    gens = gens.permute(0, 2, 1)
    assert ckpt_path is not None
    _, C, T = x_real.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FIDVAE(
        input_dim=C,
        output_dim=C,
        seq_len=T,
        hidden_size=128,
        num_layers=2,
        num_heads=8,
        latent_dim=64,
    ).to(device).eval()

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    real_embeddings = extract_embeddings(model, x_real, device)
    gen_embeddings = extract_embeddings(model, gens, device)
    fid = compute_fid_given_embeds(real_embeddings, gen_embeddings)

    return {'fid': fid}
