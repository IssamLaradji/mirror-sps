import torch
import pandas as pd
import os
import pylab as plt
import exp_configs
import time
import numpy as np
from src import models
from src import datasets
from src import optimizers

from haven import haven_wizard as hw
from haven import haven_utils as hu
import argparse

from torch.utils.data import DataLoader


def trainval(exp_dict, savedir, args):
    # Set seed and device
    # ===================
    seed = 42 + exp_dict['runs']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        device = 'cuda'
        torch.cuda.manual_seed_all(seed)
        assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    else:
        device = 'cpu'

    print('Running on device: %s' % device)

    # Load Datasets
    # ==================
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split='train',
                                     datadir=args.datadir,
                                     exp_dict=exp_dict)

    train_loader = DataLoader(train_set,
                              drop_last=True,
                              shuffle=True,
                              sampler=None,
                              batch_size=exp_dict["batch_size"])

    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                   split='val',
                                   datadir=args.datadir,
                                   exp_dict=exp_dict)

    # Load Model
    # ==================
    model = models.get_model(train_loader, exp_dict, device=device)
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    # Set Optimizer
    opt = optimizers.get_optimizer(opt=exp_dict["opt"],
                                       params=model.parameters(),
                                       train_loader=train_loader,                                
                                       exp_dict=exp_dict)
    model.set_opt(opt)

    if os.path.exists(score_list_path):
        # resume experiment
        score_list = hu.load_pkl(score_list_path)
        model.set_state_dict(torch.load(model_path))
        s_epoch = score_list[-1]["epoch"] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0
        
    # Train and Val
    # ==============
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        # Set seed
        seed = epoch + exp_dict.get('runs', 0)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Validate one epoch
        train_loss_dict = model.val_on_dataset(train_set, metric=exp_dict["loss_func"], name='loss')
        val_acc_dict = model.val_on_dataset(val_set, metric=exp_dict.get("score_func", exp_dict.get('acc_func')), name='score')


        # Train one epoch
        s_time = time.time()
        model.train_on_loader(train_loader)
        e_time = time.time()

        
        # Record metrics
        score_dict = {"epoch": epoch}
        score_dict.update(train_loss_dict)
        score_dict.update(val_acc_dict)
        score_dict["step_size"] = model.opt.state.get("step_size", {})
        score_dict["step_size_avg"] = model.opt.state.get("step_size_avg", {})
        score_dict["n_forwards"] = model.opt.state.get("n_forwards", {})
        score_dict["n_backwards"] = model.opt.state.get("n_backwards", {})
        score_dict["grad_norm"] = model.opt.state.get("grad_norm", {})
        score_dict["train_epoch_time"] = e_time - s_time
        score_dict.update(model.opt.state["gv_stats"])

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.get_state_dict())
        print("Saved: %s" % savedir)


   
if __name__ == "__main__":
    import exp_configs
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help='Define which exp groups to run.')
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument('-d', '--datadir', default=None,
                        help='Define the dataset directory.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("-c", "--cuda", default=1, type=int)
    parser.add_argument("-j", "--job_scheduler", default=None, type=str)
    parser.add_argument("-p", "--python_binary_path", default='python')
    args, others = parser.parse_known_args()

    if os.path.exists('job_configs.py'):
        import job_configs
        job_config = job_configs.JOB_CONFIG
    else:
        job_config = None

    hw.run_wizard(func=trainval, 
                  exp_groups=exp_configs.EXP_GROUPS, 
                  job_config=job_config, 
                  job_scheduler=args.job_scheduler,
                  python_binary_path=args.python_binary_path,
                  use_threads=True, args=args, results_fname='results.ipynb')
