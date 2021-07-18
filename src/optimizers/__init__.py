import torch
from . import sps_mirror

def get_optimizer(opt, params, train_loader, exp_dict):
    """
    opt: name or dict
    params: model parameters
    n_batches_per_epoch: b/n
    """
    opt_name = opt["name"]
    opt_dict = opt

    n_train = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    n_batches_per_epoch = n_train / float(batch_size)
    
    if opt_name == 'sps_mirror':
        opt = sps_mirror.SpsMirror(params, c=opt_dict["c"], 
                        mu=opt_dict["mu"], 
                        n_batches_per_epoch=n_batches_per_epoch, 
                        step_size_method=opt_dict.get('step_size_method', 'constant'),
                        fstar=0,
                        eta_max=opt_dict.get('eta_max'),
                        eps=opt_dict.get('eps', 0),
                        gamma=opt_dict.get('gamma', 2),
                        pnorm=opt_dict.get('pnorm', 2),
                        project_method=opt_dict.get('project_method', None),
                        )
    else:
        raise ValueError("opt %s does not exist..." % opt_name)

    return opt




