from haven import haven_utils as hu
import itertools
# RUNS = [0, 1]
# RUNS = [0,1,2,3,4]
RUNS = [0]


def get_benchmark(benchmarks, opt_list):
    if not isinstance(benchmarks, list):
        benchmarks = [benchmarks]

    exp_list = []
    for benchmark in benchmarks:
        if benchmark == 'syn':
            exp_dict =  {"dataset": ["synthetic"],
                    "model_base": ["logistic"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "score_func": ["softmax_accuracy"],
                    'margin':
                    [
                        0.05,
                # 0.1,
                        # 0.5,
                        0.01,
            ],
                "n_samples": [1000],
                "d": 20,
                "batch_size": [100],
                "max_epoch": [200],
                "runs": RUNS}

        elif benchmark in ["ijcnn", "mushrooms", "rcv1",  "ijcnn_convex", "mushrooms_convex", "rcv1_convex"]:
            exp_dict =  {"dataset": [benchmark],
                    "model_base": ["logistic"],
                    "loss_func": [ 'softmax_loss'],
                    "score_func": ["softmax_accuracy"],
                    "opt": opt_list,
                    "batch_size": [100],
                    "max_epoch": [
                        200, 
                    # 300
                    ],
                    "runs": RUNS}

        elif benchmark == 'mf':
            exp_dict =  {"dataset": ["matrix_fac"],
                    "model_base": ["matrix_fac_1", "matrix_fac_4", "matrix_fac_10", "linear_fac"],
                    "loss_func": ["squared_loss"],
                    "opt": opt_list,
                    "score_func": ["mse"],
                    "batch_size": [100],
                    "max_epoch": [100],
                    "runs": RUNS}

        elif benchmark == 'mnist':
            exp_dict =  {"dataset": ["mnist"],
                    "model_base": ["mlp"],
                    "loss_func": ["softmax_loss"],
                    "opt": opt_list,
                    "score_func": ["softmax_accuracy"],
                    "batch_size": [128],
                    "max_epoch": [100],
                    "runs": RUNS}

        elif benchmark == 'cifar10':
            exp_dict =  {"dataset": ["cifar10"],
                    "model_base": [
                # "densenet121",
                "resnet34"
            ],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}

        elif benchmark == 'cifar100':
            exp_dict =  {"dataset": ["cifar100"],
                    "model_base": [
                # "densenet121_100",
                "resnet34_100"
            ],
                "loss_func": ["softmax_loss"],
                "opt": opt_list,
                "score_func": ["softmax_accuracy"],
                "batch_size": [128],
                "max_epoch": [200],
                "runs": RUNS}
        else:
            raise ValueError(f'nexist pas')
        exp_list += hu.cartesian_exp_group(exp_dict)

    return  exp_list

EXP_GROUPS = {}


pnorm = [1.2, 1.4, 1.6, 1.8]

opt_list = []

step_size_method = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 1e1, 1e2, 1e3, 1e4, 1e5]
# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[1], 
                                    'mu':[None],
                                    'step_size_method':['constant'], 
                                    'project_method':[ None]})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[None], 
                                    'mu':[None],
                                    'step_size_method':step_size_method, 
                                    'project_method':[None]})



EXP_GROUPS['row1'] = (get_benchmark(benchmarks= [
    'ijcnn'], opt_list=opt_list))


# ROW  2  ---------------------

# Row 2 shows convex with clipping  on rcv1, ijcnn, syn 1 and syn2  
#  (inferred mu and `c=1` against constant step sizes)



opt_list = []

# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':2, 'c':[1], 
                                    'mu':[None],
                                    'step_size_method':['constant'], 
                                    'project_method':[ 'clip']})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':2, 'c':[None], 
                                    'mu':[None],
                                    'step_size_method':step_size_method, 
                                    'project_method':['clip']})



EXP_GROUPS['row2'] = (get_benchmark(benchmarks= [
    'syn', 'rcv1', 'ijcnn', ], opt_list=opt_list))



# ROW  3  ---------------------

# Row 3 shows convex with simplex  on  rcv1, ijcnn, 
# syn 1 and syn2   (inferred mu and `c=1` against constant step sizes)



opt_list = []

# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':2, 'c':[1], 
                                    'mu':[None],
                                    'step_size_method':['constant'], 
                                    'project_method':[ 'L1']})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':2, 'c':[None], 
                                    'mu':[None],
                                            #   'step_size_method':[ 1e-3, ], 
                                    'step_size_method':step_size_method, 
                                    'project_method':['L1']})



EXP_GROUPS['row3'] = (get_benchmark(benchmarks= [
   'ijcnn_convex', 'rcv1_convex', 'syn' ], opt_list=opt_list))


# ROW  4  ---------------------

# Row 4 shows deep learning with sweep over pnorm 1.4, 1.6, 1.8 on CIFAR10 and the last sub-plot is the 
# different step-sizes for each pnorm  (inferred mu with c=0.2 agains constant step sizes)


pnorm = [1.4, 1.6, 1.8]

opt_list = []

# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[.2,], 
                                    'mu':[None],
                                    'step_size_method':['smooth_iter'], 
                                    'project_method':[ None]})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[None], 
                                    'mu':[None],
                                    'step_size_method':step_size_method, 
                                    'project_method':[None]})




EXP_GROUPS['row4'] = (get_benchmark(benchmarks= [
    'cifar10', ], opt_list=opt_list))



pnorm = [1.2, 1.4, 1.6, 1.8]

opt_list = []

# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[ 1], 
                                    'mu':[None],
                                    'step_size_method':['constant'], 
                                    'project_method':[ None]})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[None], 
                                    'mu':[None],
                                    'step_size_method':step_size_method, 
                                    'project_method':[None]})

EXP_GROUPS['appendix1'] = EXP_GROUPS['mushrooms'] =  (get_benchmark(benchmarks= [
    'mushrooms', ], opt_list=opt_list))

opt_list = []

# ours
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[ .2], 
                                    'mu':[None],
                                    'step_size_method':['smooth_iter'], 
                                    'project_method':[ None]})
# constant
opt_list += hu.cartesian_exp_group({'name': ["sps_mirror"], 
                                    'pnorm':pnorm, 'c':[None], 
                                    'mu':[None],
                                    'step_size_method':step_size_method, 
                                    'project_method':[None]})

EXP_GROUPS['appendix2'] = (get_benchmark(benchmarks= [
     'mnist', ], opt_list=opt_list))