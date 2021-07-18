import torch


from . import classifier

def get_model(train_loader, exp_dict, device):
    if 'model_base' in exp_dict:
        model_name = exp_dict['model_base']['name'] if type(exp_dict['model_base']) is dict else exp_dict['model_base']
    else:
        model_name = 'clf'
    
    return classifier.Classifier(train_loader, exp_dict, device)

    
