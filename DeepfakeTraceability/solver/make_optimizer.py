import torch

def make_optimizer(config, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = config["solver"]['base_lr']
        weight_decay = config["solver"]['weight_decay']
        if "bias" in key:
            lr = config["solver"]['base_lr'] * config["solver"]['bias_lr_factor']
            weight_decay = config["solver"]['weight_decay_bias']
        if config["solver"]['large_fc_lr']:
            if "classifier" in key or "arcface" in key:
                lr = config["solver"]['base_lr'] * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if config["solver"]['type'] == 'SGD':
        optimizer = getattr(torch.optim, config["solver"]['type'])(params, momentum=config["solver"]['momentum'])
    elif config["solver"]['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config["solver"]['base_lr'], weight_decay=config["solver"]['weight_decay'])
    else:
        optimizer = getattr(torch.optim, config["solver"]['type'])(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=config["solver"]['center_lr'])

    return optimizer, optimizer_center
