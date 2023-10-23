import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint.write_log("total params: {}, total trainable params: {}".format(pytorch_total_params, pytorch_total_trainable_params))

    my_loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, my_loss, checkpoint)
    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()

