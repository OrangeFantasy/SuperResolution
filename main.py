import os
import time
import datetime
from typing import List
import torch

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader

from core import constant
from core.trainer import Trainer
from core.tools import logger
from core.utils import instantiate_from_config, parse_params, sys_setting, load_yaml_and_convert


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--data_config", type=str, default="configs/scenes/Bunker/Bunker_train.yaml")
    parser.add_argument("--trainer_config", type=str, default="configs/trainer/4x_H2.yaml")

    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--resume_epoch", type=int, default=70)
    parser.add_argument("--ckpt", type=str, default="checkpoints/Sampler_Render/Bunker/SuperSampler_4x_last.ckpt")
    
    parser.add_argument("--compile_mode", type=str, default="none", choices=["none", "default", "max-autotune"])
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--scales", type=List[int], default=[2])

    parser.add_argument("--override_data_params", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("-----   Program Start   -----")
    start_time = time.time()
    
    sys_setting(seed=42, cudnn_benchmark=True)
    if constant.ENABLE_DEBUG_TRACE:
        torch.autograd.set_detect_anomaly(True)

    # Parse arguments.
    args = get_args()
    logger.record_dict("[Config]", dict_msg=vars(args))
    
    # Parse and update the configuration file.
    data_config = load_yaml_and_convert(args.data_config)
    if args.override_data_params:
        for param in args.override_data_params.split(" "):
            key, value = parse_params(param)
            if not isinstance(value, dict):
                data_config["params"].update({key: value})
            else:
                data_config["params"][key].update(value)
    scene = data_config["params"]["scene"]

    trainer_config = load_yaml_and_convert(args.trainer_config)
    model_label = trainer_config["params"]["label"]
    trainer_config["params"].update({"label": f"{model_label}/{scene}"})

    logger.record_dict("[Config]", dict_msg=trainer_config)
    logger.record_dict("[Config]", dict_msg=data_config)

    # Create dataset and dataloader.
    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2**31 + worker_id
        import random
        import numpy as np
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
    train_dataset = instantiate_from_config(data_config)
    train_dataloader = DataLoader(
        train_dataset, args.batch_size, shuffle=True, num_workers=min(8, args.batch_size), pin_memory=True)#, worker_init_fn=worker_init_fn)

    # Create trainer.
    trainer: Trainer = instantiate_from_config(trainer_config)
    if args.resume:
        trainer.load_checkpoint(args.ckpt, args.resume_epoch, strict=False)
    if args.compile_mode != "none":
        os.environ["TORCH_LOGS"] = "+dynamo"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        torch._dynamo.config.suppress_errors = True
        trainer.compile(mode=args.compile_mode)
    trainer.train()
    logger.flush()
    
    # Training.
    n_batch = len(train_dataloader)
    for epoch in range(args.resume_epoch if args.resume else 0, args.max_epochs):
        epoch_start_time = time.time()
        trainer.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(train_dataloader):
            for scale in args.scales:
                trainer.training_step(batch, batch_idx, scale)

            if (batch_idx + 1) % (n_batch // 10) == 0:
                trainer.print_loss(batch_idx, n_batch)
            if (trainer.global_step + 1) % (n_batch * 4) == 0:
                trainer.visualize_curr_bacth(batch, scale)
        
        # Save the model.
        if (epoch + 1) % 50 == 0:
            trainer.save_state_dict()
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint()
        
        epoch_second = time.time() - epoch_start_time
        remaining_second = (args.max_epochs - 1 - epoch) * epoch_second
        completed_time = (datetime.datetime.now() + datetime.timedelta(seconds=remaining_second)).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            f"End of epoch {epoch}. Time taken: {int(epoch_second // 60)} min {int(epoch_second % 60)} sec. "
            f"Estimated time of training completion: {completed_time}."
        )

        trainer.update_lr()
        logger.flush()
    
    total_time = time.time() - start_time
    logger.info("Total time taken: %d min %d sec." % (int(total_time // 60), int(total_time % 60)))
    logger.info("-----   Program Completed    -----")