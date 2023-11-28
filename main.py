import argparse
import traceback
import logging
import yaml
import sys
import os
import time
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

torch.set_printoptions(sci_mode=False)

parser = argparse.ArgumentParser(description=globals()["__doc__"])

parser.add_argument("--config", type=str, default='./configs/office-home.yml', help="Path to the config file")
parser.add_argument('--device', type=int, default=0, help='GPU device id')
parser.add_argument("--tune_T", action="store_true", help="Whether to tune the scaling temperature parameter for calibration with training set.")
parser.add_argument("--add_ce_loss", action="store_true", help="Whether to add cross entropy loss")
parser.add_argument("--setting", choices=['ADA', 'SFADA'], default="ADA", help="Run ADA task or SFADA task.")
parser.add_argument("--source_domain", type=str, default= None, help="Identify the source domain.")
parser.add_argument("--target_domain", type=str, default= None, help="Identify the target domain.")
parser.add_argument("--lambda_kl",type=float,default = None)
parser.add_argument("--lambda_kd",type=float,default = None)

args = parser.parse_args()


def parse_config():
    # parse config file
    with open(os.path.join(args.config), "r") as f:
        config = yaml.safe_load(f)
        new_config = dict2namespace(config)
    # override the setting in the configure file
    if args.source_domain and args.target_domain:
        new_config.data.source_domain = args.source_domain
        new_config.data.target_domain = args.target_domain
    if args.lambda_kl:
        new_config.training.weight_kl = args.lambda_kl
    if args.lambda_kd:
        new_config.training.weight_kd = args.lambda_kd

    args.exp = new_config.data.dataset
    args.log_path = os.path.join(args.exp, "{}_to_{}".format(new_config.data.source_domain, new_config.data.target_domain) , "logs")


    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    with open(os.path.join(args.log_path, "config.yml"), "w") as f:
        yaml.dump(new_config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, "INFO", None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device
    device_name = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    if new_config.data.seed is not None:
        args.seed = new_config.data.seed
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    return new_config, logger


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    config, logger = parse_config()
    logging.info("Log file: {}".format(args.log_path))
    try:
        if args.setting == "ADA":
            from dapm_ada import Diffusion
        else:
            from dapm_sfada import Diffusion

        runner = Diffusion(args, config, device=config.device)
        start_time = time.time()
        runner.train()
        end_time = time.time()
        logging.info("\nTraining procedure finished. It took {:.4f} minutes.\n\n\n".format(
        (end_time - start_time) / 60))
        # remove logging handlers
        handlers = logger.handlers[:]
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
