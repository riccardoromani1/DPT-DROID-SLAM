import argparse
import random
from datetime import datetime


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions:

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser = self.initialize_args(parser)
        args = parser.parse_args()
        if args.datetime is None:
            args.datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        name = f"{args.datetime}_{args.name}_{args.model}"
        if hasattr(args, 'split'):
            name += f"_{args.split}"
        args.checkpoint_path = f"checkpoints/{name}"
        args.log_path = f"logs/{name}"
        args.result_path = f"results/{name}"
        if hasattr(args, 'world_size'):
            args.batch_size = args.batch_size // args.world_size
            args.master_port = f'{10000 + random.randrange(1, 10000)}'
        return args
