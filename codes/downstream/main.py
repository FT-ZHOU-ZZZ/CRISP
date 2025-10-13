import os
import json
import time
import wandb


from datasets.Dataset import Dataset
from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler

from torch.utils.data import DataLoader, SubsetRandomSampler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.resume is not None:
        results_dir = os.path.dirname(args.resume)
    else:
        results_dir = "./results/seed-{seed}/{study}/{feature}/{time}".format(
            study=args.study,
            feature=args.feature,
            seed=args.seed,
            time=time.strftime("%Y-%m-%d-%H-%M-%S"),
        )
    os.makedirs(results_dir, exist_ok=True)
    print("[dir] results directory: ", results_dir)
    # set wandb
    args.run = wandb.init(mode="offline", dir=results_dir)
    #
    dataset = Dataset(all_datasets=args.all_datasets, feature=args.feature, split_file=args.excel_file)
    args.num_classes = len(dataset.classes)
    splits = dataset.splits
    dataloaders = {split: DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(indices)) for split, indices in splits.items()}
    # build model, criterion, optimizer, schedular
    if args.model == "ABMIL":
        from models.ABMIL.network import DAttention
        from models.ABMIL.engine import Engine

        model = DAttention(n_classes=len(dataset.classes), dropout=0.25, act="relu", n_features=dataset.n_features)
        engine = Engine(args, results_dir, splits)
    else:
        raise NotImplementedError("model [{}] is not implemented".format(args.model))
    print("[model] trained model: ", args.model)
    criterion = define_loss(args)
    print("[model] loss function: ", args.loss)
    optimizer = define_optimizer(args, model)
    print("[model] optimizer: ", args.optimizer, args.lr, args.weight_decay)
    scheduler = define_scheduler(args, optimizer)
    print("[model] scheduler: ", args.scheduler)

    scores = engine.learning(model, dataloaders, criterion, optimizer, scheduler)
    with open(os.path.join(results_dir, "result.json"), "w") as f:
        json.dump(scores, f)

    args.run.finish()


if __name__ == "__main__":
    args = parse_args()
    results = main(args)
    print("finished!")
