import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--all_datasets", type=str, help="path to excel file that contains dataset information")
    parser.add_argument("--excel_file", type=str, help="path to excel file for used dataset")
    parser.add_argument("--feature", type=str, help="which feature extractor to use")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--into_memory", type=str, default="True", help="load dataset into memory")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="ABMIL", help="type of model")
    parser.add_argument("--study", type=str, help="used dataset")

    # Optimizer Parameters
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=50, help="maximum number of epochs to train (default: 50)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="ce", help="slide-level classification loss function (default: ce)")
    args = parser.parse_args()
    return args
