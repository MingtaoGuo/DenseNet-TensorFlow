import argparse
from train import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=40)
    parser.add_argument("--growth_rate", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--cifar10_path", type=str, default="./cifar10//")
    parser.add_argument("--class_nums", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--is_trained", type=int, default=True)

    args = parser.parse_args()

    if args.is_trained:
        acc = test_acc(path=args.cifar10_path+"test_batch", class_nums=args.class_nums, growth_rate=args.growth_rate, depth=args.depth)
        print("Test accuracy: %f"%(acc))
    else:
        train(batch_size=args.batch_size, class_nums=args.class_nums, growth_rate=args.growth_rate, weight_decay=args.weight_decay, depth=args.depth, cifar10_path=args.cifar10_path, train_epoch=args.epoch)
