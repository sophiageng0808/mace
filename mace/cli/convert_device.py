from argparse import ArgumentParser

import torch

from mace.tools.scripts_utils import load_model, save_model


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--target_device",
        "-t",
        help="device to convert to, usually 'cpu' or 'cuda'",
        default="cpu",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        help="name for output model, defaults to model_file.target_device",
    )
    parser.add_argument("model_file", help="input model file path")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.model_file + "." + args.target_device

    model = load_model(args.model_file, map_location=args.target_device, weights_only=False)
    model.to(args.target_device)
    save_model(model, args.output_file, config_model=model)


if __name__ == "__main__":
    main()
