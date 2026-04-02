from argparse import ArgumentParser

import torch

from mace.tools.scripts_utils import extract_model


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

    model = torch.load(args.model_file, weights_only=False)
    model.to(args.target_device)
    # Rebuild so torch.save does not hit unpicklable e3nn ScriptFunction nodes.
    model = extract_model(model, map_location=args.target_device)
    torch.save(model, args.output_file)


if __name__ == "__main__":
    main()
