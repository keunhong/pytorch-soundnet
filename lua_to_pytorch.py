import argparse
import pathlib

import torch
import torchfile
from soundnet import SoundNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='in_path', type=pathlib.Path)
    parser.add_argument(dest='out_path', type=pathlib.Path)
    args = parser.parse_args()

    lua_model = torchfile.load(args.in_path)

    print(" * Loaded lua model")
    for i, module in enumerate(lua_model['modules']):
        print(f"    {i}, {module._typename}")

    model = SoundNet()
    model_map = {
        0: model.conv1,
        4: model.conv2,
        8: model.conv3,
        11: model.conv4,
        14: model.conv5,
        18: model.conv6,
        21: model.conv7,
    }

    for lua_idx, module in model_map.items():
        print(f" * Importing {module}")
        weight = torch.from_numpy(lua_model['modules'][lua_idx]['weight'])
        bias = torch.from_numpy(lua_model['modules'][lua_idx]['bias'])
        module.weight.data.copy_(weight)
        module.bias.data.copy_(bias)

    lua_conv8_objs = lua_model['modules'][24]['modules'][0]
    lua_conv8_scns = lua_model['modules'][24]['modules'][1]

    print(f" * Importing {model.conv8_objs}")
    weight = torch.from_numpy(lua_conv8_objs['weight'])
    bias = torch.from_numpy(lua_conv8_objs['bias'])
    model.conv8_objs.weight.data.copy_(weight)
    model.conv8_objs.bias.data.copy_(bias)

    print(f" * Importing {model.conv8_scns}")
    weight = torch.from_numpy(lua_conv8_scns['weight'])
    bias = torch.from_numpy(lua_conv8_scns['bias'])
    model.conv8_scns.weight.data.copy_(weight)
    model.conv8_scns.bias.data.copy_(bias)

    print(f" * Saving pytorch model to {args.out_path!s}")
    torch.save(model.state_dict(), args.out_path)


if __name__ == '__main__':
    main()

