import argparse
import os
import torch


def build_model(arch):
    if arch == 'HPI_GCN_RP':
        from model.HPI_GCN_RP import Model
        model = Model(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph')
    elif arch == 'HPI_GCN_OP':
        from model.HPI_GCN_OP import Model
        model = Model(num_class=120, num_point=25, num_person=2, graph='graph.ntu_rgb_d.Graph')
    else:
        raise ValueError('no such model')
    return model


def convert():
    args = parser.parse_args()

    train_model = build_model(args.arch)

    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        train_model.load_state_dict(ckpt)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    for m in train_model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    torch.save(train_model.state_dict(), args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPI_GCN Conversion')
    parser.add_argument('--load', metavar='LOAD',
                        default='./test_weights/weight_ntu120/HPI_120CSub_OP_T9K9_j0/runs-94-46248.pt',
                        help='path to the weights file')
    parser.add_argument('--save', metavar='SAVE',
                        default='./test_weights/weight_ntu120/HPI_120CSub_OP_T9K9_j0/inferHPI_OP_K9.pt',
                        help='path to the weights file')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='HPI_GCN_OP')

    convert()
