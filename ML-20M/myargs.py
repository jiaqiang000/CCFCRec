import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--learning_rate', type=float, default=0.000005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--positive_number', type=int, default=10, help='contrast positive number')
    parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')
    parser.add_argument('--self negative_number', type=int, default=40, help='contrast negative number')
    parser.add_argument('--attr_num', type=int, default=18, help='item attribute number')
    parser.add_argument('--attr_present_dim', type=int, default=128, help='the dimension of present')
    parser.add_argument('--implicit_dim', type=int, default=128, help='the dimension of u/i present')
    parser.add_argument('--cat_implicit_dim', type=int, default=128, help='the q_v_c dimension')
    parser.add_argument('--user_number', type=int, default=138493, help='user number in training set')
    parser.add_argument('--item_number', type=int, default=16803, help='item number in training set')
    parser.add_argument('--tau', type=float, default=0.1, help='contrast loss temperature')
    parser.add_argument('--lambda1', type=float, default=0.5, help='collaborative contrast loss weight')
    parser.add_argument('--epoch', type=int, default=10, help='training epoch')
    parser.add_argument('--pretrain', type=bool, default=False, help='user/item embedding pre-training')
    parser.add_argument('--pretrain_update', type=bool, default=False, help='u/i pretrain embedding update')
    parser.add_argument('--contrast_flag', type=bool, default=True, help='contrast job flag')
    parser.add_argument('--user_flag', type=bool, default=False, help='use user to q_v_c flag')
    parser.add_argument('--save_batch_time', type=int, default=3000, help='every batch time save the model')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker process count')
    parser.add_argument('--pin_memory', action='store_true', help='pin host memory for faster CUDA transfer')
    parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between epochs')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor when num_workers > 0')
    parser.add_argument('--multiprocessing_context', type=str, default='', help='DataLoader multiprocessing context, e.g. fork on macOS')
    args = parser.parse_args()
    return args


def args_tostring(args):
    str_ = ""
    for arg in vars(args):
        str_ += str(arg) + ":" + str(getattr(args, arg)) + "\n"
    return str_
