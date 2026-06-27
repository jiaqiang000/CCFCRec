import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024, help="batch_size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
    parser.add_argument('--positive_number', type=int, default=5, help='contrast positive number')
    parser.add_argument('--negative_number', type=int, default=40, help='contrast negative number')
    parser.add_argument('--self negative_number', type=int, default=40, help='contrast negative number')
    parser.add_argument('--attr_num', type=int, default=3127, help='item attribute number')
    parser.add_argument('--attr_present_dim', type=int, default=256, help='the dimension of present')
    parser.add_argument('--implicit_dim', type=int, default=256, help='the dimension of u/i present')
    parser.add_argument('--cat_implicit_dim', type=int, default=256, help='the q_v_c dimension')
    parser.add_argument('--user_number', type=int, default=138493, help='user number in training set')
    parser.add_argument('--item_number', type=int, default=16803, help='item number in training set')
    parser.add_argument('--tau', type=float, default=0.1, help='contrast loss temperature')
    parser.add_argument('--lambda1', type=float, default=0.6, help='collaborative contrast loss weight')
    parser.add_argument('--epoch', type=int, default=100, help='training epoch')
    parser.add_argument('--pretrain', type=bool, default=False, help='user/item embedding pre-training')
    parser.add_argument('--pretrain_update', type=bool, default=False, help='u/i pretrain embedding update')
    parser.add_argument('--contrast_flag', type=bool, default=True, help='contrast job flag')
    parser.add_argument('--user_flag', type=bool, default=False, help='use user to q_v_c flag')
    parser.add_argument('--save_batch_time', type=int, default=300, help='every batch time save the model')
    parser.add_argument('--result_root', type=str, default='result', help='root directory for per-run training outputs')
    parser.add_argument('--seed', type=int, default=-1, help='random seed; negative means unset')
    parser.add_argument(
        '--method_variant',
        type=str,
        default='baseline',
        choices=[
            'baseline',
            'weak_q_reweight',
            'category_conf_input',
            'adaptive_conf_qbpr',
            'category_conf_fusion_gate',
        ],
        help='training method variant',
    )
    parser.add_argument('--weak_cat_threshold', type=int, default=3, help='category_count threshold for weak item reweighting')
    parser.add_argument('--weak_loss_alpha', type=float, default=0.5, help='extra loss weight for weak-category items')
    parser.add_argument('--reweight_q_bpr', action='store_true', help='reweight q_v_c-user BPR loss for weak-category items')
    parser.add_argument('--reweight_self_contrast', action='store_true', help='reweight q_v_c-item self contrast loss for weak-category items')
    parser.add_argument('--reweight_contrast', action='store_true', help='reweight q_v_c positive/negative item contrast loss for weak-category items')
    parser.add_argument('--category_conf_dim', type=int, default=16, help='category confidence embedding dimension')
    parser.add_argument('--category_conf_max_count', type=int, default=5, help='max category count used for confidence scalar clipping')
    parser.add_argument('--category_gate_scale', type=float, default=0.5, help='max residual attr/image scale shift for category confidence fusion gate')
    parser.add_argument('--adaptive_loss_alpha', type=float, default=1.0, help='extra qBPR weight scale for supported weak-category items')
    parser.add_argument('--adaptive_history_max_count', type=int, default=20, help='history length cap for adaptive support confidence')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker process count')
    parser.add_argument('--pin_memory', action='store_true', help='pin host memory for faster CUDA transfer')
    parser.add_argument('--persistent_workers', action='store_true', help='keep DataLoader workers alive between epochs')
    parser.add_argument('--prefetch_factor', type=int, default=2, help='DataLoader prefetch factor when num_workers > 0')
    parser.add_argument('--multiprocessing_context', type=str, default='', help='DataLoader multiprocessing context, e.g. fork on macOS')
    parser.add_argument('--validate_batch_size', type=int, default=512, help='validation item batch size')
    parser.add_argument(
        '--negative_sampling_mode',
        type=str,
        default='fast_uniform',
        choices=['legacy_cached', 'fast_uniform', 'original_np_choice'],
        help='fast_uniform is the fixed fast protocol for new experiments; legacy_cached uses compact candidate sampling for conservative diagnostics; original_np_choice reproduces the 2026-06-08/09 raw-item np.choice path for diagnostics',
    )
    parser.add_argument(
        '--negative_sampling_cache_size',
        type=int,
        default=512,
        help='kept for script compatibility; legacy_cached uses compact-index sampling and does not cache full negative arrays',
    )
    args = parser.parse_args()
    return args


def args_tostring(args):
    str_ = ""
    for arg in vars(args):
        str_ += str(arg) + ":" + str(getattr(args, arg)) + "\n"
    return str_
