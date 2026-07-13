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
            'task4_rsp_high_weight',
            'task4_acat_high_weight',
            'task4_acat_shuffle_high_weight',
            'task4_acat_trainhard_weight',
            'task4_highdetail_trainhard_weight',
            'task4_highdetail_trainhard_shuffle_weight',
            'task4_acat_pairmargin_weight',
            'task4_acat_rsp_residual_pairmargin',
            'task4_acat_hardonly_qmargin',
            'task4_highdetail_pairmargin',
            'task4_highdetail_pairmargin_shuffle',
            'task4_competitor_pair',
            'task4_competitor_pair_shuffle',
            'task4_competitor_pair_rsp_control',
            'task4_competitor_pair_acat_control',
            'task4_boundary_competitor_pair',
            'task4_boundary_competitor_pair_shuffle',
            'task4_boundary_competitor_pair_rsp_control',
            'task4_boundary_competitor_pair_acat_control',
            'm11_target_competitor_pair',
            'm11_target_competitor_pair_shuffle',
            'm11_target_competitor_pair_lowrsp_control',
            'm11_target_competitor_pair_rsp_control',
            'm11r1_full_target_competitor_pair',
            'm11r1_popmatch_competitor_pair_control',
            'm11r1_lowacat_competitor_pair_control',
            'm11r2_qbpr_score_weight',
            'm11r2_qbpr_focal',
            'm11r2_qbpr_curriculum',
            'm11r2_target_feature_fusion',
            'm11r3_dual_residual',
            'm11r3_norm_capped_residual',
            'm11r3_neighbor_transfer',
            'm11r3_target_film',
            'm11r4_protected_experts',
            'm11r4_continuous_fusion',
            'm11r4_relational_alignment',
            'm11r4_continuous_focal',
            'cicpr1_e4_residual',
            'cicpr1_modality_routing',
            'cicpr1_category_expert',
            'cicpr1_alignment_curriculum',
            'cicpr1_counterfactual_margin',
            'cicpr1_adaptive_attention',
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
    parser.add_argument('--task4_profile_path', type=str, default='', help='Task4 train-safe hard proxy profile CSV')
    parser.add_argument('--task4_loss_alpha', type=float, default=0.5, help='extra q-side loss weight for Task4 target items')
    parser.add_argument('--task4_shuffle_seed', type=int, default=43, help='deterministic shuffle seed for Task4 Acat shuffle control')
    parser.add_argument('--task4_disable_q_bpr_weight', action='store_true', help='disable Task4 q_v_c-user BPR weighting')
    parser.add_argument('--task4_disable_self_contrast_weight', action='store_true', help='disable Task4 self contrast weighting')
    parser.add_argument('--task4_reweight_contrast', action='store_true', help='also apply Task4 weights to q_v_c item contrast loss')
    parser.add_argument('--task4_pair_margin', type=float, default=0.2, help='q_v_c target-user vs competitor-user margin for Task4 pairwise-margin variants')
    parser.add_argument('--task4_competitor_alpha', type=float, default=0.25, help='Task4 R4 competitor-pair loss scale')
    parser.add_argument('--task4_competitor_margin', type=float, default=0.1, help='Task4 R4 target-vs-competitor softplus margin')
    parser.add_argument('--task4_competitor_k', type=int, default=20, help='Task4 R4 competitor candidate count; first implementation uses batch neg_user')
    parser.add_argument('--task4_boundary_competitor_cache_path', type=str, default='', help='Task4 R5 train-safe boundary competitor cache CSV')
    parser.add_argument('--m11r2_focal_gamma', type=float, default=2.0, help='M11-R2 focal qBPR difficulty exponent')
    parser.add_argument('--m11r2_focal_temperature', type=float, default=1.0, help='M11-R2 focal qBPR score-difference temperature')
    parser.add_argument('--m11r2_curriculum_warmup_epochs', type=int, default=20, help='M11-R2 qBPR target-weight warmup epochs')
    parser.add_argument('--m11r2_feature_dim', type=int, default=16, help='M11-R2 projected train-safe target-feature dimension')
    parser.add_argument('--m11r3_residual_max_ratio', type=float, default=0.15, help='maximum M11 residual norm as a ratio of the base hidden norm')
    parser.add_argument('--m11r3_neighbor_loss_weight', type=float, default=0.1, help='weight for one-way target-to-neighbor residual transfer')
    parser.add_argument('--m11r3_neighbor_temperature', type=float, default=0.25, help='distance temperature for train-batch M11 structural neighbors')
    parser.add_argument('--m11r3_film_strength', type=float, default=0.1, help='maximum target-conditioned FiLM modulation strength')
    parser.add_argument('--m11r4_expert_film_strength', type=float, default=0.2, help='non-target FiLM strength for M11-R4 protected experts')
    parser.add_argument('--m11r4_fusion_strength', type=float, default=0.25, help='continuous category/image modulation strength for M11-R4 fusion')
    parser.add_argument('--m11r4_relation_loss_weight', type=float, default=0.05, help='target-to-all relational alignment loss weight')
    parser.add_argument('--m11r4_focal_alpha', type=float, default=1.5, help='continuous full-coverage focal objective weight scale')
    parser.add_argument('--m11r4_focal_gamma', type=float, default=2.0, help='continuous full-coverage focal difficulty exponent')
    parser.add_argument('--m11r4_focal_temperature', type=float, default=0.5, help='continuous full-coverage focal difficulty temperature')
    parser.add_argument('--m11r4_focal_floor', type=float, default=0.35, help='minimum full-coverage M11 signal strength')
    parser.add_argument('--cicp_profile_path', type=str, default='', help='frozen train/validation CICP profile CSV')
    parser.add_argument('--cicp_feature_dim', type=int, default=16, help='CICP projected feature dimension for the E4-style residual only')
    parser.add_argument('--cicp_residual_max_ratio', type=float, default=0.15, help='maximum CICP E4-style residual norm relative to base hidden norm')
    parser.add_argument('--cicp_modality_strength', type=float, default=0.25, help='maximum category/image routing scale')
    parser.add_argument('--cicp_expert_strength', type=float, default=0.20, help='maximum category-expert mixture share')
    parser.add_argument('--cicp_alignment_weight', type=float, default=0.05, help='CICP collaborative alignment loss weight')
    parser.add_argument('--cicp_alignment_warmup_epochs', type=int, default=20, help='CICP alignment curriculum warmup epochs')
    parser.add_argument('--cicp_counterfactual_weight', type=float, default=0.05, help='CICP category counterfactual objective weight')
    parser.add_argument('--cicp_counterfactual_margin', type=float, default=0.05, help='CICP real-vs-shuffled category margin target')
    parser.add_argument('--cicp_attention_strength', type=float, default=0.50, help='CICP category attention log-temperature strength')
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
