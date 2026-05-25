import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    parser = argparse.ArgumentParser(description='CIDNet')
    parser.add_argument('--batchSize', type=int, default=8)
    parser.add_argument('--cropSize', type=int, default=256)
    parser.add_argument('--nEpochs', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--snapshots', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=16)

    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--start_warmup', type=_str2bool, default=True)
    
    parser.add_argument('--LAB_weight', type=float, default=0.5)
    parser.add_argument('--LPIPS_weight', type=float, default=0.5)

    # ============ 微调用预训练权重 ============
    parser.add_argument('--pretrain', type=str, default='',
                        help='Path to pretrained weights for finetuning')

    # ============ train datasets ============
    parser.add_argument('--data_train_lol_blur'   , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'     , type=str, default='./datasets/LOLdataset/our485')
    parser.add_argument('--data_train_lolv2_real' , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'  , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'        , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'       , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'      , type=str, default='./datasets/FiveK/train')
    parser.add_argument('--data_train_uieb'       , type=str, default='./datasets/UIEB/train')

    # ============ validation input ============
    parser.add_argument('--data_val_lol_blur'   , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'     , type=str, default='./datasets/LOLdataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real' , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'  , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'        , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'   , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'  , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_test_fivek'     , type=str, default='./datasets/FiveK/test/input')
    parser.add_argument('--data_val_uieb'       , type=str, default='./datasets/UIEB/test/low')

    # ============ validation groundtruth ============
    parser.add_argument('--data_valgt_lol_blur'   , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'     , type=str, default='./datasets/LOLdataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real' , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'  , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'        , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'   , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'  , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'      , type=str, default='./datasets/FiveK/test/target/')
    parser.add_argument('--data_valgt_uieb'       , type=str, default='./datasets/UIEB/test/high/')

    parser.add_argument('--val_folder', default='./results/')

    parser.add_argument('--HVI_weight', type=float, default=1.0)
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5)
    parser.add_argument('--E_weight',  type=float, default=50.0)
    parser.add_argument('--P_weight',  type=float, default=1e-2)

    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    parser.add_argument('--grad_detect', type=_str2bool, default=False)
    parser.add_argument('--grad_clip', type=_str2bool, default=True)

    # ============ dataset choices（加了 uieb） ============
    parser.add_argument('--dataset', type=str, default='lol_v1',
        choices=['lol_v1', 'lolv2_real', 'lolv2_syn', 'lol_blur',
                 'SID', 'SICE_mix', 'SICE_grad', 'fivek',
                 'uieb'],
        help='Select the dataset to train on')

    return parser
