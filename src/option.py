import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=3,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='/data/SR/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=5,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep_reset',
                    help='dataset file extension')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=192,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='MXBASRN',
                    help='model name')

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--no_attention', action='store_true',
                    help='not use channel attention in MXBASRN')
parser.add_argument('--use_multi_axis_atten', action='store_true',
                    help='use multi axis atten for high freq')
parser.add_argument('--num_multi_axis_atten', type=int, default='1',
                    help='number of multi axis atten block at inserted location')
parser.add_argument('--multi_axis_atten_fusion', type=str, default='sum',
                    help='multi axis atten fusion method')
parser.add_argument('--multi_axis_atten_method', type=str, default='hybrid_parallel',
                    help='multi axis atten method: pure, hybrid_parallel')
parser.add_argument('--multi_axis_atten_method_only_grid', action='store_true',
                    help='multi axis atten method only grid') 
parser.add_argument('--multi_axis_atten_method_only_block', action='store_true',
                    help='multi axis atten method only block') 
parser.add_argument('--atten_noself', action='store_true',
                    help='atten not including self postion')
parser.add_argument('--new_pytorch_version_diagonal_scatter', action='store_true',
                    help='use new pytorch version 1.11 for diagonal_scatter')
parser.add_argument('--no_perfect_balanced_blocking', action='store_true',
                    help='no perfect balanced blocking')
parser.add_argument('--small_upsampling_head', action='store_true',
                    help='small upsampling head')
parser.add_argument('--small_upsampling_head_with_width_threshold', action='store_true',
                    help='small upsampling head with width threshold')
parser.add_argument('--small_upsampling_head_with_width_threshold_value', type=int, default=32,
                    help='small upsampling head with width threshold value, default: 32.')
parser.add_argument('--full_attention', action='store_true', 
                    help='full attention')
parser.add_argument('--configurable_attention', action='store_true', 
                    help='configurable attention')
parser.add_argument('--configurable_attention_num', type=int, default='1',
                    help='configurable attention num')
parser.add_argument('--fix_position_attention', action='store_true', 
                    help='fix position attention')
parser.add_argument('--fix_position_attention_list', type=str, default='1',
                    help='fix position attention_list')
parser.add_argument('--with_layer_norm_channel', action='store_true', help='with layer norm channel')
parser.add_argument('--no_with_mxba_scale', action='store_true', help='no with mxba scale')
parser.add_argument('--no_rir', action='store_true', help='no rir')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')

# options for residual group and feature channel reduction
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')
# options for test
parser.add_argument('--testpath', type=str, default='../test/DIV2K_val_LR_our',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.data_test = args.data_test.split('+')

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

