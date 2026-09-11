import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='./data/', help='Input data path')  
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='netflix', help='Choose a dataset from {movieLens, netflix}')
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')  
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')                     
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='Early Stop Patience') 
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')  
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='Feature Reg Decay') 
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  
    parser.add_argument('--cf_model', nargs='?', default='lightgcn', help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, mmgcn, vbpr, hafr, bm3}')   
    parser.add_argument('--point', default="", type=str, help='')  #

    # train
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')  
    parser.add_argument('--de_lr', type=float, default=0.0002, help='Decoder learning rate.')  
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight_decay')  #

    # model
    parser.add_argument('--layers', type=int, default=1, help='Number of graph conv layers')  
    parser.add_argument('--drop_rate', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--mask_rate', type=float, default=0.0, help='Mask rate')   
    parser.add_argument('--mask', type=bool, default=False, help='If mask')   
    parser.add_argument('--user_cat_rate', type=float, default=2.8, help='User cat rate')
    parser.add_argument('--item_cat_rate', type=float, default=0.005, help='Item cat rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.02, help='Model cat rate')
    parser.add_argument('--de_drop1', default=0.31, type=float, help='for D model2')  #
    parser.add_argument('--de_drop2', default=0.5, type=float, help='')  #

    # loss
    parser.add_argument('--aug_mf_rate', type=float, default=0.012, help='Augmentation mf rate')      # 
    parser.add_argument('--prune_loss_drop_rate', type=float, default=0.71, help='Prune loss drop rate')    
    parser.add_argument('--mm_mf_rate', type=float, default=0.0001, help='MM mf rate')    
    parser.add_argument('--feat_loss_type', default="sce", type=str, help='Feature loss type')  #
    parser.add_argument('--att_re_rate', type=float, default=0.00000, help='Attribute restoration rate')      # 
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument('--aug_sample_rate', type=float, default=0.1, help='Augmentation sample rate')
    parser.add_argument('--mf_emb_rate', type=float, default=0.0, help='MF embedding rate')

    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='auto uses CUDA when available, otherwise CPU; sparse MPS is not enabled')
    parser.add_argument('--num_threads', type=int, default=4, help='CPU threads for PyTorch')
    parser.add_argument('--max_batches', type=int, default=0, help='Limit batches per epoch for a smoke test; 0 means all')
    parser.add_argument('--eval_users', type=int, default=0, help='Limit evaluation users for a smoke test; 0 means all')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='Evaluation users per score matrix')
    args = parser.parse_args()
    args.data_path = args.data_path.rstrip('/') + '/'
    if args.epoch < 1 or args.batch_size < 1 or args.num_threads < 1 or args.eval_batch_size < 1:
        parser.error('epoch, batch_size, num_threads and eval_batch_size must be positive')
    if args.max_batches < 0 or args.eval_users < 0:
        parser.error('max_batches and eval_users must be nonnegative')
    if not 0 <= args.prune_loss_drop_rate < 1 or not 0 <= args.aug_sample_rate <= 1:
        parser.error('prune_loss_drop_rate must be in [0, 1); aug_sample_rate in [0, 1]')
    return args
