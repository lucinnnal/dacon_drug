import argparse
import os

def get_arguments():

    parser = argparse.ArgumentParser(description="ChemBERT Training Script")
    
    #===== K-fold and Validation =====================================#
    parser.add_argument('--is_kfold', action='store_true', help='Use k-fold cross-validation')
    parser.add_argument('--k_fold', type=int, default=5, help='Number of folds for k-fold cross-validation')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    
    #================= Training Parameters ===========================#    
    parser.add_argument('--logging_steps', type=int, default=10, help='Logging steps')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--per_device_train_batch_size', type=int, default=64, help='Per device train batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='Per device eval batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer')

    #================= Model Parameters =============================#
    parser.add_argument('--model_name', type=str, default='DeepChem/ChemBERTa-77M-MLM', help='Pretrained ChemBERT model name or path')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for model')
    parser.add_argument('--bert_out_dim', type=int, default=1, help='Output dimension of BERT model')
    parser.add_argument('--mol_f_dim', type=int, default=2083, help='Molecular feature dimension')

    #================= Data and Save Paths ==========================#
    parser.add_argument('--save_dir', type=str, default='chembert_baseline', help='Save directory')
    parser.add_argument('--data_dir', type=str, default='./src/data/data', help='Data directory')
    parser.add_argument('--train_file', type=str, default='../data/train.csv', help='Training data file')
    parser.add_argument('--test_file', type=str, default='../data/test.csv', help='Test data file')
    parser.add_argument('--submission_file', type=str, default='../sample_submission.csv', help='Submission file template')
    parser.add_argument('--base_file', type=str, default='./src/data/data', help='Output submission file')
    parser.add_argument('--num_labels', type=int, default=1, help='Number of labels (regression=1)')

    args = parser.parse_args()

    return args