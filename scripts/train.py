import wandb
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import KFold
from datasets import Dataset

import os
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from configs.training_arguements import get_arguments
from src.datasets.dataset import get_dataset
from src.models.get_model import get_model
from src.utils.compute_metrics import compute_metrics
from src.trainer import RMSETrainer

def main(args):
    # environmental variables
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    """

    # Load dataset
    train_dataset, _ = get_dataset(args)
    
    # K-fold Cross Validation
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)

    # 전체 데이터를 numpy array로 변환 (K-fold를 위해)
    all_data = train_dataset.to_pandas()
    X = all_data.drop(['labels'], axis=1)
    y = all_data['labels'].values

    fold_scores = []
    fold_models = []  # fold별 모델 객체 저장용
    fold_detailed_results = []  # fold별 상세 결과 저장용
    best_score = -np.inf  # 초기화 (custom_score가 클수록 좋은 경우)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold + 1}/{args.k_fold} ===")

        fold_run = wandb.init(
            project='drug', 
            name=f'{args.save_dir}_fold{fold+1}',
            job_type='fold',
            group=f'{args.save_dir}_kfold',                
            reinit=True
            )
        
        """
        # 각 fold별로 새로운 wandb run 생성
        if local_rank == 0:
            fold_run = wandb.init(
                project='drug', 
                name=f'{args.save_dir}_fold{fold+1}',
                job_type='fold',
                group=f'{args.save_dir}_kfold',
                reinit=True
            )
        else:
            fold_run = None  # wandb logging 안함
        """

        fold_train_data = all_data.iloc[train_idx].reset_index(drop=True)
        fold_val_data = all_data.iloc[val_idx].reset_index(drop=True)

        fold_train_dataset = Dataset.from_pandas(fold_train_data)
        fold_val_dataset = Dataset.from_pandas(fold_val_data)
        model = get_model(args)

        training_args = TrainingArguments(
            output_dir=f"./ckpt/{args.save_dir}/fold{fold+1}",
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            logging_steps=args.logging_steps,
            metric_for_best_model="eval_custom_score",
            save_strategy="epoch",
            save_total_limit=1,
            report_to="wandb",  # 현재 fold run에 기록
            dataloader_num_workers=0,
            remove_unused_columns=False,
            eval_strategy="epoch",
            load_best_model_at_end=True,
            greater_is_better=True
        )

        trainer = RMSETrainer(
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
        )

        # 학습 실행 (이때 자동으로 train/eval 로그가 현재 fold run에 기록됨)
        trainer.train()
        torch.save(
            trainer.model.state_dict(),
            f"./ckpt/{args.save_dir}/fold{fold+1}/pytorch_model.bin"
        )
        
        fold_run.finish()

if __name__ == "__main__":
    args = get_arguments()
    main(args)