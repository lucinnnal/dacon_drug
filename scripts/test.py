import os
import pandas as pd
import torch
from transformers import TrainingArguments

from configs.training_arguements import get_arguments
from src.data.dataset_bert import get_dataset
from src.models.get_model import get_model
from utils.compute_metrics import compute_metrics
from trainer import RMSETrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

args = get_arguments()
train_dataset, test_dataset = get_dataset()

training_args = TrainingArguments(
    output_dir="./inference_tmp",
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    report_to="none",
    dataloader_num_workers=4,
    remove_unused_columns=False,
    save_strategy="no",
    evaluation_strategy="no",
    logging_strategy="no"
)

# 1. 커스텀 ChemBERT 모델 객체 생성 (내부에서 AutoModelForSequenceClassification.from_pretrained() 호출됨)
model = get_model(args)

# 2. 저장한 파인튜닝 가중치를 명시적으로 덮어쓰기 (load_state_dict)
for i in range(5):
    ckpt_path = f"/home/urp_jwl/.vscode-server/data/drug/scripts/ckpt/p_DeepChem_loss/fold{i+1}/pytorch_model.bin"
    state_dict = torch.load(ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    # 강제로 덮어쓰기
    model.load_state_dict(state_dict)

    # 이후 Trainer 등 나머지 코드 진행
    trainer = RMSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        compute_metrics=compute_metrics,
    )

    print(f"\n=== Final Prediction with Best Model ===")
    preds = trainer.predict(test_dataset).predictions.squeeze()

    submit = pd.read_csv(args.submission_file)
    submit["Inhibition"] = preds
    submit.to_csv(f"{args.base_file}/fold{i+1}.csv", index=False)
    print(f"Submission file saved to {args.base_file}/fold{i+1}.csv")
