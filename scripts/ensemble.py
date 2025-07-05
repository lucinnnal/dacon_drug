# 5개의 csv 에서 Inhibition 컬럼을 평균내는 코드
import pandas as pd
import os

# 5개의 csv 파일 경로
csv_files = [
    "/home/urp_jwl/.vscode-server/data/drug/src/data/data/fold1.csv",
    "/home/urp_jwl/.vscode-server/data/drug/src/data/data/fold2.csv",
    "/home/urp_jwl/.vscode-server/data/drug/src/data/data/fold3.csv",
    "/home/urp_jwl/.vscode-server/data/drug/src/data/data/fold4.csv",
    "/home/urp_jwl/.vscode-server/data/drug/src/data/data/fold5.csv"
]

# Inhibition 컬럼을 평균내기 위한 DataFrame 리스트
dfs = []

for csv_file in csv_files:
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        dfs.append(df)

# 모든 DataFrame을 하나로 합치고 Inhibition 컬럼의 평균을 계산
if dfs:
    combined = pd.concat(dfs)
    submission = combined.groupby("ID").agg({"Inhibition": "mean"}).reset_index()
    submission.to_csv("/home/urp_jwl/.vscode-server/data/drug/ensemble_submission.csv", index=False)
    print("Ensemble submission file saved to /home/urp_jwl/.vscode-server/data/drug/ensemble_submission.csv")