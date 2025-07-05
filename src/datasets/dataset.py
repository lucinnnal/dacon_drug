import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from datasets import Dataset

# 검색 모듈 경로 추가
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from configs.training_arguements import get_arguments


# Fingerprint Extraction
def get_morgan_fp(mol, fp_bits=2048, radius=2):
    """Morgan Fingerprint 계산"""
    generator = GetMorganGenerator(radius=radius, fpSize=fp_bits)
    fp = generator.GetFingerprint(mol)
    return list(fp)

# Total Feature Extraction
def extract_features(smiles, fp_bits=2048):
    """
    CYP3A4 저해 예측에 특화된 분자 특성 추출

    Parameters:
    - smiles: SMILES 문자열
    - fp_bits: Morgan fingerprint 비트 수

    Returns:
    - List of molecular features optimized for CYP3A4 inhibition prediction
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # 총 특성 수: 기본(6) + fingerprint(fp_bits) + CYP3A4 특화(20) + 추가(8) = 34 + fp_bits
            return [0] * (34 + fp_bits)

        # 1. 기본 약물성 특성 (Drug-likeness)
        basic_features = [
            Descriptors.MolWt(mol),              # 분자량
            Descriptors.MolLogP(mol),            # 지질친화성
            Descriptors.NumHAcceptors(mol),      # 수소결합 받개 수
            Descriptors.NumHDonors(mol),         # 수소결합 주개 수
            Descriptors.TPSA(mol),               # 극성 표면적
            Descriptors.NumRotatableBonds(mol),  # 회전 가능한 결합 수
        ]

        # 2. Morgan Fingerprint (구조적 특징)
        fp_array = get_morgan_fp(mol, fp_bits)

        # 3. CYP3A4 저해와 관련된 특화 특성들
        cyp3a4_features = [
            # 분자 크기 및 형태
            rdMolDescriptors.CalcNumRings(mol),           # 고리 수
            Descriptors.NumAromaticRings(mol),            # 방향족 고리 수
            Descriptors.NumAliphaticRings(mol),           # 지방족 고리 수
            Descriptors.FractionCSP3(mol),                # SP3 탄소 비율

            # 전자적 특성
            Descriptors.MaxEStateIndex(mol),              # 최대 전자상태 지수
            Descriptors.MinEStateIndex(mol),              # 최소 전자상태 지수
            Descriptors.BalabanJ(mol),                    # Balaban J 지수

            # 친수성/소수성 균형
            Crippen.MolLogP(mol),                         # Crippen LogP
            Descriptors.LabuteASA(mol),                   # Labute 접근가능 표면적

            # 분자 복잡성
            Descriptors.BertzCT(mol),                     # Bertz 복잡성 지수
            rdMolDescriptors.CalcNumHeteroatoms(mol),     # 헤테로원자 수

            # CYP 결합 부위 관련 특성
            Descriptors.NumAromaticCarbocycles(mol),      # 방향족 탄소고리 수
            Descriptors.NumAromaticHeterocycles(mol),     # 방향족 헤테로고리 수
            Descriptors.NumSaturatedCarbocycles(mol),     # 포화 탄소고리 수
            Descriptors.NumSaturatedHeterocycles(mol),    # 포화 헤테로고리 수

            # 결합 특성
            Descriptors.NumAromaticRings(mol),            # 방향족성
            Descriptors.fr_benzene(mol),                  # 벤젠 고리 수
            Descriptors.fr_pyridine(mol),                 # 피리딘 고리 수

            # 분자의 입체적 특성
            rdMolDescriptors.CalcNumRotatableBonds(mol),  # 유연성
            Descriptors.Kappa3(mol) if Descriptors.Kappa3(mol) != 0 else 0,  # 분자 형태 지수
        ]

        # 4. 추가 약동학 관련 특성
        additional_features = [
            Lipinski.HeavyAtomCount(mol),        # 중원자 수
            Descriptors.NHOHCount(mol),          # NH, OH 그룹 수
            Descriptors.NOCount(mol),            # N, O 원자 수
            Descriptors.RingCount(mol),          # 총 고리 수
            Descriptors.Chi0v(mol),              # 연결성 지수
            Descriptors.HallKierAlpha(mol),      # Hall-Kier alpha
            Descriptors.PEOE_VSA1(mol),          # PEOE VSA 기술자
            Descriptors.EState_VSA1(mol),        # EState VSA 기술자
        ]

        return basic_features + fp_array + cyp3a4_features + additional_features

    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return [0] * (34 + fp_bits)

# 4. Tokenization + mol_f feature addition
def preprocess(example, tokenizer):
    tokens = tokenizer(example["Canonical_Smiles"], truncation=True, padding="max_length", max_length=128)
    tokens["mol_f"] = example["mol_f"]
    return tokens

def get_dataset(args):
    # Load data
    train_df = pd.read_csv(args.train_file)[['Canonical_Smiles', 'Inhibition']]
    test_df = pd.read_csv(args.test_file)[['ID', 'Canonical_Smiles']]

    # Tokenizer Deinition
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Feature Extraction + Tokenization
    train_df["mol_f"] = train_df["Canonical_Smiles"].apply(extract_features)
    test_df["mol_f"] = test_df["Canonical_Smiles"].apply(extract_features)

    # Train
    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer))
    train_dataset = train_dataset.rename_column("Inhibition", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "mol_f"])

    # Test
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(lambda x: preprocess(x, tokenizer))
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "mol_f"])

    return train_dataset, test_dataset

if __name__ == "__main__":
    args = get_arguments()
    train_dataset, test_dataset = get_dataset(args)
    breakpoint() # debug