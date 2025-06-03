import pandas as pd
import numpy as np
import os
import argparse
import json

# 解析命令行参数
parser = argparse.ArgumentParser(description="Extract and match samples from GMrepo database.")
parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file.")
args = parser.parse_args()

# 加载配置文件
try:
    with open(args.config, 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"配置文件 {args.config} 未找到，请提供有效的配置文件。")
    exit()

# 从配置文件中获取参数
target_samples_per_class = config.get("target_samples_per_class", 200)
age_min = config.get("age_min", 10)
age_max = config.get("age_max", 35)
bmi_min = config.get("bmi_min", 10.0)
bmi_max = config.get("bmi_max", 50.0)
age_match_range = config.get("age_match_range", 15)
bmi_match_range = config.get("bmi_match_range", 5.0)
min_genera = config.get("min_genera", 5)
phenotype_uc = config.get("phenotype_uc", "Colitis, Ulcerative")
phenotype_health = config.get("phenotype_health", "Health")
experiment_type = config.get("experiment_type", "metagenomics")
output_file = config.get("output_file", "gmrepo_data/uc_healthy_genus_abundances_matched.csv")

# 创建输出目录
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 数据类型定义
dtypes_runs = {
    "checking": str, "project_id": str, "our_project_id": str, "sample_name": str,
    "original_sample_description": str, "curated_sample_description": str, "run_id": str,
    "sample_id": str, "second_sample_id": str, "experiment_type": str, "nr_reads_sequenced": float,
    "instrument_model": str, "disease": str, "phenotype": str, "is_disease_stage_available": str,
    "disease_stage": str, "more": str, "more_info": str, "country": str, "collection_date": str,
    "sex": str, "host_age": float, "diet": str, "longitude": float, "latitude": float,
    "BMI": float, "Recent.Antibiotics.Use": str, "antibiotics_used": str, "Antibiotics.Dose": str,
    "Days.Without.Antibiotics.Use": str
}

dtypes_processed = {
    "uid": str, "accession_id": str, "data_type": str, "tool_used": str,
    "results_version": str, "last_updated": str, "QCStatus": str, "QCMessage": str
}

# 加载元数据
try:
    runs_metadata = pd.read_csv("sample_to_run_info.txt.gz", sep="\t", compression="gzip", dtype=dtypes_runs, low_memory=False)
except FileNotFoundError:
    print("请从 https://gmrepo.org/downloads 下载 sample_to_run_info.txt.gz。")
    exit()

# 筛选样本
runs_metadata["experiment_type"] = runs_metadata["experiment_type"].str.lower()
uc_metadata = runs_metadata[
    (runs_metadata["experiment_type"] == experiment_type) &
    (runs_metadata["phenotype"] == phenotype_uc) &
    (runs_metadata["host_age"].between(age_min, age_max, inclusive="both") | runs_metadata["host_age"].isna()) &
    (runs_metadata["BMI"].isna() | runs_metadata["BMI"].between(bmi_min, bmi_max, inclusive="both"))
]
health_metadata = runs_metadata[
    (runs_metadata["experiment_type"] == experiment_type) &
    (runs_metadata["phenotype"] == phenotype_health) &
    (runs_metadata["host_age"].between(age_min, age_max, inclusive="both") | runs_metadata["host_age"].isna()) &
    (runs_metadata["BMI"].isna() | runs_metadata["BMI"].between(bmi_min, bmi_max, inclusive="both"))
]

uc_metadata = uc_metadata.drop_duplicates(subset=["run_id"])
health_metadata = health_metadata.drop_duplicates(subset=["run_id"])

print(f"筛选后 UC 样本数: {len(uc_metadata)}")
print(f"筛选后健康样本数: {len(health_metadata)}")

# 样本匹配
uc_samples = uc_metadata[["run_id", "host_age", "BMI"]]
health_samples = []
for _, uc_row in uc_samples.iterrows():
    uc_age = uc_row["host_age"]
    uc_bmi = uc_row["BMI"]
    matches = health_metadata[
        (health_metadata["host_age"].isna() | health_metadata["host_age"].between(uc_age - age_match_range, uc_age + age_match_range, inclusive="both") if not pd.isna(uc_age) else health_metadata["host_age"].isna()) &
        (health_metadata["BMI"].isna() |
         (health_metadata["BMI"].between(uc_bmi - bmi_match_range, uc_bmi + bmi_match_range, inclusive="both") if not pd.isna(uc_bmi) else health_metadata["BMI"].between(bmi_min, bmi_max)))
    ]
    if not matches.empty:
        health_samples.append(matches.iloc[0][["run_id", "host_age", "BMI"]])
        health_metadata = health_metadata.drop(matches.index[0])
    if len(health_samples) >= len(uc_samples):
        break

health_samples = pd.DataFrame(health_samples)
print(f"选择 {len(uc_samples)} 个 UC 样本，{len(health_samples)} 个健康样本")

# 加载 samples_loaded 文件
try:
    processed_runs = pd.read_csv("samples_loaded.txt.gz", sep="\t", compression="gzip", dtype=dtypes_processed)
except FileNotFoundError:
    print("请从 https://gmrepo.org/downloads 下载 samples_loaded.txt.gz。")
    exit()
except pd.errors.ParserError:
    print("解析 samples_loaded.txt.gz 出错，请检查文件格式或列名。")
    exit()

if 'uid' not in processed_runs.columns or 'accession_id' not in processed_runs.columns:
    print("samples_loaded.txt.gz 中缺少 'uid' 或 'accession_id' 列，可用列名:", processed_runs.columns.tolist())
    exit()

# 加载丰度数据
try:
    abundance_data = pd.read_csv("species_abundance.txt.gz", sep="\t", compression="gzip")
except FileNotFoundError:
    print("请从 https://gmrepo.org/downloads 下载 species_abundance.txt.gz。")
    exit()

# 统一数据类型
abundance_data["loaded_uid"] = abundance_data["loaded_uid"].astype(str)
processed_runs["uid"] = processed_runs["uid"].astype(str)
processed_runs["accession_id"] = processed_runs["accession_id相对于str"]

# 映射 loaded_uid 到 run_id
abundance_with_run = pd.merge(abundance_data, processed_runs, left_on="loaded_uid", right_on="uid", how="inner")
print(f"合并后的丰度数据行数: {len(abundance_with_run)}")
genus_data = abundance_with_run[abundance_with_run["taxon_rank_level"] == "genus"]

# 提取目标运行的属水平数据
target_run_ids = set(uc_samples["run_id"].tolist() + health_samples["run_id"].tolist())
genus_data = genus_data[genus_data["accession_id"].isin(target_run_ids)]

# 处理属水平丰度
all_data = []
failed_run_ids = []
for run_id in target_run_ids:
    run_genus_data = genus_data[genus_data["accession_id"] == run_id]
    if run_genus_data.empty:
        failed_run_ids.append(run_id)
        continue
    if len(run_genus_data) < min_genera:
        print(f"Run ID {run_id} 的属数量少于 {min_genera}")
        failed_run_ids.append(run_id)
        continue
    genus_abundances = {"run_id": run_id}
    for _, row in run_genus_data.iterrows():
        genus_abundances[row["ncbi_taxon_id"]] = row["relative_abundance"]
    genus_abundances["phenotype"] = 1 if run_id in uc_samples["run_id"].values else 0
    age = uc_samples[uc_samples["run_id"] == run_id]["host_age"].iloc[0] if run_id in uc_samples["run_id"].values else health_samples[health_samples["run_id"] == run_id]["host_age"].iloc[0]
    bmi = uc_samples[uc_samples["run_id"] == run_id]["BMI"].iloc[0] if run_id in uc_samples["run_id"].values else health_samples[health_samples["run_id"] == run_id]["BMI"].iloc[0]
    genus_abundances["age"] = age
    genus_abundances["bmi"] = bmi
    all_data.append(genus_abundances)

# 保存失败的 Run IDs
with open(os.path.join(os.path.dirname(output_file), "failed_run_ids.txt"), "w") as f:
    f.write("\n".join(failed_run_ids))
print(f"失败的 Run IDs 已保存至 {os.path.join(os.path.dirname(output_file), 'failed_run_ids.txt')}")

# 整合数据
df = pd.DataFrame(all_data)
if len(df) == 0:
    print("未检索到有效样本，请检查 failed_run_ids.txt。")
    print("下载项目数据: https://gmrepo.org/Downloads/RunsByProjectID/")
    print("或联系 GMrepo 支持: https://gmrepo.org/contact")
    exit()

df.fillna(0, inplace=True)

genus_columns = [col for col in df.columns if col not in ["run_id", "phenotype", "age", "bmi"]]
df = df[df[genus_columns].gt(0).sum(axis=1) >= min_genera]

if len(df) > target_samples_per_class * 2:
    uc_df = df[df["phenotype"] == 1].head(target_samples_per_class)
    health_df = df[df["phenotype"] == 0].head(target_samples_per_class)
    df = pd.concat([uc_df, health_df])

df.to_csv(output_file, index=False)
print(f"数据已保存至 {output_file}")
print(f"总样本数: {len(df)} (UC: {len(df[df['phenotype'] == 1])}, 健康: {len(df[df['phenotype'] == 0])})")