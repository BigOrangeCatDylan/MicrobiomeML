import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve
from sklearn.feature_selection import VarianceThreshold
import shap
import joblib
import argparse
import json

# 解析命令行参数
parser = argparse.ArgumentParser(description="Machine learning analysis for GMrepo dataset.")
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
input_file = config.get("input_file", "gmrepo_data/uc_healthy_genus_abundances_matched.csv")
n_estimators = config.get("n_estimators", 100)
max_depth = config.get("max_depth", 10)
feature_selection_threshold = config.get("feature_selection_threshold", 0.01)
scaler_type = config.get("scaler", "MinMaxScaler")
test_size = config.get("test_size", 0.2)
random_state = config.get("random_state", 42)
save_model = config.get("save_model", True)
model_path = config.get("model_path", "rf_model.pkl")
plot_roc = config.get("plot_roc", True)
plot_feature_importance = config.get("plot_feature_importance", True)
plot_shap = config.get("plot_shap", True)

# 设置随机种子
np.random.seed(random_state)


# 1. 加载数据
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"数据加载成功，形状: {df.shape}")
        print("CSV 列名:", df.columns.tolist())
        return df
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
        exit()


# 2. 数据预处理
def preprocess_data(df, feature_selection_threshold, scaler_type):
    X = df.drop(columns=['phenotype', 'run_id'])
    y = df['phenotype']

    X = X.fillna(0)

    if scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "StandardScaler":
        scaler = StandardScaler()
    else:
        raise ValueError(f"不支持的标准化方法: {scaler_type}")
    X_scaled = scaler.fit_transform(X)

    selector = VarianceThreshold(threshold=feature_selection_threshold)
    X_selected = selector.fit_transform(X_scaled)
    print(f"特征从 {X_scaled.shape[1]} 减少到 {X_selected.shape[1]}")

    feature_names = X.columns[selector.get_support()]
    return X_selected, y, feature_names, selector


# 3. 训练模型
def train_model(X_train, y_train, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# 4. 评估模型
def evaluate_model(model, X_test, y_test, plot_roc):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"AUC: {auc:.3f}")
    print(f"F1 分数: {f1:.3f}")
    print("混淆矩阵:")
    print(cm)

    if plot_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC 曲线 (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('ROC 曲线')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')
        plt.close()


# 5. 特征重要性可视化
def plot_feature_importance(model, feature_names, plot):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    top_features = feature_names[indices]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.title("前 20 个重要特征")
        plt.bar(range(20), importances[indices], align='center')
        plt.xticks(range(20), top_features, rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    return top_features


# 6. SHAP 分析
def shap_analysis(model, X_train, X_test, y_test, feature_names, plot_shap):
    if not plot_shap:
        return

    explainer = shap.TreeExplainer(model)
    shap_values_train = explainer.shap_values(X_train)
    shap_values_test = explainer.shap_values(X_test)

    shap_values_train_for_uc = shap_values_train[:, :, 1]
    shap_values_test_for_uc = shap_values_test[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values_test_for_uc, X_test, feature_names=feature_names.tolist(), show=False)
    plt.savefig('shap_summary_plot.png')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values_test_for_uc, X_test, feature_names=feature_names.tolist(), plot_type="bar",
                      show=False)
    plt.savefig('shap_bar_plot.png')
    plt.close()

    uc_indices = np.where(y_test == 1)[0]
    if len(uc_indices) > 0:
        sample_idx = uc_indices[0]
        plt.figure()
        shap.force_plot(explainer.expected_value[1], shap_values_test_for_uc[sample_idx],
                        X_test[sample_idx], feature_names=feature_names.tolist(),
                        matplotlib=True, show=False)
        plt.savefig('shap_force_plot_uc_sample.png')
        plt.close()


# 主函数
def main():
    df = load_data(input_file)

    X, y, feature_names, selector = preprocess_data(df, feature_selection_threshold, scaler_type)
    print("保留的特征名:", feature_names.tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    model = train_model(X_train, y_train, n_estimators, max_depth)

    evaluate_model(model, X_test, y_test, plot_roc)

    top_features = plot_feature_importance(model, feature_names, plot_feature_importance)
    print("前 20 个 ncbi_taxon_id:", top_features)

    try:
        taxonomy = pd.read_csv("ncbi_taxonomy_table.txt.gz", sep="\t")
        taxonomy['ncbi_taxon_id'] = taxonomy['ncbi_taxon_id'].astype(str)
        top_features = [str(x) for x in top_features]
        top_features_names = taxonomy[taxonomy['ncbi_taxon_id'].isin(top_features)][
            ['ncbi_taxon_id', 'scientific_name']]
        print("映射结果:")
        print(top_features_names)
    except FileNotFoundError:
        print("请从 https://gmrepo.org/downloads 下载 ncbi_taxonomy_table.txt.gz")

    shap_analysis(model, X_train, X_test, y_test, feature_names, plot_shap)
    print("SHAP 分析已完成，图表保存为 shap_summary_plot.png、shap_bar_plot.png 和 shap_force_plot_uc_sample.png")

    if save_model:
        joblib.dump(model, model_path)
        print(f"模型已保存为 {model_path}")


if __name__ == "__main__":
    main()