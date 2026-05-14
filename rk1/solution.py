from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_theme(style='whitegrid')

out_dir = Path(__file__).parent / 'artifacts'
out_dir.mkdir(exist_ok=True)

# Датасет №6 из задания
url = 'https://raw.githubusercontent.com/hamzanasirr/Predicting-Graduate-Admissions-using-Machine-Learning-in-Python/master/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(url)
df.columns = [c.strip() for c in df.columns]

raw_path = out_dir / 'Admission_Predict_Ver1.1.csv'
df.to_csv(raw_path, index=False)

# По условию при наличии пропусков удаляем строки/колонки с пропусками
missing_cols = df.columns[df.isna().any()].tolist()
clean_df = df.dropna(axis=0)

# Удаляем технический идентификатор, если присутствует
if 'Serial No.' in clean_df.columns:
    clean_df = clean_df.drop(columns=['Serial No.'])

corr = clean_df.corr(numeric_only=True)
corr.to_csv(out_dir / 'correlation_matrix.csv')

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Корреляционная матрица (RK1, вариант 6)')
plt.tight_layout()
plt.savefig(out_dir / 'correlation_heatmap.png', dpi=170)
plt.close()

target_col = 'Chance of Admit '
if target_col not in clean_df.columns and 'Chance of Admit' in clean_df.columns:
    target_col = 'Chance of Admit'

abs_corr_target = corr[target_col].drop(labels=[target_col]).abs().sort_values(ascending=False)

report = {
    'dataset_shape_before': list(df.shape),
    'dataset_shape_after_dropna': list(clean_df.shape),
    'columns_with_missing': missing_cols,
    'target_column': target_col,
    'top_features_by_abs_corr_with_target': abs_corr_target.head(5).to_dict(),
}

(out_dir / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

with open(out_dir / 'conclusions.txt', 'w', encoding='utf-8') as f:
    f.write('1) Наиболее связанные с целевой переменной признаки имеют наибольший смысл для первичной модели.\n')
    f.write('2) При сильной межпризнаковой корреляции линейные модели могут требовать регуляризации.\n')
    f.write('3) Для оценки вклада признаков разумно сравнить линейные и ансамблевые модели (RandomForest/GB).\n')

print('RK1 completed. Artifacts in:', out_dir)
