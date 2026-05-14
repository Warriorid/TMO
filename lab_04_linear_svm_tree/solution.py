from pathlib import Path
import json
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42
sns.set_theme(style='whitegrid')

base = Path(__file__).resolve().parents[2]
dataset_path = base / 'courses_current' / 'notebooks' / 'features' / 'data' / 'titanic.csv'
out_dir = Path(__file__).parent / 'artifacts'
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(dataset_path)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

num_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_cols = ['Sex', 'Embarked']

prep = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

models = {
    'logistic_regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    'svm_rbf': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE),
    'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
}

metrics = {}
for name, model in models.items():
    pipe = Pipeline([('prep', prep), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    metrics[name] = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'f1': float(f1_score(y_test, pred)),
    }
    if name == 'decision_tree':
        feature_names = list(pipe.named_steps['prep'].get_feature_names_out())
        tree_model = pipe.named_steps['model']
        imp = pd.Series(tree_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(12)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=imp.values, y=imp.index)
        plt.title('Важность признаков (Decision Tree)')
        plt.tight_layout()
        plt.savefig(out_dir / 'tree_feature_importance.png', dpi=160)
        plt.close()

        rules = export_text(tree_model, feature_names=feature_names)
        (out_dir / 'tree_rules.txt').write_text(rules, encoding='utf-8')

(out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
print('Lab 4 completed. Artifacts in:', out_dir)
