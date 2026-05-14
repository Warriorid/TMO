from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42

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
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

models = {
    'bagging_dt': BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=4, random_state=RANDOM_STATE), n_estimators=200, random_state=RANDOM_STATE),
    'random_forest': RandomForestClassifier(n_estimators=300, max_depth=6, random_state=RANDOM_STATE),
    'adaboost': AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    'gradient_boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('prep', prep), ('model', model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'f1': float(f1_score(y_test, pred)),
    }

(out_dir / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print('Lab 5 completed. Artifacts in:', out_dir)
