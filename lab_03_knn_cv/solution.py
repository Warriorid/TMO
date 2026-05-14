from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

baseline = Pipeline([('prep', preprocessor), ('model', KNeighborsClassifier(n_neighbors=5))])
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

param_grid = {
    'model__n_neighbors': list(range(2, 31)),
    'model__weights': ['uniform', 'distance'],
    'model__p': [1, 2],
}

cv1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RANDOM_STATE)

grid = GridSearchCV(Pipeline([('prep', preprocessor), ('model', KNeighborsClassifier())]), param_grid, cv=cv1, scoring='f1', n_jobs=1)
grid.fit(X_train, y_train)

rnd = RandomizedSearchCV(
    Pipeline([('prep', preprocessor), ('model', KNeighborsClassifier())]),
    param_distributions=param_grid,
    n_iter=20,
    cv=cv2,
    scoring='f1',
    random_state=RANDOM_STATE,
    n_jobs=1,
)
rnd.fit(X_train, y_train)

best_grid_pred = grid.best_estimator_.predict(X_test)
best_rnd_pred = rnd.best_estimator_.predict(X_test)

results = {
    'baseline': {
        'accuracy': float(accuracy_score(y_test, y_pred_base)),
        'f1': float(f1_score(y_test, y_pred_base)),
    },
    'grid_search': {
        'best_params': grid.best_params_,
        'accuracy': float(accuracy_score(y_test, best_grid_pred)),
        'f1': float(f1_score(y_test, best_grid_pred)),
    },
    'randomized_search': {
        'best_params': rnd.best_params_,
        'accuracy': float(accuracy_score(y_test, best_rnd_pred)),
        'f1': float(f1_score(y_test, best_rnd_pred)),
    },
}

(out_dir / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
print('Lab 3 completed. Artifacts in:', out_dir)
