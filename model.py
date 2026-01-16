from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import seaborn as sns
from scipy.stats import uniform, randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

features = [
    'online_order',
    'book_table', 
    'votes',
    'approx_cost',
    'location',
    'rest_type',
    'cuisines_count',
    'has_dish_liked',
    'listed_in(type)',
    'listed_in(city)'
]

# ползваме log трансформация за votes и cost

data = pd.read_csv('rest_cleaned.csv')
X = data[features].copy()
X['log_votes'] = np.log1p(data['votes'])
X['log_cost'] = np.log1p(data['approx_cost'])
y = data['high_rating']
X = X.drop(['votes', 'approx_cost'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# Pipeline
numeric_features = ['log_votes', 'log_cost', 'cuisines_count']
categorical_features = ['location', 'rest_type', 'listed_in(type)', 'listed_in(city)']
binary_features = ['online_order', 'book_table', 'has_dish_liked']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('binary', 'passthrough', binary_features)
])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'SVC': SVC(probability=True),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Параметри за туниране
param_distributions = {
    'Logistic Regression': {
        'model__C': [0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l2', 'none'],
        'model__solver': ['lbfgs', 'saga']
    },
    
    'SVC': {
        'model__C': [0.5, 1, 2, 3, 5],
        'model__kernel': ['rbf'],
        'model__gamma': ['scale']
    },
    
    'Decision Tree': {
        'model__max_depth': [10, 20, 30, 40, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4, 8]
    },
    
    'Random Forest': {
        'model__n_estimators': [300, 500, 800, 1000],
        'model__max_depth': [None, 20, 40, 60],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
        'model__max_features': ['sqrt', 'log2']
    },
    
    'Gradient Boosting': {                     # ← най-важният
        'model__n_estimators': [300, 500, 800, 1000, 1200],
        'model__learning_rate': [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
        'model__max_depth': [3, 4, 5, 6, 7],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4],
        'model__subsample': [0.8, 0.9, 1.0],
        'model__max_features': ['sqrt', 'log2', None]
    },
    
    'AdaBoost': {
        'model__n_estimators': [200, 500, 800, 1000],
        'model__learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5]
    }
}
results = []
best_models = {}   # ← запазваме най-добрия модел за всеки тип

for name, model in models.items():
    print(f"\n=== Туниране на {name} ===")
    
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Ако има param_distributions – RandomizedSearchCV, иначе просто fit
    if name in param_distributions:
        random_search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions[name],
            n_iter=60,                    # 60 случайни комбинации – перфектно и бързо
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
        print(f"Най-добър ROC AUC (CV): {random_search.best_score_:.4f}")
        print(f"Най-добри параметри: {random_search.best_params_}")
    else:
        best_model = pipe.fit(X_train, y_train)
        print("Без туниране на хиперпараметри")
    
    # Запазваме най-добрия модел
    best_models[name] = best_model
    
    # Предвиждане на test сета
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Метрики
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba)
    }
    
    results.append(metrics)
    print(classification_report(y_test, y_pred))
# Резултати в DataFrame
results_df = pd.DataFrame(results).round(4)
results_df

results_melt = results_df.melt(id_vars='Model', var_name='Metric', value_name='Score')

plt.figure(figsize=(14,8))
sns.barplot(x='Model', y='Score', hue='Metric', data=results_melt)
plt.title('Сравнение на 6-те модела по 5 метрики')
plt.xticks(rotation=90)
plt.ylim(0.7, 0.95)
plt.legend(loc='upper roght')
plt.tight_layout()
plt.show()