import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, r2_score, mean_squared_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Feature Engineering
for df in [X_train, X_test]:
    df['installment_term_ratio'] = df['installment'] / (df['term'] + 1)
    df['loan_to_revol_bal'] = df['loan_amnt'] / (df['revol_bal'] + 1)
    df['interaction_1'] = df['int_rate'] * df['loan_amnt']
    df['interaction_2'] = df['open_acc'] * df['total_acc']

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Targets
train_targets = {
    'loan_status_binary': y_train['loan_status_binary'],
    'delinq_flag': y_train['delinq_flag'],
    'dti_class': pd.cut(y_train['dti'], bins=[-1, 10, 30, 1e5], labels=[0, 1, 2]).astype(int)
}

test_targets = {
    'loan_status_binary': y_test['loan_status_binary'],
    'delinq_flag': y_test['delinq_flag'],
    'dti_class': pd.cut(y_test['dti'], bins=[-1, 10, 30, 1e5], labels=[0, 1, 2]).astype(int)
}

model = RandomForestClassifier(n_estimators=100, random_state=42)
kf = KFold(n_splits=3, shuffle=True, random_state=42)

predictions = {}

# Loop through each target
for name in train_targets:
    print(f"\n===== Target: {name} =====")
    y_tr = train_targets[name]
    y_te = test_targets[name]

    model.fit(X_train_scaled, y_tr)
    y_pred_test = model.predict(X_test_scaled)
    acc_test = accuracy_score(y_te, y_pred_test)
    print(f"Accuracy (Test): {acc_test:.2%}")
    predictions[name] = int(np.round(np.mean(y_pred_test)))

    if y_tr.nunique() == 2:
        y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
        auc_test = roc_auc_score(y_te, y_proba_test)
        print(f"AUC (Test): {auc_test:.2%}")

    cm = confusion_matrix(y_te, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Hold-Out Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{name}_holdout_confusion_3fold.png")
    plt.close()

    model.fit(X_train_scaled, y_tr)
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title(f"Top 10 Features - {name} (3-Fold CV)")
    plt.tight_layout()
    plt.savefig(f"{name}_feature_importance_combo_3fold.png")
    plt.close()

# ===== CATBOOST REGRESSION =====

# Feature engineering
for df, y in [(X_train, y_train), (X_test, y_test)]:
    df['payment_to_income'] = df['installment'] / (y['annual_inc'] / 12 + 1)
    df['loan_to_income'] = df['loan_amnt'] / (y['annual_inc'] + 1)
    df['credit_util_efficiency'] = y['revol_util'] / (df['open_acc'] + 1)

features_revol = ['int_rate', 'sub_grade', 'grade', 'revol_bal',
                  'installment', 'funded_amnt_inv', 'funded_amnt', 'loan_amnt',
                  'payment_to_income', 'loan_to_income', 'credit_util_efficiency']
features_inc = ['loan_amnt', 'funded_amnt', 'funded_amnt_inv',
                'installment', 'revol_bal', 'total_acc', 'open_acc',
                'payment_to_income', 'loan_to_income']

Xr_train_full = X_train[features_revol]
yr_train = y_train['revol_util']
Xi_train_full = X_train[features_inc]
yi_train = np.log1p(y_train['annual_inc'])

Xr_test_full = X_test[features_revol]
yr_test = y_test['revol_util']
Xi_test_full = X_test[features_inc]
yi_test_log = np.log1p(y_test['annual_inc'])
yi_test_real = y_test['annual_inc']

scaler_r = RobustScaler()
Xr_train_scaled = scaler_r.fit_transform(Xr_train_full)
Xr_test_scaled = scaler_r.transform(Xr_test_full)

scaler_i = RobustScaler()
Xi_train_scaled = scaler_i.fit_transform(Xi_train_full)
Xi_test_scaled = scaler_i.transform(Xi_test_full)

model_r = CatBoostRegressor(verbose=0, random_state=42, iterations=300, learning_rate=0.05)
model_i = CatBoostRegressor(verbose=0, random_state=42, iterations=300, learning_rate=0.05)
model_r.fit(Xr_train_scaled, yr_train)
model_i.fit(Xi_train_scaled, yi_train)

yr_pred_test = model_r.predict(Xr_test_scaled)
yi_pred_test = np.expm1(model_i.predict(Xi_test_scaled))

# Bin predictions to classes (0,1,2)
predictions['revol_util_class'] = int(pd.qcut(yr_pred_test, q=3, labels=[0,1,2]).value_counts().idxmax())
predictions['annual_inc_class'] = int(pd.qcut(yi_pred_test, q=3, labels=[0,1,2]).value_counts().idxmax())

# ===== FINAL SMART DECISION BASED ON 5 FACTORS =====
print(f"Predicted revol_util_class: {predictions['revol_util_class']}")
print(f"Predicted annual_inc_class: {predictions['annual_inc_class']}")


from sklearn.metrics import accuracy_score

print("\n===== Accuracy Summary for ALL 5 features =====")

# revol_util classification accuracy
revol_util_true_class = pd.qcut(yr_test, q=3, labels=[0, 1, 2])
revol_util_pred_class = pd.qcut(yr_pred_test, q=3, labels=[0, 1, 2])
acc_revol_util = accuracy_score(revol_util_true_class, revol_util_pred_class)
print(f"Accuracy (revol_util_class): {acc_revol_util:.2%}")

# annual_inc classification accuracy
annual_inc_true_class = pd.qcut(yi_test_real, q=3, labels=[0, 1, 2])
annual_inc_pred_class = pd.qcut(yi_pred_test, q=3, labels=[0, 1, 2])
acc_annual_inc = accuracy_score(annual_inc_true_class, annual_inc_pred_class)
print(f"Accuracy (annual_inc_class): {acc_annual_inc:.2%}")

# ===== SMART RECOMMENDATION BASED ON 5 FACTORS =====
score = 0
score += 0.35 * (predictions['loan_status_binary'])
score += 0.15 * (1 - predictions['delinq_flag'])
score += 0.10 * (1 - predictions['dti_class'] / 2)
score += 0.20 * (1 - predictions['revol_util_class'] / 2)
score += 0.20 * (predictions['annual_inc_class'] / 2)

print("\n===== SMART RECOMMENDATION BASED ON 5 FACTORS =====")
print(f"Final score (0-1): {score:.2f}")
if score >= 0.75:
    print("RECOMMENDATION: APPROVE the loan")
elif score >= 0.6:
    print(" RECOMMENDATION: MANUAL REVIEW")
else:
    print("RECOMMENDATION: DECLINE the loan")