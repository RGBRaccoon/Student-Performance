import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# GPU 사용 가능 여부 확인
print("XGBoost 버전:", xgb.__version__)
print("GPU 지원 여부 확인 중...")
try:
    test_model = xgb.XGBClassifier(tree_method='gpu_hist', n_estimators=1)
    print("✅ GPU 가속 사용 가능!")
    gpu_available = True
except Exception as e:
    print("❌ GPU 가속 사용 불가능:", str(e))
    print("CPU 모드로 실행됩니다.")
    gpu_available = False

# 데이터 로드
print("\n데이터 로드 중...")
train = pd.read_csv('archive/train.csv')
test = pd.read_csv('archive/test.csv')

# 특성 선택
grade_features = ['GPA', 'TestScore_Math', 'TestScore_Reading', 'TestScore_Science']

# 범주형 변수 인코딩
le_race = LabelEncoder()
le_gender = LabelEncoder()

y_race_train = le_race.fit_transform(train['Race'])
y_race_test = le_race.transform(test['Race'])
x_train = train[grade_features]

y_gender_train = le_gender.fit_transform(train['Gender'])
y_gender_test = le_gender.transform(test['Gender'])
x_test = test[grade_features]

# GPU 가속 설정
tree_method = 'gpu_hist' if gpu_available else 'hist'
print(f"\n사용 중인 트리 메소드: {tree_method}")

# 최적화된 하이퍼파라미터 분포 (RandomizedSearchCV용)
param_distributions = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
    'n_estimators': [50, 100, 150, 200],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

# 인종 예측 모델 (다중 분류)
race_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    tree_method=tree_method,
    random_state=42
)

# 성별 예측 모델 (이진 분류)
gender_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    tree_method=tree_method,
    random_state=42
)

print("\n=== 최적화된 XGBoost 튜닝 시작 ===")
print("RandomizedSearchCV 사용 (GridSearchCV보다 빠름)")
print("n_iter=20으로 제한하여 빠른 튜닝")

# 인종 예측 모델 튜닝 (RandomizedSearchCV)
print("\n인종 예측 모델 튜닝 중...")
start_time = time.time()
race_random = RandomizedSearchCV(
    race_model, 
    param_distributions, 
    n_iter=20,  # 20개 조합만 테스트
    cv=3,       # 3-fold CV (5-fold 대신)
    scoring='f1_weighted',
    random_state=42,
    n_jobs=-1   # 모든 CPU 코어 사용
)
race_random.fit(x_train, y_race_train)
race_time = time.time() - start_time
print(f"인종 예측 모델 튜닝 완료 (소요시간: {race_time:.2f}초)")

# 성별 예측 모델 튜닝 (RandomizedSearchCV)
print("\n성별 예측 모델 튜닝 중...")
start_time = time.time()
gender_random = RandomizedSearchCV(
    gender_model, 
    param_distributions, 
    n_iter=20,  # 20개 조합만 테스트
    cv=3,       # 3-fold CV
    scoring='f1',
    random_state=42,
    n_jobs=-1   # 모든 CPU 코어 사용
)
gender_random.fit(x_train, y_gender_train)
gender_time = time.time() - start_time
print(f"성별 예측 모델 튜닝 완료 (소요시간: {gender_time:.2f}초)")

# 최적 모델 선택
best_race_model = race_random.best_estimator_
best_gender_model = gender_random.best_estimator_

print(f"\n총 튜닝 시간: {race_time + gender_time:.2f}초")

# 모델 평가
print("\n=== 모델 평가 ===")
# 인종 예측 평가
race_pred = best_race_model.predict(x_test)
race_pred_proba = best_race_model.predict_proba(x_test)

print("\n인종 예측 모델 성능:")
print(classification_report(y_race_test, race_pred, target_names=le_race.classes_))

# 성별 예측 평가
gender_pred = best_gender_model.predict(x_test)
gender_pred_proba = best_gender_model.predict_proba(x_test)

print("\n성별 예측 모델 성능:")
print(classification_report(y_gender_test, gender_pred, target_names=le_gender.classes_))

# 시각화
print("\n시각화 생성 중...")

# 특성 중요도 플롯
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
xgb.plot_importance(best_race_model, title='인종 예측 특성 중요도')
plt.subplot(1, 2, 2)
xgb.plot_importance(best_gender_model, title='성별 예측 특성 중요도')
plt.tight_layout()
plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

# 혼동 행렬
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_race_test, race_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=le_race.classes_,
            yticklabels=le_race.classes_)
plt.title('인종 예측 혼동 행렬')
plt.xlabel('예측')
plt.ylabel('실제')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_gender_test, gender_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=le_gender.classes_,
            yticklabels=le_gender.classes_)
plt.title('성별 예측 혼동 행렬')
plt.xlabel('예측')
plt.ylabel('실제')
plt.tight_layout()
plt.savefig('confusion_matrix_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC 커브 (성별 예측의 경우)
fpr, tpr, _ = roc_curve(y_gender_test, gender_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('성별 예측 ROC 커브')
plt.legend(loc="lower right")
plt.savefig('roc_curve_optimized.png', dpi=300, bbox_inches='tight')
plt.close()

# 결과 저장
print("\n결과 저장 중...")
results = {
    'race_model': {
        'best_params': race_random.best_params_,
        'best_score': race_random.best_score_,
        'training_time': race_time,
        'feature_importance': dict(zip(grade_features, best_race_model.feature_importances_))
    },
    'gender_model': {
        'best_params': gender_random.best_params_,
        'best_score': gender_random.best_score_,
        'training_time': gender_time,
        'feature_importance': dict(zip(grade_features, best_gender_model.feature_importances_))
    }
}

# 결과를 DataFrame으로 변환하여 저장
race_importance = pd.DataFrame({
    'Feature': grade_features,
    'Importance': best_race_model.feature_importances_
}).sort_values('Importance', ascending=False)

gender_importance = pd.DataFrame({
    'Feature': grade_features,
    'Importance': best_gender_model.feature_importances_
}).sort_values('Importance', ascending=False)

race_importance.to_csv('race_prediction_importance_optimized.csv', index=False)
gender_importance.to_csv('gender_prediction_importance_optimized.csv', index=False)

# 최종 결과 출력
print("\n=== 최적화된 분석 결과 ===")
print(f"GPU 사용 여부: {'예' if gpu_available else '아니오'}")
print(f"트리 메소드: {tree_method}")
print(f"튜닝 방법: RandomizedSearchCV (n_iter=20)")
print(f"교차 검증: 3-fold CV")

print("\n1. 인종 예측 모델")
print(f"최적 하이퍼파라미터: {race_random.best_params_}")
print(f"최적 교차 검증 점수: {race_random.best_score_:.3f}")
print(f"학습 소요시간: {race_time:.2f}초")
print("\n특성 중요도:")
print(race_importance)

print("\n2. 성별 예측 모델")
print(f"최적 하이퍼파라미터: {gender_random.best_params_}")
print(f"최적 교차 검증 점수: {gender_random.best_score_:.3f}")
print(f"학습 소요시간: {gender_time:.2f}초")
print("\n특성 중요도:")
print(gender_importance)

print(f"\n총 학습 시간: {race_time + gender_time:.2f}초")
print("모든 결과가 '_optimized' 접미사가 붙은 파일로 저장되었습니다.") 