import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# GPA를 등급으로 변환하는 함수
def convert_gpa_to_grade(gpa):
    if gpa >= 3.5:
        return "A"
    elif gpa >= 3.0:
        return "B"
    elif gpa >= 2.5:
        return "C"
    elif gpa >= 2.0:
        return "D"
    else:
        return "F"


# 데이터 로드
train_data = pd.read_csv("archive/train.csv")
validation_data = pd.read_csv("archive/validation.csv")
test_data = pd.read_csv("archive/test.csv")

# GPA 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(data=train_data, x="GPA", bins=30)
plt.title("GPA Distribution")
plt.savefig("gpa_distribution.png")
plt.close()

# GPA를 등급으로 변환
train_data["Grade_Level"] = train_data["GPA"].apply(convert_gpa_to_grade)
validation_data["Grade_Level"] = validation_data["GPA"].apply(convert_gpa_to_grade)
test_data["Grade_Level"] = test_data["GPA"].apply(convert_gpa_to_grade)

# 등급 분포 확인
grade_dist = train_data["Grade_Level"].value_counts()
plt.figure(figsize=(10, 6))
grade_dist.plot(kind="bar")
plt.title("Grade Level Distribution")
plt.xlabel("Grade Level")
plt.ylabel("Count")
plt.savefig("grade_distribution.png")
plt.close()

# 특성과 타겟 설정
features = ["Gender", "Race", "Grade"]
target = "Grade_Level"

# 데이터 전처리
train_x = pd.get_dummies(train_data[features])
train_y = train_data[target]
val_x = pd.get_dummies(validation_data[features])
val_y = validation_data[target]
test_x = pd.get_dummies(test_data[features])
test_y = test_data[target]

# 로지스틱 회귀 모델 학습
logistic_model = LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial")
logistic_model.fit(train_x, train_y)

# 예측
train_pred = logistic_model.predict(train_x)
val_pred = logistic_model.predict(val_x)
test_pred = logistic_model.predict(test_x)

# 결과 출력
print("\n=== 훈련 데이터 성능 ===")
print(classification_report(train_y, train_pred))
print("\n=== 검증 데이터 성능 ===")
print(classification_report(val_y, val_pred))
print("\n=== 테스트 데이터 성능 ===")
print(classification_report(test_y, test_pred))

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
cm = confusion_matrix(test_y, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")
plt.close()

# 특성 중요도 시각화
feature_importance = pd.DataFrame(
    {"Feature": train_x.columns, "Importance": np.abs(logistic_model.coef_[0])}  # 첫 번째 클래스에 대한 계수 사용
)
feature_importance = feature_importance.sort_values("Importance", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance, x="Importance", y="Feature")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
