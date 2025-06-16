# Student-Performance
머신러닝 기말 발표 Student Performance의 분석

1. 레포트 공지란의 기말발표 과제를 통해서 접수받음 (6월 20일 24시까지)
2. 제출은 분석 보고서, 코드, 포스터 파일 세 종임
3. 기말발표 자료 미제출시 F학점이 될 수 있으니 주의 바람

 
기본 목표는 데이터를 선택해서 일반회귀분석, 벌점화 회귀분석(RIDGE & LASSO), GAM, 나무 모형, 부스팅 방법을 적용해 성능을 비교하는 것임 
* 중요: 데이터 크기는 2500 이상 변수는 10개 이상이어야 함(변수 선택에 따른 평가는 25% 비중임) 

1. 분석 보고서에는 아래 내용이 포함되어야 함
    - 표지(이름과 학번 포함)
    - 선택 데이터의 이름과 크기, 변수의 개수(반응변수 및 설명변수로 구분)
    - 기초 통계량 및 간단한 시각화
    - 반드시 한 절로 구성해서 각 방법의 적용 및 결과에 대해서 아래 내용을 설명할 것
      - 각 방법에 대한 대한 간단한 설명과 사용한 설명변수의 처리 방법
      - 각 방법에 따른 최적화 방법 
      - 예측결과 (훈련데이터 7, 평가 데이터 3 내지 훈련 데이터 5, 검증 데이터 2, 평가데이터 3로 쪼개서 평가 예측오차 보고)
      - 평가 예측오차는 하나가 아니라 여러 형태로 보고 가능(MSE, RMSE, Acc. , AUC)
      - 결과에 대한 해석을 위해서 Importance 등을 살펴보는 것을 추천
2. 코드
    - 정리해서 제출하면 되며 형식은 hwp, R, ipynb 다 무방함 (R과 python 중 하나만 내지 둘다 사용해도 좋음)
3. 포스터 
    - 주되게 목표와 데이터 설명, 각 방법에 따른 최종 결과 비교 중심으로 구성하면 되며 견본을 첨부하니 참고바람

['Age', 'Grade', 'Gender', 'Race', 'SES_Quartile', 'ParentalEducation',
   'SchoolType', 'Locale', 'TestScore_Math', 'TestScore_Reading',
   'TestScore_Science', 'GPA', 'AttendanceRate', 'StudyHours',
   'InternetAccess', 'Extracurricular', 'PartTimeJob', 'ParentSupport',
   'Romantic', 'FreeTime', 'GoOut']

일반회귀분석, 벌점화 회귀분석(RIDGE & LASSO), GAM, 나무 모형, 부스팅 

### 보고서 초안
1. 표지
2. 데이터 소개
    - 데이터 간단한 설명
    - 기초 통계량 등 간단한 시각화

3. 실험
    - 반드시 한 절로 구성해서 각 방법의 적용 및 결과에 대해서 아래 내용을 설명할 것
      - 각 방법에 대한 대한 간단한 설명과 사용한 설명변수의 처리 방법
      - 각 방법에 따른 최적화 방법 
      - 예측결과 (훈련데이터 7, 평가 데이터 3 내지 훈련 데이터 5, 검증 데이터 2, 평가데이터 3로 쪼개서 평가 예측오차 보고)
      - 평가 예측오차는 하나가 아니라 여러 형태로 보고 가능(MSE, RMSE, Acc. , AUC)
    
    3.1 인종과 성별은 성적과의 영향은 있는가? 
    미국에서는 대학에서 학생들이 입학 할 때, 인종별 쿼터제를 사용한다. 즉, 미국의 대학에서는 각 인종별로 학업성취도가 다르게 나타난다고 생각한다. 
    우리는 살면서, 남녀의 몇가지 차이가, 남녀가 서로 다른 과목을 잘하는 경향을 보이게 한다는 말을 듣는다. 
    이 실험에서는 위의 두 가정에 대해 실제로 그러한지, 실험하고자 한다.
    입력 칼럼 : 'Gender', 'Race'
    예측 칼럼 : 'TestScore_Math', 'TestScore_Reading', 'TestScore_Science', 'GPA'
    기법 : XGBoosting
    파일명 : race_gender.ipynb

    3.2 집안의 부는 성적에 영향을 미치는가?
    일반적으로, 집안이 부유하다면, 그 학생은 더 쉽게 각종 자료나, 추가적인 교육의 기회를 얻기 쉽다. 이 실험에서는 집안의 부 자체가 얼마나 성적에 영향을 미치는가를분석한다.
    입력 칼럼 : 'SchoolType', 'Locale', 'InternetAccess','SES_Quartile'
    예측 칼럼 :  'GPA'
    기법 : GAM
    파일명 : wealth.ipynb

    3.3 공부 시간과 다른 시간 사이의 비율은 성적에 영향을 미치는가?
    공부를 잘하기 위해서는 당연히 공부를 많이 해야 한다. 하지만, 단순히 계속 공부를 한다고 해서 성적이 잘 나올 것인가? 적절한 휴식 시간도 학업의 성취를 위해서 필요하다.이번 실험을 통해, 공부 시간, 그리고 그 외의 시간의 비율에 따른 학업 성적의 향상 정도를 확인하고자 한다.
    입력 칼럼 : 'FreeTime', 'GoOut', 'StudyHours'
    예측 칼럼 :  'GPA'
    기법 : Linear Regression
    파일명 : time_spent.ipynb

    3.4 공부하는 환경은 얼마나 환경에 영향을 받는가?
    공부환경은 얼마나 그 학생이 공부에 전념하고 집중해 공부를 할 수 있는지를 말해준다. 당연히, 공부 환경이 좋을수록 학생은 더 나은 학업 성취를 얻을 수 있을 것이다. 공부환경이라고 부를 수 있는 것은 간단하지 않다. 학교의 위치, 연애 여부, 부모의 최종학력등 많은 것들이 학생의 공부 환경 조성에 영향을 미친다. 이 실험에서는 이 환경들이각각  얼마나 성적에 영향을 미치는지를 확인하고자 한다.
    입력 칼럼 : SchoolType', 'Locale', 'InternetAccess','SES_Quartile','ParentalEducation','Romantic','Extracurricular','ParentSupport'
    예측 칼럼 :'TestScore_Math', 'TestScore_Reading', 'TestScore_Science', 'GPA'
    기법 : Decision Tree
    파일명 : enviroment.ipynb

    3.5 공부 환경은 인종에 따라 확연하게 차이가 나는가?
    우리는 실험 1에서 인종간의 성적 차이를 분석하였다. 이것이 실제 차이가 나는 이유가 인종 때문인지, 아니면 인종별로 학습 환경이 차이나는지를 확인하기 위해 실험을진행한다.
    입력 칼럼 :  SchoolType', 'Locale', 'InternetAccess','SES_Quartile','ParentalEducation','Romantic','Extracurricular','ParentSupport'
    예측 칼럼 : 'race'
    기법 :  Logistic
    파일명 : enviroment_race.ipynb

    3.6 일반적으로 성실하다고 여겨지는 학생은, 성적도 잘 나오는가?
    일반적으로, 사람들은 성실한 학생은 성적 또한 좋을 것이라고 생각한다. 추가적인 클럽활동, 출석률, 학습 시간을 이용하여 예측을 시도하고, 실제 관계가 있는지 분석한다.
    입력 칼럼 : 'AttendanceRate', 'StudyHours',
    예측 칼럼 : 'GPA'
    기법 : ridge
    파일명 : good_looking.ipynb

    3.7 가장 예측에 강력한 영향을 미치는 핵심 요인은 무엇인가?
    전체 칼럼을 사용하여, 최종적으로 가장 강력한 영향력을 미치는 핵심 요인들의 판별을 시도한다.
    입력 칼럼 : total( 'TestScore_Math', 'TestScore_Reading','TestScore_Science'는 제외)
    예측 칼럼 : 'GPA'
    기법 : lasso
    파일명 : total.ipynb

4. 결론
    그래서 학생들의 성적은 어떤 것들에 영향을 받는가?
    각 실험 결과 ~~~~ 된다면, 인종이 영향을 미친다고 보기 어려움.
    의외로 ~~~는 영향을 적게 미쳤으며, 
    ~~~~~~그래서 결과는 이렇다




5. 포스터
