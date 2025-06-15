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
    3.1 실험 목록
    - 인종에 따른 성적('Gender', 'Race','Grade')
        성적으로 인종과 성별을 예측시도하기(classification)
        인종과 성별에따라, 성적이 달라지는가?

    - 공부시간과 일하는 시간에 따는 성적('FreeTime', 'GoOut', 'StudyHours')
        공부시간과, 다른시간 사이의 비율이 성적에 영향을 미치는가?

    - 공부하는 환경('ParentalEducation','SchoolType', 'Locale')
        공부하는 환경은 어떤 영향을 미치는가.
        홈스쿨링과 학교교육의 차이와 지역에 따른 차이가 존재하는가 


    - 성실해보이는 학생은 공부를 잘하는가?
        ('Extracurricular','AttendanceRate', 'StudyHours')
        일반적으로 성실할 것으로 기대되는 학생은, 공부를 잘하는가?

    - 공부외 다른것에 시간이 투자되는 학생은 공부를 못하는가?
        ('Extracurricular','Romantic','GoOut')

    - 돈이 역할을 많이 하는가?
        ('ParentSupport','SchoolType'?, 'Locale','InternetAccess')
        집에 돈이 많은 것이 성적에 영향을 미치는가?