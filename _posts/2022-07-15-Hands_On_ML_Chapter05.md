---
title: "Kotlin 겅부"
excerpt: "Kotlin in action"

categories:
- machine-learning
tags:
- [HandsOnML], [SVM]
use_math: true

permalink: /categories/machine-learning/HandsOn

toc: true
toc_sticky: true

date: 2022-07-15
last_modified_at: 2022-07-15
---

# Chapter 05. Support Vector Machine(SVM)

- `SVM`은 선형이나 비선형 분류, 회귀, 이상치 탐색에도 사용할 수 있는 다목적 머신러닝 모델

## 5.1 Linear SVM classification

- `Large margin classification(라지 마진 분류)`
    
    ![스크린샷 2022-07-07 14.35.11.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4ba8a7e5-9409-435b-9f9e-8c61fbbfc095/스크린샷_2022-07-07_14.35.11.png)
    
    - SVM 분류기를 클래스 사이에 가장 폭이 넓은 도로를 찾는 것으로 생각 가능
    - 도로 바깥쪽에 훈련 샘플을 더 추가해도 decision boundary에 영향을 미치지 않음
    - decision boundary에 위치한 샘플에 의해 결정되는데 이를 `support vector` 라고 함
- SVM은 feature의 scale에 민감
    
    ![스크린샷 2022-07-07 18.10.55.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/edd2df9b-aed1-4642-b081-ce74bde0308c/스크린샷_2022-07-07_18.10.55.png)
    

### 5.1.1 Soft Margin Classification

- `hard margin classification`
    - 모든 sample이 street의 바깥쪽에 올바르게 분류되어 있는 경우
    - 문제점
        - 데이터가 선형적으로 구분될 수 있어야 제대로 작동
        - 이상치에 민감
            
            ![스크린샷 2022-07-07 19.56.01.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46e3d56e-ce3a-4e2f-b6cd-610880fb8b47/스크린샷_2022-07-07_19.56.01.png)
            
            - 왼쪽 그래프에서는 hard margin을 찾을 수 없음.
- `soft margin classification`
    - street의 margin을 가능한 넓게 유지하는 것과 `margin violation`(sample이 street의 middle이나 wrong side에 있는 경우) 사이에 적절한 균형을 잡아야 됨
- Scikit-Learn’s SVM class는 여러 hyperparameter를 지정할 수 있는데 이중 C라는 것이 있음
    
    ![스크린샷 2022-07-07 21.31.55.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4bbeb621-8a07-4fa9-99e1-ce815df5de6b/스크린샷_2022-07-07_21.31.55.png)
    
    - SVM model이 과대적합이라면 C를 감소시켜 모델을 규제할 수 있음
- SVC class를 사용하려고할 때
    
    ```python
    SVC(kernel = "linear", C = 1)
    ```
    
    라고 쓰면 됨.
    
    하지만 large training sets의 경우 매우 느리기 때문에 추천하지 않음
    
    다른 option은 SGDClassifier class를 사용하는 것인데
    
    ```python
    SGDClassifier(loss = "hinge", alpha = 1/(m*C)) # m: # of sample 
    ```
    
    이는 SVM classifier를 train시키기 위해 SGD를 사용함
    
    LinearSVC class보다 빠르게 수렴하진 않지만 huge datasets이기 때문에 memory에서 다룰 수 없거나 온라인 학습으로 classifier task를 다루기는 좋음
    

LinearSVC class는 규제에 편향을 포함시켜 (훈련세트) - (평균)을 하여 training set을 center에 맞춰야 됨

## 5.2 **Nonlinear SVM Classification**

- linearly separable할 수 없는 데이터셋을 처리하는 방법은 nonlinear datasets에 add more features를 하면 됨(polynomial features같은) → linearly separable한 dataset을 만들 수 있음

![스크린샷 2022-07-07 22.07.22.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/18f62dfd-a73e-4dcd-8b73-aa530d8998b1/스크린샷_2022-07-07_22.07.22.png)

- 선형적으로 볼 경우 이 dataset은 구분할 수 없음
- second feature ($x_2 = {x_1}^2$)를 추가한다면 오른쪽처럼 2D dataset은 완벽히 선형적으로 구분할 수 있음

```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
    polynomial_svm_clf = Pipeline([
            ("poly_features", PolynomialFeatures(degree=3)),
            ("scaler", StandardScaler()),
            ("svm_clf", LinearSVC(C=10, loss="hinge"))
        ])
    polynomial_svm_clf.fit(X, y)
```

![스크린샷 2022-07-09 14.48.57.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9579ea6-97e0-4d6e-ae49-c0d35dc5cbe3/스크린샷_2022-07-09_14.48.57.png)

### 5.2.1 Polynomial Kernel

- polynomial features들을 추가하는 것은 간단하고 모든 ML algorithm에서 잘 작동하지만, low polynomial degree는 very complex한 datasets을 잘 표현하지 못하고, 높은 차수의 다항식은 a huge number of features를 포함하므로 모델을 too slow하게 만든다.
- SVM에서는 `kernel trick` 이라는 것을 사용할 수 있다.
    - kernel trick이란, 실제론 어떤 feature를 추가하지 않기 때문에 combinatorial explosion of the number of features(엄청난 수의 특성 조합)가 생기지 않는다.

```python
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)

plt.figure(figsize=(11, 4))

plt.subplot(121)
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)

plt.subplot(122)
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)

save_fig("moons_kernelized_polynomial_svc_plot")
plt.show()
```

![스크린샷 2022-07-09 15.48.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9c773962-3e95-4294-9c55-b7726d3d8291/스크린샷_2022-07-09_15.48.53.png)

- d: 차수
- 과대적합이라면 차수를 줄이고, 과소적합이라면 차수를 늘려야 됨
- coef0: model이 high-degree polynomial과 low-degree polynomial에 얼마나 영향을 받을지 control하는 역할

→ 적절한 hyperparameter를 찾기 위해선 grid search를 하면 됨
   (처음엔 폭을 크게 하여 빠르게 검색후, 최적의 값을 찾기 위해 grid를 세밀하게 검색)

### 5.2.2 **Adding Similarity Features**

- 비선형 feature를 다루는 또 다른 기법은 각 sample이 특정 `landmark`와 얼마나 닮았는지 측정하는 `similarity function`으로 계산한 feature를 추가하는 것

![스크린샷 2022-07-09 17.00.22.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f7c77559-245c-419e-9fbb-8ede0ded590d/스크린샷_2022-07-09_17.00.22.png)

- 앞의 1차원 dataset에 $x_1 = -2,\ x_2 = -1$을 추가(왼쪽 그래프)
- $\gamma = 0.3$인 가우시안 `radial basis function(RBF, 방사 기저 함수)`를 유사도 함수로 정의
    
    ![스크린샷 2022-07-09 17.09.00.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f2f30fff-ab02-4648-bac0-512e2d808015/스크린샷_2022-07-09_17.09.00.png)
    
    - 이 함수 값은 0(landmark에서 아주 멀리 떨어진 경우)부터 1(landmark와 같은 위치에 있는 경우)까지 변화하여 bell-shaped function임
    - 위의 샘플은 첫 번째 landmark에서 1만큼, 두 번째 landmark에서 2만큼 떨어져있음
    
    → 새로 만든 feature는 $x_2 = exp(-0.3\times1^2) \approx 0.74$ 와 $x_3 = exp(-0.3\times2^2) \approx 0.30$ 
    
    이렇게 오른쪽의 graph처러 linearly separable이 가능해짐
    
    - how to select the landmarks
        - simplest approach: create landmark at the location of each and every instance in the datatset(데이터셋에 있는 모든 샘플 위치에 랜드마크를 설정)
        - dimension 커짐 → transformed training set이 linearly separable해질 chance가 증가
        - downside(단점): training set에 있는 n개의 feature를 가진 m개의 sample이 m개의 feature를 가진 m개의 sample로 변환됨(original feature는 drop한다고 가정)
        만약, training set이 매우 클 경우, 아주 많은 feature가 생성됨

### 5.2.3 Gaussian RBF Kernel

- additional feature 계산은 cost가 많이 발생, especially large training set같은 경우 더 그럼
- kernel trick을 한다면 many similarity features를 add하는 것과 similar한 result를 얻을 수 있음

```python
rbf_kernel_svm_clf = Pipeline([ ("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)) ])
rbf_kernel_svm_clf.fit(X, y)
```

![스크린샷 2022-07-09 18.38.16.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/012f6897-078a-4709-848e-8a25b205a671/스크린샷_2022-07-09_18.38.16.png)

```python
from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)

plt.figure(figsize=(11, 7))

for i, svm_clf in enumerate(svm_clfs):
    plt.subplot(221 + i)
    plot_predictions(svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)

save_fig("moons_rbf_svc_plot")
plt.show()
```

![스크린샷 2022-07-09 18.36.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6224907d-2dc0-4fa1-b68e-40b8f891108c/스크린샷_2022-07-09_18.36.53.png)

- hyperparameter $\gamma$와 $C$를 바꿔서 훈련시킨 model
- increasing $\gamma$ → bell-shape curve narrower, as a result each instance’s range of influence is smaller(샘플의 영향범위가 작아짐): decision boundary more irregular(불규칙), wiggling individual sample(샘플을 따라 구불구불하게 휘어짐)
- small $\gamma$ → bell-shaped curve wider, so instances have a larger ranger of influence, and the decision boundary ends up smoother

→ $\gamma$  acts like a regularization hyperparameter(과대적합 → 감소, 과소적합 → 증가)

similar to the C hyperparameter

- other kernel are used much more rarely
    - string kernel은 text documents나 DNA sequences를 분류할 때 사용됨
- how can you decide which one to use?
    - LinearSVC is much faster than SVC(kernel = “linear”), especially if the training sets is very large
    - if the training set is not too large, you shoild try the Gaussian RBF kernel as well

### 5.2.4 **Computational Complexity**

- LinearSVC python class is based on liblinear library
- kernel trick은 지원 X, but it scales almost linearly woth the number of training instances and the number of features
- time complexity: $O(m\times n)$

- 높은 precision을 요구하면 수행 시간이 길어짐
    
    → tolerance hyperparameter $\epsilon$(epsilon)으로 controll
    
- most classification tasks → tolerance를 default로 두면 잘 작동

- SVC의 training time complexity는 보통 $O(m^2\times n)$  과 $O(m^3\times n)$ 사이
    
    → this means that it gets dreadfully slow when the number of training instances gets large
    
    → This algoritm is perfect for complex but small or medium training sets
    
- the number of features들은 잘 확장됨(especially sparse features,(각 샘플에 0이 아닌 feature가 몇 개 없는 경우))
    
    → In this case, the algorithm scales roughly woth the average number of nonzero features per instance(샘플이 가진 0이 아닌 특성의 평균 수에 거의 비례)
    

![스크린샷 2022-07-09 19.10.53.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/92c9cbc2-b29c-45fe-87be-39621405d671/스크린샷_2022-07-09_19.10.53.png)

## 5.3 SVM Regression

- SVM algorithm은 다목적(versatile)으로 사용할 수 있음
    - 선형, 비선형 분류뿐만 아니라 선형, 비선형 회귀에서도 사용 가능
    - 분류가 아닌 회귀에 적용하는 방법은 the obective(목표)를 반대로 하는 것
        - trying to fit the largest posible street(도로 폭을 최대로, margin을 최대로) between two classes while limiting margin violations(일정한 margin 오류 안에서), SVM Regression은 limiting margin violations 안에서 street 안에 가능한 많은 instances가 들어가도록 fit함
        → 즉 아래 그래프에선 분홍색 data들이 최대한 적어지게!
            
            → margin 밖에 있는 error가 최소가 되도록 동작(margin안에 최대한 많은 관측치가 포함되도록 하는 것)
            
        - width of the street은 hyperparameter $\epsilon$ 로 조절
            
            ![스크린샷 2022-07-11 20.12.35.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/927d1c94-738d-4acf-8fcc-218a44a656df/스크린샷_2022-07-11_20.12.35.png)
            
            - 왼쪽은 margin을 크게, 오른쪽은 margin을 작게 설정한 모습
            - margin 안에 training instances들이 추가되어도 model’s predictions에는 영향을 주지 않아 $`\epsilon$-insensitive` 하다고 표현
            
            ```python
            from sklearn.svm import LinearSVR 
            
            svm_reg = LinearSVR(epsilon=1.5)
            svm_reg.fit(X, y)
            ```
            
            - Scikit-Learn의 LinearSVR class를 사용할 수 있음(Figure 5-10)
            - 초기 작업으로는 data의 scale을 맞추고, cetered 해야됨)

![스크린샷 2022-07-11 20.40.18.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/946fa42f-f073-4bfa-98c5-3cab305647ff/스크린샷_2022-07-11_20.40.18.png)

```python
from sklearn.svm import SVR

svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
svm_poly_reg.fit(X, y)
```

- nonlinear regression task를 처리하기 위해서는 kernelized SVM model을 사용
- random quardractic training set에 2차 polynomial kernel을 사용한 SVM Regression
- 왼쪽은 little regularization(large C value), 오른쪽은 much more regularization(small C value) 
→ 오른쪽은 규제가 커져서 과대적합을 막음(보면 pink data가 오른쪽이 더 많음)
- supports kernel trick

## 5.4 Under the Hood

- 즉 SVM의 이론에 대해서 알아볼 예정
- how SVMs make predictions and how their training algorithms work, starting with linear SVM classifier.

- `notations`
    - bias term: $\theta_0$
    - input feature weights: $\theta_1\ to\ \theta_n$
    
    ⇒ one vector $\theta$에 다 넣음
    
    - all instances의 bias에 해당되는 input $x_0 = 1$을 추가
    
    - 즉, bias term을 $b$로, feature weights를 $w$로 부를 예정
        
        (No bias feature will be added to the input feature vectors.)
        

### 5.4.1 **Decision Function and Predictions**

- linear SVM classifier model은 decision function $w^Tx+b = w_1x_1 + \dots w_nx_n + b$ 를 계산해서 new instance $x$의 class를 예측
    - result가 positive → the predicted class $\hat{y}$ is positive class
    - negative → the predicted class $\hat{y}$ is negative class
    
    ![스크린샷 2022-07-11 21.11.17.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/37868a2a-1a2e-444f-9a05-318cd61fa0b4/스크린샷_2022-07-11_21.11.17.png)
    

![스크린샷 2022-07-11 23.10.12.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/007064ce-4139-4c29-abd0-02cf9fd3bcac/스크린샷_2022-07-11_23.10.12.png)

- 특성이 두 개(petal width and petal length)이기 때문에 two-dimensional
- decision boundary는 decision function이 0인 점들의 집합(set of points)
    - the intersection of two planes, which is a straight line(solid line)
- decision function이 -1 또는 1인 points의 집합 ⇒ The dashed lines
    - parallel and at equal distance to the decision boundarym forming a margin around it
- `Training a linear SVM classifier`
    
    == margin이 margin violation(hard margin)을 하나도 발생하지 않거나, 제한적인 soft margin을 가지면서 가능한 margin을 wide하게 하는 $w, b$를 찾는 것
    

### 5.4.2 Training Objective

![스크린샷 2022-07-11 23.42.55.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e719afd3-6aad-4f57-b992-5d7544e772d8/스크린샷_2022-07-11_23.42.55.png)

- the slope if the decision function == the norm of the weight vector, $||w||$
- 만약 이를 2로 나눈다면 decision function이 $\pm1$이 되는 점들이 decision boundary로부터 
2배만큼 멀어짐
    
    ⇒ slope에 $n$을 나눈다면 margin에 $n$배를 하는 것과 같음
    
- The smaller the weight vector $w$, the larger the margin

margin을 크게 하기 위해서 $||w||$를 최소화 하려고 함.

- mragin violation을 만들지 않으려면(hard margin) decision function이 all positive training instances에 대해 1보다 커야되고, negative training instances에 대해서는 -1보다 작아야됨
- 음성샘플$(y^{(i)} = 0)$일 때, $t^{(i)} = -1$로, 양성샘플$(y^{(i)} = 1)$일 때, $t^{(i)} = 1$로 정의하면,
    
    $t^{(i)}(w^Tx^{(i)}+b)\ge1$로 표현 가능
    

![스크린샷 2022-07-12 00.15.44.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b4604125-16fd-45b5-985a-20a7ca3416af/스크린샷_2022-07-12_00.15.44.png)

- the hard margin linear SVM classifier의 objective function을 
제약이 있는 `constrained optimization` 문제로 표현 가능

> $||w||$를 최소화하는 대신 $\frac{1}{2}w^Tw$를 최소화함

$\frac{1}{2}||w||^2$가 더 깔끔하고 간단하게 미분됨, 또한 $||w||$는 $w = 0$에서 미분 불가능
(Optimization algorithms은 differentiable functions(미분가능함수)에서 잘 작동)
> 

- soft margin objective(soft margin classficator의 목적함수)를 구성하기 위해서는 each instance에 대한 `slack variables` $(\zeta^{(i)}) \ge0$를 도입해야 됨
    - $\zeta^{(i)}$는 $i^{th}$ instance가 얼마나 margin을 violation할지 정함.
- making the slack variables as small as possible to reduce the margin violations, and making $\frac{1}{2}w^Tw$ as small as possible to increase the margin (conflicting objectives)
    
    ⇒ hyperparameter $C$ allows us to define the trade-off between these two obejctives
    
    ![스크린샷 2022-07-12 00.25.10.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d8921537-aca7-4114-b3f2-194a029eaec3/스크린샷_2022-07-12_00.25.10.png)
    
    - this gives us the constrained optimization problem in *Equation 5-4*

### 5.4.3 **Quadratic Programming**

- `Quadratic Programming(QP)`
    - hard margin, soft margin problem → convex quadratic optimization problems with linear constraints(선형적인 제약 조건이 있는 볼록 함수의 이차 최적화 문제)
- (여러 technique으로 QP를 푸는 algorithms이 있지만 책의 범위를 벗어난대여,,,)
- The general problem formulation
    
    ![스크린샷 2022-07-12 00.30.18.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b185460c-f06a-4604-82e7-a900f9a6e9a7/스크린샷_2022-07-12_00.30.18.png)
    

![스크린샷 2022-07-12 00.50.22.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/51738aa5-2a71-4208-bfc9-11afa89f7cec/스크린샷_2022-07-12_00.50.22.png)

- 일단 optimization problem은 최적화 해야 될 variable이 있고, obeject function $f$가 있는 상태에서 inequality constraint function(부등식 제약 함수), equality constraint function 등이 존재하는 경우, 이 제약조건을 만족하는 경우에서 ojective function $f$를 최소로 만드는 $x'$을 찾는 것
    - inequality constraint function이 convex function
    - equality constraint function이 affine function(y절편이 0이 아닌 함수)인 경우 convex optimization problem이라고 할 수 있음.
    - Convex sets
        - 두 점 $x_1, x_2$를 잇는 선분
        : $x = \theta x_1 + (1-\theta)x_2 \ with\ 0 \le \theta \le1$
        - 두 점 $x_1, x_2$를 잇는 선분이 이 집합에 다시 포함될 때 이 집합을 convex set이라고 부름
        - 집합 $C$가 convex가 될 조건은 다음과 같음
            
            $x_1, x_2 \in C, 0 \le \theta \le 1$  ⇒ $\theta x_1 + (1-\theta )x_2 \in C$
            
        - set은 오목한 구간이 없고 볼록한 구간만 존재해야됨
    - Convex function
        - 볼록함수이므로 local minimum이 항상 global minimum이 됨
        -