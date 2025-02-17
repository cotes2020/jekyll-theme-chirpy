---
title: "[Deep Learning Basic] 03_Pytorch를 이용한 데이터 핸들링 및 분석"
categories: [AI, Deep Learning]
tags: [Deep Learning, Dataset, DataLoader, TensorDataset]
---

## 1. Pytorch

> **손쉽게 인공 신경망 모델을 만들고 이용할 수 있도록 지원하는 딥러닝 프레임워크**

##### Tensor Data Types

- **Pytorch 텐서는 Python의 number 및 ndarray를 일반화한 데이터 구조**

![](https://miro.medium.com/max/875/1*-C10tKbZ2h0Zd7maau86oQ.png)

- **dtype: 파이토치로 다루고 있는 데이터 타입**
- **CPU tensor: CPU에 올라갈 수 있는 데이터**
- **GPU tensor: GPU에 올라갈 수 있는 데이터**

##### Numpy array와 Pytorch tensor의 차이

| **Numpy array**                                            | **Pytorch tensor**                                                                                               |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **더 빠른 수학 연산 지원을 위한 numpy 패키지의 핵심 기능** | **CUDA 지원 nvidia\*\***GPU에서도 작동\*\*                                                                       |
| **기계 학습 알고리즘에 사용**                              | **무거운 행렬 계산이 필요한 딥 러닝에 사용**                                                                     |
| **-**                                                      | **- devices_type(계산이 CPU/GPU 중 발생 여부) 및\*\***required_grad(도함수 계산)\*\* **- 동적 계산 그래프 제공** |

## 2. Data Handling 시 사용되는 Class

### `Dataset` Class

> **torch.utils.data.Dataset**
>
> - **파이토치에서 Dataset을 제공하는 추상 클래스**
> - **샘플 및 해당 레이블을 제공**

##### 필수 오버라이드

- **def **`__init__`: dataset의 전처리
- **def **`__len__` : dataset의 크기를 리턴
- **def **`__getitem__` : i번쨰 샘플을 찾을 때 사용

```
# 클래스 생성
class toy_set(Dataset):
  def __init__(self, length=10, transform=None):
    self.x = 10 * torch.ones(length, 2); #  PyTorch에서 특정 크기의 텐서를 생성 ones(행 크기, 열 크기)
    self.y = torch.ones(length, 1);
    self.len = length
    self.transform = transform

  def __getitem__(self, idx):
    sample = self.x[idx], self.y[idx]
    if self.transform: # 데이터 전처리
      sample = self.transform(sample)
    return sample

  def __len__(self):
    return self.lennes(length, 1);
```

`self.x = 10 * torch.ones(length, 2);`

- **크기가 [length, 2]인 텐서를 생성**
- **모든 요소를 1로 초기화**
- **각 요소에 10 곱함**
- **self.x에 할당**

`self.y = torch.ones(length, 1);`

- **크기가 [length, 1]인 텐서를 생성**
- **모든 요소를 1로 초기화**
- **self.y에 할당**

##### Transform 적용 (전처리)

- **Transform 함수 적용**

```
# 사용자 정의 transform module 생성
def scaling(sample):
  x, y = sample
  scaled_x = x / 10.
  scaled_y = y / 10.
  return scaled_x, scaled_y


# Transform 함수 적용
dataset_ = toy_set(transform=scaling)
dataset_[0] # (tensor([1., 1.]), tensor([0.1000]))
```

- **Transform 클래스 적용**

```
# 클래스 생성
class add_ones:
  def __init__(self, added=1):
    self.added = added

  def __call__(self, sample):
    x, y = sample
    x = x + self.added
    y = y + self.added
    sample = x, y
    return sample

# Transform class 적용
a_m = add_ones()

dataset_ = toy_set(10, transform=a_m)
dataset_[-1] # (tensor([11., 11.]), tensor([2.]))
```

- **여러개의 Transform 적용**

```
from torchvision import transforms

# scaling : 10으로 나누기
# a_m: 1 더하기
data_transforms = transforms.Compose([scaling, a_m])

data1 = toy_set(5)
next(iter(data1)) # (tensor([10., 10.]), tensor([1.]))

# 적용
data2 = toy_set(5, transform=data_transforms)
# 초기값에서 10나누고 1 더한값으로 출력
next(iter(data2)) # (tensor([2., 2.]), tensor([1.1000]))
```

---

### `DataLoader` Class

> **torch.utils.data.DataLoader**
>
> - **Dataset은 **`한 번에 한 개씩 샘플`의 feature 와 label 을 retreive한다.
> - **모델을 훈련하는 동안 일반적으로 **`minibatch`로 샘플을 전달하고, 매 epoch 마다 데이터를 reshuffle 하여 overfitting을 줄이며, Python의 multiprocessing을 사용하여 읽는 속도를 향상
> - **쉬운 API로 이러한 복잡성 내용을 추상화한 반복자(iterable)**

- **트레이닝 데이터 준비**

```
from torch.utils.data import DataLoader

# batch_size: 한번에 가져오는 사이즈
# shuffle=True: 가져올때마다 이미지 순서 섞기
# shuffle=False: 가져올때마다 이미지 순서 고정

# 훈련 데이터
training_data = DataLoader(training_data, batch_size=64, shuffle=True)
# 검증 데이터
test_data = DataLoader(test_data, batch_size=64, shuffle=False)
```

- **DataLoader를 통해 반복**
  ```
  # train_features 및 train_labels ( batch_size=64 의 feature 및 label) 의 배치를 반환
  train_features, train_labels = next(iter(training_data))
  train_features.shape # torch.Size([64, 1, 28, 28])
  train_labels.shape # torch.Size([64])
  ```
- **feature와 label 이미지로 확인**
  ```
  img = train_features[0].squeeze()
  label = train_labels[0]
  plt.imshow(img, cmap="gray")
  plt.show()
  print(f"Label: {label}")
  ```

---

### `TensorDataset` Class

> **torch.utils.data.TensorDataset**
>
> - **Dataset을 상속한 클래스**
> - **학습 데이터 x와 레이블 y로 묶어 놓는 컨테이너**
> - **DataLoader에 전달하면 for 루프에서 데이터의 일부분만 간단히 추출**

```
from torch.utils.data import TensorDataset

x = np.random.randn(5, 4)
y = np.random.randint(0, 2, size=5)

# numpy를 torch로 변환
X_train = torch.from_numpy(x)
Y_train = torch.from_numpy(y)

print(X_train)
# tensor([[ 1.7328,  1.3759,  0.3379,  0.9125],
#        [-0.0654,  1.3192, -0.2439,  0.7948],
#        [-1.0853,  0.4325, -0.4286, -0.5648],
#        [-0.7461, -0.7425, -0.0842,  0.1015],
#        [-0.4128,  2.1929, -1.2489,  1.1898]], dtype=torch.float64)
print(Y_train) # tensor([0, 1, 0, 0, 0])

# TensorDataset 적용
train_ds = TensorDataset(X_train, Y_train)

# DataLoader를 통해 할당
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

# feature와 label 획득
train_features, train_labels = next(iter(train_loader))

print(train_features)
# tensor([[ 1.7328,  1.3759,  0.3379,  0.9125],
#        [-0.0654,  1.3192, -0.2439,  0.7948]], dtype=torch.float64)

print(train_labels)
# tensor([0, 1])
```

## 3. 데이터 분석

### 진행 순서

1. **데이터 읽어오기**
2. **데이터 전처리**

   1. **데이터프레임과 시리즈로 구분**

   > **Pandas 라이브러리의 기본 데이터 구조**

   - **데이터프레임: 2차원 테이블 (여러개의 시리즈의 결합)**
     - **=> 독립 변수값(입력 특성)**
   - **시리즈: 1차원 레이블(인덱스) => 하나의 컬럼**
     - **=> 종속 변수값(예측 대상)**

   2. **train과 test 데이터 나누기**
   3. **스케일링 진행**
      - **데이터의 크기가 편차가 클 때 오차를 줄이기 위해 스케일링 진행**
   4. **torch tensor로 변환**

3. **선형회귀 모형 생성**

   1. **모델 class 정의**
   2. **model instance 생성**
   3. **손실함수/optimizer 정의**

4. **Dataset Loader 생성**
5. **Train set으로 훈련 수행**

   - **Batch data Load**
   - **model 을 이용하여 batch data 예측**
   - **loss value 계산**
   - **optimizer 에 저장된 grad value clear**
   - **loss value backpropagate**
   - **optimizer update**

6. **Test set으로 모델 평가**

   - **criterion 으로 손실값 확인**
   - **Loss 시각화**

7. **Test set으로 예측**
8. **평가**

   1. **MSE, R2 계산**
   2. **True vs. Predicted 시각화**

---

### 데이터 구분 종류

- **train data: 모델 학습용 데이터**
- **valid data: 학습된 모델이 예측한 값과 비교하기 위한 실제 데이터**
- **test data**
  - **훈련과정에서 valid data를 직접 훈련에 사용하진 않지만, 생성된 모델이 valid data에 적합할때까지 모델을 변경하기 때문에 간접적으로 모델에 영향을 끼쳤다고 볼 수 있다.**
  - **그렇기 때문에 train 데이터와 valid data을 통해 생성한 모델을 최종적으로 검증하기 위해 사용하는 데이터**

**❗간단하게 사용하는 경우는 test data를 별도로 사용하지 않고, valid data를 test data라는 이름으로 진행하는 경우도 있다.**

---

### 코드 적용 예시

1. **데이터 읽어오기**

   ```
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import MinMaxScaler
   import pandas as pd
   import matplotlib.pyplot as plt
   import torch
   import torch.nn as nn
   import torch.optim as optim
   import warnings
   warnings.filterwarnings('ignore')

   torch.manual_seed(100)

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   # 원본 데이터
   df_boston = pd.read_csv("boston_house.csv", index_col=0)
   ```

2. **데이터 전처리**

   1. **데이터프레임과 시리즈로 구분**

   ```
   # boston : 데이터프레임(테이블)
   boston = df_boston.drop('MEDV', axis=1)
   # target: 시리즈(컬럼)
   target = df_boston.pop('MEDV')
   ```

   2. **train과 test 데이터 나누기**

   ```
   # input/target 지정
   X = boston.values
   y = target.values
   X.shape, y.shape # ((506, 13), (506,))

   # Train 모델 학습
   # Test 모델 검증
   # test_size : train과 test의 비율 (0.2인 경우 train 8, test2)
   # random_state: 섞을 때 동일한 난수로 섞음
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   X_train.shape, X_test.shape, y_train.shape, y_test.shape # ((404, 13), (102, 13), (404,), (102,))
   ```

   3. **스케일링 진행**

   ```
   sc = MinMaxScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.fit_transform(X_test)
   ```

   4. **torch tensor로 변환**

   ```
   X_train_ts = torch.FloatTensor(X_train)
   X_test_ts = torch.FloatTensor(X_test)
   y_train_ts = torch.FloatTensor(y_train).view(-1, 1)
   y_test_ts = torch.FloatTensor(y_test).view(-1, 1)

   X_train_ts.size(), X_test_ts.size(), y_train_ts.size(), y_test_ts.size()
   ```

3. **선형회귀 모형 생성**

   1. **모델 class 정의**

   ```
   class LinearReg(nn.Module):
   def __init__(self, input_size, output_size):
     super().__init__()
     self.fc1 = nn.Linear(input_size, 64)
     self.fc2 = nn.Linear(64, 32)
     self.fc3 = nn.Linear(32, output_size)

   # 순방향 전파 정의
   def forward(self, x):
     x = torch.relu(self.fc1(x))
     x = torch.relu(self.fc2(x))
     output = self.fc3(x)
     return output
   ```

   2. **model instance 생성**

   ```
   model = LinearReg(X_train.shape[1], 1).to(device)
   model
   ```

   3. **손실함수/optimizer 정의**

   ```
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   ```

4. **Dataset Loader 생성**

   ```
   train_ds = torch.utils.data.TensorDataset(X_train_ts, y_train_ts)
   train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
   ```

5. **Train set으로 훈련 수행**

   ```
   # 손실 값 저장
   Loss = []
   # 모델이 훈련할 총 epoch 수
   num_epochs = 100
   for epoch in range(num_epochs):
   for x, y in train_loader:
     # x: 입력 데이터, y: 정답 레이블
     x, y = x.to(device), y.to(device)

     # yhat: 모델이 예측한 값
     yhat = model(x)
     # loss: 손실 값
     loss = criterion(yhat, y)

     # 가중치 업데이트: 그래디언트 0으로 초기화하여 축적 방지
     optimizer.zero_grad()

     # 손실 값에 대한 그래디언트를 역전파 -> 손실함수의 미분값 계산
     loss.backward()

     # 역전파에서 계산된 그래디언트를 사용하여 모델의 매개변수 업데이트
     optimizer.step()

   print("epoch {} loss: {:.4f}".format(epoch+1, loss.item()))
   Loss.append(loss.item())

   print("total : {}".format(Loss))
   ```

6. **Test set으로 모델 평가**

   - **criterion 으로 손실값 확인**

     ```
     criterion(model(X_test_ts.to(device)), y_test_ts.to(device)).item()
     ```

   - **Loss 시각화**

     ```
     plt.plot(Loss)
     ```

     ![image-20240704152905767]({{"/assets/img/posts/a.png"  | relative_url }})

7. **Test set으로 예측**

   ```
   y_pred= model(X_test_ts.to(device)).cpu().detach().numpy()
   y_pred.shape # (102, 1)
   ```

8. **평가**

   1. **MSE, R2 계산**

   ```
   from sklearn.metrics import mean_squared_error, r2_score

   print('MSE : {}'.format(mean_squared_error(y_test, y_pred))
   # MSE : 34.30465347918625

   print('R2: {}'.format(r2_score(y_test, y_pred)))
   # R2: 0.5787140970108864
   ```

   2. **True vs. Predicted 시각화**

   ```
   plt.scatter(y_test, y_pred)
   plt.plot(y_test, y_test, color='red')
   plt.xlabel("True Price")
   plt.ylabel("Predicted Price")
   plt.show()
   ```

   ![image-20240704152905767]({{"/assets/img/posts/b.png"  | relative_url }})
