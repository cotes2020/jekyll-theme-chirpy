---
title: "[Deep Learning Basic] 04_Dataset, DataLoader, TensorDataset 알아보기"
categories: [AI, Deep Learning]
tags: [Deep Learning, Dataset, DataLoader, TensorDataset]
---
### `Dataset` Class

> **torch.utils.data.Dataset**
>
> - **파이토치에서 Dataset을 제공하는 추상 클래스**
> - **샘플 및 해당 레이블을 제공**

##### 필수 오버라이드

- def `__init__`: dataset의 전처리
- def `__len__` : dataset의 크기를 리턴
- def `__getitem__` : i번쨰 샘플을 찾을 때 사용

**클래스 생성**

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
> - Dataset은 `한 번에 한 개씩 샘플`의 feature 와 label 을 retreive한다.
> - 모델을 훈련하는 동안 일반적으로 `minibatch`로 샘플을 전달하고, 매 epoch 마다 데이터를 reshuffle 하여 overfitting을 줄이며, Python의 multiprocessing을 사용하여 읽는 속도를 향상
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
