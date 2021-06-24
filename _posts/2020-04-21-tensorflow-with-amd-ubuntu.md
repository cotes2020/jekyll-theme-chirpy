---
title: Sử dụng Tensorflow với AMD GPU trên Ubuntu
date: 2020-04-21
tags: ['Machine Learning', 'Tensorflow', 'ROCm']
---

![Cover](/assets/img/2020-04-21/cover.png)

---

Sau khi tôi được cho card đồ  họa dòng Radeon RX 570 của hãng AMD từ anh họ (ngân sách hạn chế :D), tôi đã có một khoảng thời gian chật vật với nó, bởi vì nó không support cho việc train các model deep learning. Tôi thử test với Tensorflow, một framework hỗ trợ việc phát triển các model deep learning, và kết quả hiển nhiên là cực kì chậm, chậm như lúc train với CPU vậy.

Trải qua những ngày nhàm chán, tôi cuối cùng đã setup thành công Tensorflow với Docker chạy trên GPU của AMD. Vì vậy trong bài vì vậy trong bài viết này, tôi xin được chia sẻ cách để setup nhanh nhất.



### Cài đặt ROCm trên Ubuntu

1. Chạy các dòng lệnh bên dưới để đảm bảo hệ thống đã được cập nhật:
```bash
sudo apt update
sudo apt dist-upgrade
sudo apt install libnuma-dev
sudo reboot
```

2. Cài đặt ROCM:
```bash
wget -q -O - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
```

3. Cấp quyền:
```bash
groups
sudo usermod -a -G video $LOGNAME
echo 'ADD_EXTRA_GROUPS=1' | sudo tee -a /etc/adduser.conf
echo 'EXTRA_GROUPS=video' | sudo tee -a /etc/adduser.conf
```

4. Khởi động lại máy

5. Xác nhận ROCm được cài đặt thành công:
```bash
/opt/rocm/bin/rocminfo
/opt/rocm/opencl/bin/x86_64/clinfo
```

![/opt/rocm/bin/rocminfo](/assets/img/2020-04-21/command-1.png)

![/opt/rocm/opencl/bin/x86_64/clinfo](/assets/img/2020-04-21/command-2.png)

__Lưu ý:__ Chạy những dòng này để ROCm hoạt động hiểu quả hơn:
```bash
echo 'export PATH=$PATH:/opt/rocm/bin:/opt/rocm/profiler/bin:/opt/rocm/opencl/bin/x86_64' | sudo tee -a /etc/profile.d/rocm.sh
```



### Cài đặt Docker

Thật sự thì đã có rất nhiều hướng dẫn cài đặt Docker trên mạng, nhưng tôi khuyến khích các bạn làm theo hướng dẫn ở link [này](https://do.co/2zcd8NI).



### Cài đặt Tensorflow với ROCm

1. Pull Tensorflow image (đảm bảo Docker của bạn đang hoạt động):
```bash
# image này khoảng 5-7 GB
sudo docker pull rocm/tensorflow:latest
```

2. Khởi động container với Tensorflow image:
```bash
docker run -i -t \
    --name=tensorflow \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --workdir=/docker \
    -v $HOME/your-working-directory:/docker rocm/tensorflow:latest /bin/bash
```

__Lưu ý:__ Nếu bạn khởi động lại máy, bạn cần khởi động lại container và exec vào container để chạy:
```bash
sudo docker container start tensorflow # container name
sudo docker exec -it tensorflow /bin/bash
```

(Một cách khác là bạn có thể dùng daemon trong Docker để chạy container mỗi khi khởi động, tôi thì nghĩ cách này không hiệu quả lắm, khá tốn resource của máy).

3. Chạy Jupyter notebook:
```bash
jupyter notebook --allow-root
```
![Chạy notebook](/assets/img/2020-04-21/image-1.png)

4. Tạo một notebook mới:

![Tạo notebook](/assets/img/2020-04-21/image-2.png)



### Sử dụng Tensorflow

Tôi sẽ dùng Python để test ROCm đã thực sự chạy hay chưa. (Nếu bạn chưa biết Python, tôi khuyến khích bạn nên học một số khóa trên Udemy).

Chúng ta sẽ tạo một neural network cơ bản trên tập dữ liệu MNIST:

```python
# Cài đặt một số thư viện cần thiết
!pip install cv2 numpy matplotlib
```

```python
# Load thư viện
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten

import os
import cv2
import numpy as np
```

```python
# Load dữ liệu bằng Keras mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
```

```python
# Phân loại các class
y_cat_train = to_categorical(y_train)
y_cat_test = to_categorical(y_test)
```

```python
# Chuẩn hóa input
X_train = X_train / 255
X_test = X_test / 255
X_train = X_train.reshape(len(X_train), 28, 28, 1)
X_test = X_test.reshape(len(X_test), 28, 28, 1)
```

```python
# Xác định model
model = Sequential([
    Conv2D(filters=32, kernel_size=(5,5), input_shape=(28,28,1), activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=(5,5), input_shape=(28,28,1), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
             optimizer='rmsprop',
             metrics=['accuracy'])
```

```python
model.summary()
```

![Kiến trúc của model](/assets/img/2020-04-21/image-3.png)

```python
model.fit(X_train, y_cat_train, epochs=3)
```

![Kết quả sau khi train](/assets/img/2020-04-21/image-4.png)

```python
model.evaluate(X_test, y_cat_test)
```

![Kết quả đánh giá model](/assets/img/2020-04-21/image-5.png)

```python
# Lưu model
model.save('model.h5')
```



### Tổng kết
Sau khi đi qua các bước này, chúng ta đã chạy được Tensorflow với AMD và GPU. Theo tôi, dành thời gian để setup ROCm còn tốt hơn việc train model mà không có GPU.

Tuy nhiên thì tôi cũng có một số đánh giá sau khi dùng ROCm trong một khoảng thời gian:
- Chưa hỗ trợ Darknet framework (rất nổi tiếng với YOLO, một model nhận diện đối tượng real-time)
- Làm việc với PyTorch không hiệu quả (có một số bug không fix được)

Vì vậy, với các dòng card yếu thì tôi nghĩ dùng Google Colab sẽ hiệu quả hơn rất nhiều (11 GB GPU + 16 GB RAM), còn nếu bạn có card AMD mạnh thì bạn nên setup rồi đánh giá thử xem.

Cảm ơn các bạn đã đọc bài viết này và hãy tiếp tục nghiên cứu nhé :D.

### Tài liệu
- [Source code](https://github.com/tailtq/ml-learning/blob/master/handmade-products/digit-recognition/model.ipynb)
- [Cài đặt ROCm](https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html)
- [Nguồn cover và bài viết hữu ích](https://towardsdatascience.com/train-neural-networks-using-amd-gpus-and-keras-37189c453878)
