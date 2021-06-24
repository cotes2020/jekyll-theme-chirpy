---
title: Phân loại chó mèo bằng Keras
date: 2020-04-26
tags: ['Machine Learning', 'Deep Learning', 'Tensorflow', 'Computer Vision']
---

![Cover](/assets/img/2020-04-26/cover.jpg)

---

Đối với những người mới bắt đầu học Deep Learning và Computer Vision, bài toán nhận diện chó mèo là một vấn đề tương đối trực quan mà mọi người nên thử sức, bên cạnh [nhận diện số viết tay](https://en.wikipedia.org/wiki/MNIST_database).

Hôm nay, tôi sẽ xây dựng một model để phân loại chó hoặc mèo ở trong một ảnh. Qua bài viết này, tôi hi vọng nó sẽ mang lại cho bạn những khái niệm cơ bản về Deep Learning. Hãy bắt đầu thôi!



### Tải dữ liệu

Đầu tiên, truy cập [Kaggle](https://www.kaggle.com/) bằng đường [link](https://www.kaggle.com/c/dogs-vs-cats/data) này, bạn có thể tìm thấy dữ liệu gồm ảnh của chó và mèo tại đây. Sau đó, bấm vào nút "Download all" để tải tập dữ liệu về (như ảnh dưới).

![Download dataset](/assets/img/2020-04-26/download-dataset.png)

(Cho những người chưa biết về Kaggle, đây là một nền tảng phổ biến để bạn có thể thực hành và cải thiện Machine Learning skills. Tại Kaggle, bạn sẽ có cơ hội làm việc với những vấn đề thực tế và được học từ những Data Scientist hoặc Machine Learning Practitioner khác. Sau khi nắm một số khái niệm về Machine Learning, tôi khuyến khích các bạn nên tham gia vào nền tảng này để nâng cao khả năng của mình hơn nữa).



### Cấu trúc thư mục

Trong bài viết này, chúng ta sẽ dùng `ImageDataGenerator` trong thư viện Keras, đồng thời class này sẽ cần một cấu trúc thư mục chính xác để thực hiện. Vì vậy, tôi cần phải chuyển tất cả các data được tải về vào thư mục có tên `dataset`, sau đó tôi sẽ phân loại các ảnh được huấn luyện theo tên nhãn.

Về Jupyter notebook, file này sẽ được tạo ở project level.

Cấu trúc hiện tại sẽ như thế này:

![Current directory structure](/assets/img/2020-04-26/directory-structure.png)

Kì vọng:

![Expected directory structure](/assets/img/2020-04-26/directory-structure-2.png)

Ở mục tiếp theo, chúng ta sẽ cùng nhau viết một đoạn python script nho nhỏ để chuyển tất cả hình ảnh vào thư mục tương thích.



### Phân loại ảnh huấn luyện

```python
import os
import shutil
from os.path import isfile, join

# Truy xuất vào thư mục dataset/train
os.chdir('dataset/train')

# Tạo các thư mục nếu không tồn tại
if not os.path.exists('cat'):
    os.mkdir('cat')

if not os.path.exists('dog'):
    os.mkdir('dog')

# List hết các file, bạn có thể dùng glob
files = [f for f in os.listdir() if isfile(join(f))]

for file in files:
    # So sánh tên và sử dụng shutil để chuyển ảnh vào thư mục hợp lệ
    if file[0:3] == 'cat':
        shutil.move(file, 'cat')
    else:
        shutil.move(file, 'dog')
```

Cuối cùng, cấu trúc dự án của chúng ta sẽ như thế này:

![Expected directory structure](/assets/img/2020-04-26/directory-structure-3.png)



### Import thư viện và phân tích

```python
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
```

Với 4 dòng đầu tiên, có thể bạn đã quá quen với những thư viện này rồi. Nhưng nếu bạn thắc mắc phần còn lại sẽ thực hiện những gì, tôi sẽ viết một bài chi tiết vào những buổi tiếp theo.

Chúng ta sẽ hiển thị một ảnh mèo trong tập huấn luyện:

```python
cat0 = cv2.imread('dataset/train/cat/cat.0.jpg')
cat0 = cv2.cvtColor(cat0, cv2.COLOR_BGR2RGB)

plt.imshow(cat0)
```

![Cat sample](/assets/img/2020-04-26/image-2.png)

Và đây là ảnh chó:

```python
dog0 = cv2.imread('dataset/train/dog/dog.0.jpg')
dog0 = cv2.cvtColor(dog0, cv2.COLOR_BGR2RGB)

plt.imshow(dog0)
```

![Dog sample](/assets/img/2020-04-26/image-1.png)



### Tăng cường dữ liệu

Tăng cường dữ liệu (Data augmentation) là một quá trình quan trọng trong những tập dữ liệu nhỏ hoặc không có tính khái quát cao. Vậy kĩ thuật này có ý nghĩa gì khi huấn luyện model? Nó giúp chúng ta tạo ra nhiều biến thể của ảnh để làm tăng tính khái quát của model.

Tôi xin liệt kê một số phương thức để tăng cường dữ liệu cho ảnh:
- Dịch chuyển ngang và dọc (Horizontal and Vertical Shift)
- Lật ngang và dọc (Horizontal and Vertical Flip)
- Xoay (Rotate)
- Tăng giảm độ sáng (Adjust brightness)
- Thu phóng (Zoom)
- ...

 Chúng ta có thể nhìn ảnh của chú sư tử bên dưới:

![Image augmentation](/assets/img/2020-04-26/image-augmentation.png)

Bằng kĩ thuật tăng cường dữ liệu này, chúng ta đã có những bức ảnh sư tử với nhiều góc độ khác nhau, điều này giúp model hiểu có thể khái quát được các thuộc tính sư tử khi đang học.

Bây giờ chúng ta sẽ đi vào công đoạn tăng cường dữ liệu bằng Keras.

```python
image_gen = ImageDataGenerator(width_shift_range=0.1,
                               height_shift_range=0.1,
                               rescale=1/255,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True,
                               validation_split=0.2,
                               fill_mode='nearest')
```

```python
# Thử tăng cường dữ liệu với ảnh chú chó chúng ta vừa tạo lúc nãy
plt.imshow(image_gen.random_transform(dog0))
```

![Augmented image](/assets/img/2020-04-26/augmented-image-1.png)

```python
# Còn đây là ảnh chú mèo sau khi tăng cường
plt.imshow(image_gen.random_transform(cat0))
```

![Augmented image](/assets/img/2020-04-26/augmented-image-2.png)

Sau khi tạo `generator`, chúng ta sẽ dùng object này để load image.

```python
directory = 'dataset/train'
batch_size = 32
image_shape = (127, 127, 3)

train_data = image_gen.flow_from_directory(directory,
                                          target_size=image_shape[:2],
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          subset='training')

test_data = image_gen.flow_from_directory(directory,
                                          target_size=image_shape[:2],
                                          batch_size=batch_size,
                                          class_mode='categorical',
                                          subset='validation')
```



### Xây dựng và huấn luyện model

Đây có vẻ như là nhiệm vụ tuyệt vời nhất cho chúng ta. Chúng ta sẽ tạo một model với **8 layer** với lượng filter tăng dần sau vài convolutional layer.

```python
model = Sequential([
    Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=image_shape),
    Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
    Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
    Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),

    Dense(500, activation='relu'),
    Dropout(0.5),

    Dense(2, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
```

Tóm tắt về model:

![Model summary](/assets/img/2020-04-26/model-summary.png)

```python
import warnings
warnings.filterwarnings('ignore')

results = model.fit_generator(train_data,
                              epochs=25,
                              validation_data=test_data,
                              validation_steps=test_data.samples//batch_size,
                              steps_per_epoch=train_data.samples//batch_size)
```

![Training process](/assets/img/2020-04-26/augmented-image-2.png)

Sau 25 epoch, hẳn là chúng ta đã khá thoải mái khi model đã đạt được độ chính xác tương đối cao mà không bị `overfit` với tập data huấn luyện. Trong thực tế, nếu chúng ta không sử dụng `pretrained models`, thì chúng ta sẽ cần dành nhiều thời gian hơn để tìm ra kiến trúc phù hợp và tối ưu để giải quyết vấn đề này.


```python
# Lưu model để xây dựng app
model.save('model.h5')
```

### Phát họa lịch sử huấn luyện

```python
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
ax[0].plot(results.history['accuracy'], label='Train accuracy')
ax[0].plot(results.history['val_accuracy'], label='Validation accuracy')
ax[0].legend(['Train accuracy', 'Validation accuracy'])

ax[1].plot(results.history['loss'], label='Train loss')
ax[1].plot(results.history['val_loss'], label='Validation loss')
ax[1].legend(['Train loss', 'Validation loss'])

plt.show()
```

![Augmented image](/assets/img/2020-04-26/history.png)



### Tài liệu
- [Tăng cường dữ liệu](https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/)
