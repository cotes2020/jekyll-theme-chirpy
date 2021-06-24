---
title: BentoML - Tích hợp và triển khai PyTorch model ở local
date: 2020-07-08
tags: ['Machine Learning', 'Deep Learning', 'DevOps']
---



### Giới thiệu

BentoML là một framework mã nguồn mở được xây dựng nhằm mục đích triển khai và chạy các model trên production một cách tiện lợi nhất. BentoML còn hỗ trợ những thư viện và framework phổ biến hiện nay như Scikit-Learn, PyTorch, Tensorflow, Keras và vv. Trong bài viết này, tôi sẽ hướng dẫn các bạn triển khai một model có sẵn ở local.

![Overview](/assets/img/2020-07-08/bentoml-overview.png)

Dựa theo ảnh trên, sau khi xây dựng xong một model, BentoML sẽ chạy nó theo nhiều kiểu khác nhau, chẳng hạn như Docker Container, Restful API,... hoặc là triển khai trực tiếp lên các Cloud như Google Cloud Platform (GCP), Amazon Web Services (AWS), Azure như một hệ thống Microservices.

Theo tôi, điều này rất lý tưởng cho chúng ta, những Data Scientist và Machine Learning Practitioner có thể không có chuyên môn cao trong việc phát triển Server và DevOps, để dựng service cho model như một entry point hiệu quả.

**Tóm lại, BentoML có thể:**
- Tạo API chỉ bằng một số dòng code
- Hỗ trợ các Machine Learning Framework phổ biến
- Triển khai model với hiệu suất cao sử dụng adaptive micro-batching
- Chạy model bằng nhiều cách khác nhau
- Phối hợp triển khai linh hoạt với các best-practices của DevOps, hỗ trợ cả Docker và kiến trúc microservices



### Cách tích hợp

Hiện tại, tôi đã train sẵn một PyTorch model sử dụng tập dữ liệu CIFAR-10. Bây giờ công việc của tôi chỉ là tích hợp model này vào API, truy cập bằng Command Line Interface (CLI).

Ban đầu, chúng ta sẽ tạo `main.py`, chúng ta sẽ thực hiện thao tác deploy chủ yếu trên file này. Sau đó, hãy tham khảo kiến trúc model cho tập dữ liệu này.

```python
from torch import nn

import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.norm4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.3)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.norm6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.4)

        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.norm7 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.norm1(x)
        x = F.relu(self.conv2(x))
        x = self.norm2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv3(x))
        x = self.norm3(x)
        x = F.relu(self.conv4(x))
        x = self.norm4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv5(x))
        x = self.norm5(x)
        x = F.relu(self.conv6(x))
        x = self.norm6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = x.view(-1, 256 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = self.norm7(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        return x
```

Sau đó, nhiệm vụ tiếp theo là load lại model và build nó. Trong trường hợp các bạn đang train và có ý định dùng BentoML, các bạn cũng có thể dựa theo đoạn code bên dưới dòng comment `# Init BentoML` để lưu lại version mới nhất.

```python
import pytorch_image_classifier as PIC

if __name__ == '__main__':
    PATH = 'torch_scratch_cifar10_net.pth'

    # Tôi đang chạy bằng Macbook pro nên chỉ có CPU thôi :D
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    # Khởi tạo BentoML
    bento_svc = PIC.PytorchImageClassifier()
    bento_svc.pack('net', net)

    # Tạo Bento
    saved_path = bento_svc.save()
    print('Finished Training:', saved_path)
```

Có thể bạn sẽ thắc mắc rằng file `pytorch_image_classifier.py` được tạo lúc nào. Thực tế thì chưa, chúng ta sẽ tạo file này trong cùng folder với file `main.py`.

```python
from PIL import Image
import os
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import transforms

import bentoml
from bentoml.artifact import PytorchModelArtifact
from bentoml.adapters import ImageInput

# Tôi không chắc về dòng lệnh bên dưới, bạn có thể test xem nó có chạy hay không. Tuy nhiên dòng này là bắt buộc đối với Macbook, thiết bị mà tôi đang sử dụng
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Đây là những lớp trong CIFAR-10, các lớp này sẽ được trả về sau khi gọi API hoặc chạy command line
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Định nghĩa hàm tăng cường dữ liệu cho ảnh trong giai đoạn test
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Định nghĩa thư viện cần thiết cho model này
@bentoml.env(pip_dependencies=['torch', 'Pillow'])
# Định nghĩa thuộc tính để truy cập tới model, thường ta sẽ dùng "net" cho PyTorch
@bentoml.artifacts([PytorchModelArtifact('net')])
class PytorchImageClassifier(bentoml.BentoService):
    # Định nghĩa loại dữ liệu bạn sẽ truyền lên, trong trường hợp này đầu vào sẽ là một ảnh
    @bentoml.api(ImageInput)
    def predict(self, img):
        # Truy cập thuộc tính "net" và thiết lập trạng thái đánh giá (evaluation mode)
        self.artifacts.net.eval()
        input_data = []

        img = Image.fromarray(img).resize((32, 32))
        input_data.append(transform(img))

        outputs = self.artifacts.net(Variable(torch.stack(input_data)).to(device))
        _, output_classes = outputs.max(dim=1)

        return [classes[output_class] for output_class in output_classes]

```

Tiếp theo, chúng ta sẽ chạy lệnh `python main.py`. Kết quả cho ra sẽ là một Bento (nghe như ẩm thực Nhật Bản vậy). Tôi nghĩ tên này có nghĩa khác là package (hoặc là container trong Docker).

![Bento](/assets/img/2020-07-08/bento.jpg)

Log sẽ được in ra như bên dưới:

```
[2020-07-08 17:02:56,350] INFO - BentoService bundle 'PytorchImageClassifier:20200708170229_EE710E' saved to: /Users/tailtq/bentoml/repository/PytorchImageClassifier/20200708170229_EE710E
Finished Training: /Users/tailtq/bentoml/repository/PytorchImageClassifier/20200708170229_EE710E
```

Thay vì dùng `bento_svc.save()` để lưu vào thư mục mặc định của BentoML, chúng ta còn có thể lưu vào bất cứ đâu khi sử dụng hàm `save_to_dir(path)`.

Vậy là oke rồi! Chúng ta đã tích hợp BentoML vào model, bây giờ là lúc để tận hưởng nổ lực của chúng ta.



### Những cách để "thưởng thức" Bento của chúng ta:


#### 1. Sử dụng Restful API:

Chúng ta có thể build 1 bộ API ngay lập tức bằng lệnh bên dưới:

`bentoml serve PytorchImageClassifier:latest`

với `PytorchImageClassifier` là `class` và `latest` là phiên bản. Bạn cũng có thể thay đổi `latest` với một version cụ thể nào đó, chẳng hạn như `20200708170229_EE710E`, để so sánh hiệu suất giữ phiên bản cũ và mới hơn.

Mặc định, BentoML sử dụng thư mục được định trước để tải model. Nếu bạn sử dụng hàm `save_to_dir`, bạn phải nhập thêm đường dẫn đến model đó, ví dụ như `/my/path/PytorchImageClassifier:version`.

Dưới đây là lệnh của tôi:

![Restful API command](/assets/img/2020-07-08/restful-api-command.png)

Đặc biệt là API cũng có Swagger:

![API](/assets/img/2020-07-08/swagger-api-interface.png)



#### 2. Sử dụng CLI:

Đây là lúc chúng ta sử dụng CLI:

`bentoml run PytorchImageClassifier:latest predict --input="dog-puppy-on-garden-royalty.jpg"`

Lưu ý: Tôi cố ý cắt ngắn bớt đường dẫn của input để nhìn gọn hơn.

![CLI result](/assets/img/2020-07-08/cli-result.png)

Model này được huấn luyện từ đầu và đạt được độ chính xác khoảng 85% trong tập `validation set`. Do đó kết quả chắc hẳn sẽ là `dog`.



#### 3. Sử dụng model trong code:

Theo hướng tiếp cận này, chúng ta sẽ tải model trực tiếp bằng Python code.

```python
import cv2

bento_svc = bentoml.load('/my/path/PytorchImageClassifier/20200708170229_EE710E')
image = cv2.imread('dog-puppy-on-garden-royalty.jpg')

result = bento_svc.predict(image)
```

Bên cạnh đó, chúng ta cũng có thể container hoá cho từng phiên bản và đẩy lên Dockerhub. Làm cách này, các thiết bị và server khác có thể được hưởng lợi.



### Tổng kết

Chúng ta vừa trải nghiệm một công cụ tiện lợi cho Data Scientist và Machine Learning Practitioner, những người có thể không quen với phát triển ứng dụng server hoặc DevOps. Đồng thời BentoML cũng cho chúng ta thấy cách mà model chạy trên môi trường production.

Mặc dù vậy, nó vẫn còn những hạn chế. Trong những ứng dụng yêu cầu performance cao và tối ưu hơn, thì chúng ta cần phải dựng model trong môi trường Native, như Android hoặc iOS.

Trong tương lai, tôi sẽ đào sâu hơn vào framework để hiểu được BentoML thực sự hoạt động như thế nào và thử nghiệm `Micro Batching`, phần mà tôi chưa nhắc đến trong bài viết này.

Cảm ơn các bạn đã đọc bài viết này và hãy tiếp tục nghiên cứu nhé :D.

### Tài liệu

[[1] BentoML homepage](https://docs.bentoml.org/en/latest/)

