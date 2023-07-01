---
title: Ray Tune Trouble Shootings
date: 2023-07-02 01:49 +0900
category: [Framework]
tag: [Python, Ray]
---

Ray Tune을 사용하면서 겪은 여러 버그와 해결방법을 정리하였다. 아래 모든 문제를 해결한 코드는 [여기](https://github.com/seokjin1013/FSR-prediction/blob/apply-ray/ray_tune_new.ipynb)에서 확인할 수 있다.

### Tuner Class api를 사용할 때 Resource 최대치를 지정하는 법

ray에서 학습에 사용되는 cpu나 gpu 수를 resource라 부른다.
resource를 얼마나 할당할지 직접 지정하지 않으면 자동으로 적절하게 할당되지만
때때로 oom이 일어날 수 있기 때문에 더 안정적으로 학습하기 위해서는
더 적은 resource를 직접 할당해줘야 한다.

아래 예시 코드와 같이 `ray.tune.with_resources`를 사용한다.
```python
tuner = ray.tune.Tuner(
    trainable=ray.tune.with_resources(
        Trainable,
        resources={'cpu':2}
    ),
    tune_config=...,
    run_config=...,
)
```

resources는 dict와 ray.air.ScalingConfig 클래스가 들어갈 수 있다.

dict가 들어갈 경우 key에는 cpu나 gpu가 들어가고, value로는 하나의 worker 당 할당되는 자원 수를 넣는다.
위 코드의 경우 하나의 worker 당 cpu가 2개씩, gpu는 0개 쓰인다.

ScalingConfig 클래스가 들어갈 경우 [여기](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.ScalingConfig.html#ray.air.ScalingConfig)를 참고하여 다양한 설정을 할 수 있다.

>Trainable이 Class Api인 경우에는 ScalingConfig를 사용하면 적절히 동작하지 않는다. dict를 사용하거나 다른 방법을 사용해야 한다.
{: .prompt-danger}

### Memory Leak를 막는 법

메모리 누수가 일어나 각 Trial마다 사용하고 있는 메모리 양이 늘어나서 결국 oom이 발생하는 현상이 있을 수 있다.

다양한 원인이 있지만 그 중 하나는 Trainable 안에서 직접 학습 데이터를 불러오는 것이 원인이 될 수 있다.
학습 데이터는 용량이 크므로 효율적으로 다루어야 한다. 그러기 위해서는 `ray.tune.with_parameters` 함수를 사용한다.

```python
tuner = ray.tune.Tuner(
    trainable=ray.tune.with_parameters(Trainable, data=...)
    tune_config=...,
    run_config=...,
)
```
tune을 실행하기 전에 data를 준비해두고 tune.with_parameters로 전달한다.
그 다음 Function Api인 경우 인자를 추가해서 데이터를 전달받는다.
```python
def trainable(config, data):
    ...
```
Class Api인 경우에는 setup에서 인자를 추가하여 받아오면 된다.
```python
class Trainable(ray.tune.Trainable):
    def setup(self, config, data):
        ...
```

### Parameter Space를 탐색할 때 branching 하는 법

Hyper Parameter를 탐색할 때 특정 파라미터가 정해지는 것에 따라 다르게 탐색하고 싶을 때가 있다.
예를 들어 Optimizer의 종류를 먼저 결정하고 그에 따라 들어가는 인자를 탐색하고 싶다고 하자.
그러면 Optimizer가 Adam이 걸리면 lr, beta1, beta2를 탐색하고 싶고 SGD가 걸리면 lr, momentum을 탐색하고 싶을 수 있다.
하나의 탐색 결과에 따라 아예 다른 노선을 타고 싶다면 Ray의 탐색방법만으로는 불가능하다.

Optuna를 사용하여 해결한다. Ray에서 Optuna와 호환되는 `ray.tune.search.optuna.OptunaSearch`를 만들어놓았기 때문에 호환이 잘 된다.
아래와 같이 `ray.tune.TuneConfig`에 search_alg로 해당 클래스 인스턴스를 넣어준다.
```python
tuner = ray.tune.Tuner(
    trainable=...,
    tune_config=ray.tune.TuneConfig(
        num_samples=-1,
        search_alg=ray.tune.search.optuna.OptunaSearch(
            space=define_searchspace,
            metric='rmse',
            mode='min',
        ),
    ),
    run_config=...,
)
```
이 인스턴스는 3개의 인자를 반드시 지정해줘야 한다.

1. space는 Parameter Space를 지정하는 함수이다.
2. metric은 최적화 하고싶은 값을 지정한다. Function Api의 경우 session.report, Class Api인 경우 step함수의 반환값으로 나오는 결과의 key 중 하나로 한다.
3. mode는 'min', 'max'로 지정하여 최소로 최적화할지 최대로 최적화할지를 지정한다.

space인자로 들어가는 함수는 아래와 같이 예시를 들어보았다.
```python
def define_searchspace(trial):
    model_type = trial.suggest_categorical('model', ['fsr_model.LSTM', 'fsr_model.CNN_LSTM'])
    if model_type == 'fsr_model.LSTM':
        trial.suggest_int('model_args/input_size', 6, 6)
        trial.suggest_int('model_args/output_size', 6, 6)
        trial.suggest_categorical('model_args/hidden_size', [8, 16, 32, 64, 128])
        trial.suggest_int('model_args/num_layer', 1, 8)
    elif model_type == 'fsr_model.CNN_LSTM':
        trial.suggest_int('model_args/input_size', 6, 6)
        trial.suggest_int('model_args/output_size', 6, 6)
        trial.suggest_categorical('model_args/cnn_hidden_size', [8, 16, 32, 64, 128])
        trial.suggest_categorical('model_args/lstm_hidden_size', [8, 16, 32, 64, 128])
        trial.suggest_int('model_args/cnn_num_layer', 1, 8)
        trial.suggest_int('model_args/lstm_num_layer', 1, 8)
```
trial은 [여기](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html)에 나와있는 대로
`suggest_categorical`, `suggest_int`, `suggest_float` 함수를 가진다. 모두 uniform sampling을 하고 kwargs로 step이나 log가 있다.

ray의 방식과는 달리 dict가 아닌 함수로 파라미터를 탐색하기 때문에 if안에서 탐색이 가능하다. 위 함수의 경우 아래와 같은 결과가 config로 전달될 수 있다.
```python
{
    'model':'fsr_model.LSTM'
    'model_args':{
        'input_size':6,
        'output_size':6,
        'hidden_size':64,
        'num_layer':4,
    }
}
```
```python
{
    'model':'fsr_model.CNN_LSTM'
    'model_args':{
        'input_size':6,
        'output_size':6,
        'cnn_hidden_size':16,
        'cnn_num_layer':1,
        'lstm_hidden_size':32,
        'lstm_num_layer':8,
    }
}
```
관찰하면 알 수 있다시피, 파라미터 이름을 'model_args/input_size'로 지정하면 config['model_args']['input_size']으로 참조할 수 있도록 dict를 만들어준다.
항상 '/'문자가 기준으로, '/'가 여러개 나와도 된다.
> metric과 mode는 `TuneConfig`에서 전달했더라도 생략하면 안된다. 반면 `TuneConfig`의 kwarg인 scheduler로 전달되는 객체에는 metric과 mode를 생략해도 됐었다.
{: .prompt-tip}

### Parameter Space로 학습에 사용할 클래스를 지정하는 방법

Parameter Space로 탐색할 땐 항상 primitive type만 사용해야 한다. 클래스를 바깥에서 인스턴스화 하여 전달하면 오버헤드가 발생할 수 있다.
또한 클래스 type을 전달하는 것도 ray 기본 Parameter 탐색 방식에서는 사용할 수 있지만 권장하지는 않고, branching을 사용하기 위해 Optuna로 탐색할 경우
primitive type 외에는 엄격히 허용하지 않는다.

다른 패키지나 모듈에 있는 클래스를 import해서 사용할 수 있도록 지시할 수 있는 문장을 문자열로 전달해야 하는 것을 목표로,
ray에서 권장하는 방식인지는 모르겠으나 나는 다음과 같이 해결하였다:

`torch.optim` 모듈에 있는 클래스인 `Adam`을 사용하고 싶을 때 'torch.optim.Adam'을 전달하고 Trainable 내부에서 아래와 같은 함수를 호출한다.
```python
def _import_class(name:str):
    import importlib
    index = name.rfind('.')
    module_name = name[:index] if index != -1 else '__main__'
    class_name = name[index + 1:]
    return getattr(importlib.import_module(module_name), class_name)
```
'torch.optim.Adam'은 마지막 '.'으로 분할되어 module_name은 'torch.optim'이 되고 class_name은 Adam이 된다.
importlib.import_module은 모듈 이름 문자열로 모듈을 import 할 수 있다. 또한 getattr로 모듈에 있는 특정 클래스를 클래스 이름 문자열로 가져올 수 있다.

### Wandb를 연결하는 방법

두 가지 방법이 있는데 쉬운 방법은 아래와 같이 `run_config`에 `WandbLoggerCallback`을 연결해주는 것이다. project 이름만 넣으면 된다.
```python
tuner = ray.tune.Tuner(
    trainable=...,
    tune_config=...,
    run_config=ray.air.RunConfig(
        callbacks=[
            ray.air.integrations.wandb.WandbLoggerCallback(project='FSR-prediction'),
        ],
    ),
)
```
다른 한 가지 방법은 callback을 넣지 말고 Trainable 클래스 안에서 `wandb_setup`함수를 호출하여 직접 wandb를 사용하는 것인데,
group이름이나 trial이름이나 기록할 metric을 직접 다 설정해줘야 해서 불편했기도 하고 wandb에 능숙하지 않아서 힘들었다.