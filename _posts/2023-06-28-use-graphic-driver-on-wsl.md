---
title: Use Graphic Driver on WSL
date: 2023-06-28 03:20 +0900
category: [Environment Settings]
tag: [WSL Ubuntu]
---

### 환경 버전확인

WSL 버전 확인
: `wsl -l -v`

버전이 2인지 확인한다.

Windows 버전 확인
: [여기](https://support.microsoft.com/en-us/windows/which-version-of-windows-operating-system-am-i-running-628bec99-476a-2c13-5296-9dd081cdd808)에서 확인할 수 있다.

21H2버전이거나 그 이후 버전인지 확인한다. 아닐 경우 해당 버전으로 재설치

### 그래픽 드라이버 설치

[여기](https://www.nvidia.com/Download/index.aspx)에서 그래픽 드라이버를 설치한다.

이 때 우리는 WSL에서 그래픽 작업을 할 것이라 하더라도 WSL에서 그래픽 드라이버를 설치하면 안된다. Windows에서 설치해야 한다. [참고](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#step-1-install-nvidia-driver-for-gpu-support)

> GRD (Game Ready Driver) vs SD (Studio Driver)
: GRD는 게이머, SD는 콘텐츠 제작자가 사용할 목적으로 나왔다. 인공지능 학습의 경우 SD에 속한다. 하지만 반대로 해도 크게 상관은 없다고 한다. 
{: .prompt-tip}

### CUDA 툴킷 설치

[여기](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)에 나와있듯이 WSL에서 Linux의 CUDA 툴킷을 설치한다.

### Ref.

<https://docs.nvidia.com/cuda/wsl-user-guide/index.html#step-1-install-nvidia-driver-for-gpu-support>