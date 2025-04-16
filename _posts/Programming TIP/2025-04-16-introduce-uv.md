---
title: "uv: 새로운 파이썬 패키지 관리자 소개"
categories: [Programming TIP]
tags: [uv, pip, python, 가상환경, mcp]
---

Python 개발 환경에서 패키지 관리와 가상 환경 설정은 필수적인 작업입니다. 기존에는 `pip`, `pipenv`, `poetry` 등의 도구를 조합하여 사용했지만, 최근에는 이러한 기능을 통합하여 제공하는 uv가 주목받고 있습니다.

또한 MCP 서버 설치 및 실행을 할 때도 `uv`, `uvx` 라는 것이 굉장히 자주 보이죠.

`uv`가 무엇인지, 그래서 뭐가 더 좋은 건지, 어떻게 쓰는건지에 대해 알아보겠습니다!

더 나아가 그럼 기존에 다른 방식으로 관리하던 패키지관리 방법을 `uv`로 마이그레이션하는 방법도 알아보겠습니다.

---

## **1. uv 소개**

<!-- prettier-ignore -->
> **uv는 만능 python 패키지관리 도구입니다.**
{: .prompt-info }

Python 프로젝트에서 패키지를 관리하지 위해서는 아래의 과정들이 필수적이었습니다.

1. 패키지 설치 : `pip install openai`
2. 가상환경 생성 : `python -m venv .venv`
3. 파이썬 버전관리 : `py -3.11`

이 모든 것을 `uv` 하나로 모두 관리할 수 있습니다!

## **2. uv의 장점**

> **빠르고, 편하고, 새로운 표준이다!**

### 2-1. 빠르다

streamlit library 설치로 비교 테스트를 해보았습니다.

기존 방식은 pip 방식으로 가상환경을 생성하고, 가상환경 실행 후, pip install을 통해 streamlit을 설치합니다.

uv 방식은 uv init을 통해 uv 프로젝트를 만들고, uv add를 통해 streamlit을 설치합니다.

**기존 방식은 약 60초 소요되었으며, uv 방식은 놀랍게도 1초 소요되었습니다.**

기존 방식에 비해 uv가 훨씬 빠른 것을 확인하였습니다.

uv가 빠른 이유는 다음과 같습니다.

1. Rust 기반 구현: uv는 Rust로 작성되어 있어 메모리 안전성과 성능이 뛰어납니다.
2. 병렬 처리: 여러 패키지를 동시에 다운로드하고 설치할 수 있습니다.
3. 캐싱 시스템: 이전에 설치한 패키지를 캐시하여 재사용합니다.
4. 의존성 해결 최적화: 의존성 트리를 효율적으로 분석하고 해결합니다.

### 2-2. 편하다

| 구분               | pip 이용할 때                                                                                                                                       | uv 이용할 때                                               |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **[만들 때]**      | - `python -m venv .venv` <br> - `.venv\scripts\activate` <br> - `pip install openai` <br> - `python main.py` <br> - `pip freeze > requirements.txt` | - `uv init` <br> - `uv add openai` <br> - `uv run main.py` |
| **[공유 받을 때]** | - `python -m venv venv` <br> - `venv\scripts\activate` <br> - `pip install -r requirements.txt` <br> - `python main.py`                             | - `uv run main.py`                                         |

pip와 비교하여 uv를 이용하는 경우 명령어가 훨씬 간단해집니다.

1. 자동 가상환경 관리: uv는 모든 프로젝트는 각자의 가상환경을 가진다는 기본 철학이 있습니다. 이를 통해 프로젝트의 가상환경을 자동으로 생성하고, 자동으로 인식합니다.
2. 명령어 단순화: 복잡한 가상환경 활성화 과정 없이 바로 패키지 설치와 실행이 가능합니다.
3. 통합된 인터페이스: 패키지 설치, 가상환경 관리, Python 버전 관리 등이 하나의 도구로 통합되어 있습니다.
4. 자동 의존성 관리: `pyproject.toml`과 `.uvlock` 파일을 통해 의존성을 자동으로 관리합니다.

### 2-3. 표준이다

대부분의 MCP Tool이 uv를 지원하고 있으며, MCP Server들도 내부적으로 uv를 사용하고 있습니다.
Python을 이용하고, MCP를 개발할 때 뿐먼 아니라 다른 사람들이 만든 Python 프로젝트를 사용할 때에도 앞으로는 **기본 uv 명령어를 알고 있는 것**이 굉장히 유용하겠죠?

## **3. uv 설치하기**

[uv 공식사이트](https://docs.astral.sh/uv/#installation)를 참고해 설치할 수 있습니다.

**macOS/Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

저는 pip를 이용해 설치하는 방법으로 진행했습니다 !

```bash
pip install uv
```

## **4. uv 프로젝트 실습해보기**

### 4-1. uv 프로젝트 생성하기

```bash
uv init [project명]
```

- `uv init`: 현재 디렉토리 내에서 uv project 구조에 필요한 설정 파일 생성됨
- `uv init PROJECT_NAME`: 작성한 프로젝트 명으로 디렉토리 생성 후 해당 디렉토리 내에 uv project가 생성됨

uv 프로젝트를 생성하면 다음과 같은 **기본 설정 파일**들이 생성됩니다:

- `.gitignore`: Git 버전 관리에서 제외할 파일 및 디렉토리를 지정합니다. 일반적으로 가상환경, 빌드 파일, 개인 설정 파일 등이 포함됩니다.
- `.python-version`: 프로젝트에서 사용할 Python 버전을 명시합니다. 필요에 따라 변경할 수 있습니다.
- `main.py`: 프로젝트의 메인 실행 파일입니다. Python 스크립트의 진입점으로 사용되며, 프로젝트의 주요 로직이 구현됩니다.
- `pyproject.toml`: Python 프로젝트의 명세서 역할로, 메타데이터와 빌드 설정을 포함하는 파일입니다. 의존성 관리, 빌드 시스템 설정 등을 정의할 수 있습니다.
- `README.md`: 프로젝트에 대한 설명을 담고 있는 파일입니다. 프로젝트의 목적, 설치 방법, 사용법 등을 문서화하여 다른 사용자들이 쉽게 이해할 수 있도록 돕습니다.

### 4-2. 패키지 설치

```bash
uv add [패키지명]
```

패키지를 설치하면 **프로젝트에 몇 가지 변화**가 생깁니다.

1. `.venv` 가상환경 생성되며 활성화됨
2. `.uvlock` 파일 생성
   - 설치한 패키지의 구체적인 버전과 하위패키지의 버전 등 의존성 정보가 기재됩니다.
3. `pyproject.toml` 파일의 `dependencies` 변경
   - 추가한 패키지와 버전이 명시됩니다.

그 외 패키지 관리를 위한 명령어는 아래를 참고하세요.

- `uv pip list`: 현재 가상환경에 설치된 패키지와 버전을 확인할 수 있습니다.
- `uv remove [패키지명]`: 패키지를 삭제하고 싶은 경우 사용하면 자동으로 가상환경에서 삭제되고, `.uvlock` 파일과 `pyproject.toml` 파일의 `dependencies` 에서 제거됩니다.
- `uv update [패키지명]`: 패키지를 업데이트하고 싶은 경우 사용합니다.

### 4-3. 프로그램 실행

```bash
uv run python main.py
```

- UV가 자동으로 프로젝트 가상 환경을 사용하여 python을 실행합니다.
- 실제로 가상 환경을 활성화/비활성화할 필요 없이 편리하게 실행 가능합니다.

이 외에도 [FastAPI](https://docs.astral.sh/uv/guides/integration/fastapi/) 환경에서 uv를 활용하는 방법도 있으니 참고해보세요 🥳

## 5. **기존 `requirements.txt`를 uv로 전환하기**

기존 `pip`를 이용해서 `requirements.txt`를 통해 가상환경을 구성했던 프로젝트를 `uv`를 이용해 전환해보았습니다!

그 방법을 단계 별로 알려드릴게요. 😊

> 샘플로 사용한 [FastAPI github project](https://github.com/marciovrl/fastapi.git)

### 5-1. uv 초기화

**[기존 프로젝트 구조]**
![Image]({{"/assets/img/posts/2025-04-16-21-12-37.png" | relative_url }})
프로젝트 루트에 `requirements.txt` 파일이 존재합니다.
원래의 방법대로 라면 가상환경을 생성하고, 활성화한 후, `requirements.txt`을 통해 pip install받는 과정을 해야겠었죠?

하지만 이제 그럴 필요가 없습니다! 😀

```bash
uv init
```

uv 초기화 명령어를 통해 uv 기본 설정파일을 생성합니다.
![Image]({{"/assets/img/posts/2025-04-16-21-19-36.png" | relative_url }})

1. `.python-version`

   - python 버전을 작성해주세요.

2. `pyproject.toml`

   - project의 설정 정보를 확인 후 수정이 필요하다면 수정해주세요.

기존에 `README.md` 등 이미 존재하던 파일은 생성되지 않았습니다.
하지만 `main.py`는 프로젝트 루트디렉토리에 없어 생성되었습니다. 하지만 저희는 app 디렉토리 내에 존재하기 때문에 지워줍니다. (⚠️ main.py 파일이 루트디렉토리에도 존재하면 나중에 실행 시 오류가 발생하니 꼭 지워주세요!)

### 5-2. 가상환경 및 패키지 설치

```bash
uv add -r requirements.txt
```

명령어 수행 시 아래의 과정을 실행합니다.

1. 가상환경 생성 및 활성화
2. 가상환경에 패키지 설치 (하위 패키지까지 설치)
3. `pyproject.toml`과 `uv.lock`파일에 의존성이 기록

{: file="pyproject.toml" }

```
[project]
name = "fastapi-uvtest"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [ # 패키지 추가됨
    "fastapi>=0.115.12",
    "pandas>=2.2.3",
    "pytest>=8.3.5",
    "uvicorn>=0.34.1",
]
```

{: file="uv.lock" }

```
version = 1
revision = 1
requires-python = ">=3.11"
resolution-markers = [
    "python_full_version >= '3.12'",
    "python_full_version < '3.12'",
]

[[package]]
name = "annotated-types"
version = "0.7.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.pythonhosted.org/packages/ee/67/531ea369ba64dcff5ec9c3402f9f51bf748cec26dde048a2f973a4eea7f5/annotated_types-0.7.0.tar.gz", hash = "sha256:aff07c09a53a08bc8cfccb9c85b05f1aa9a2a6f23728d790723543408344ce89", size = 16081 }
wheels = [
    { url = "https://files.pythonhosted.org/packages/78/b6/6307fbef88d9b5ee7421e68d78a9f162e0da4900bc5f5793f6d3d0e34fb8/annotated_types-0.7.0-py3-none-any.whl", hash = "sha256:1f02e8b43a8fbbc3f3e0d4f0f4bfc8131bcb4eebe8849b8e5c773f3a1c582a53", size = 13643 },
]
```

### 5-3. FastAPI 실행하기

```bash
uv run fastapi dev
```

fastapi 구조는 위 명령어를 통해 바로 실행할 수 있습니다!

> 참고: https://docs.astral.sh/uv/guides/integration/fastapi/

⚠️ 단 한가지 수정이 필요합니다. `pyproject.toml` 파일에서 FastAPI Standard 버전으로 변경해주세요! ⚠️

{: file="pyproject.toml" }

```
...
dependencies = [
    "fastapi[standard]", # fastapi standard로 변경해주세요!
    "pandas>=2.2.3",
    "pytest>=8.3.5",
    "uvicorn>=0.34.1",
]
```

![Image]({{"/assets/img/posts/2025-04-16-22-27-26.png" | relative_url }})

짜잔- 성공!

### 5-4. uv로 관리하기

이제 `requirements.txt` 파일은 지워주어도 됩니다.
패키지 정보는 `pyproject.toml`와 `uv.lock`에서 해주기 때문이죠!

다른 사람이 프로젝트를 clone 받았을 때는 uv 실행 명령어만 수행하면 됩니다.

> `uv run fastapi dev`

**이제 별도의 가상환경을 생성/활성화/패키지 설치 과정 없이 실행 명령어 하나만으로 모든 것이 가능해집니다!**

---

이제 여러분도 `uv`를 통해 더 빠르고 편리한 Python 개발 환경을 경험해보세요! 🚀

혹시 `uv`에 대해 더 궁금한 점이 있으시다면 댓글로 남겨주세요. 함께 알아가요! 😊
