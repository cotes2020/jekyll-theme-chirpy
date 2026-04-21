---
title: "[메이플스토리 클론] #1 프로젝트 설정, 간단한 네모 그리기"
# description: ""
categories: [작업물]
tags: [작업물, 프로젝트_템플릿]
image: "/assets/img/background/20230112-151539.jpg"
hidden: true

date: 2025-10-09. 15:04 # Init
last_modified_at: 2025-10-09. 23:44 # E 초고
---

## 머리말

---

- **목표:**
  - 프로젝트 환경 구성 (VS project, Github repository)
  - DirectX 11 API를 이용해 화면에 네모 그리기
  - 네모 움직이기

## AI 로드맵

---

- 창(Window) 생성 및 메시지 루프:
  - Win32 API를 사용하여 게임 윈도우를 생성합니다.
  - 창 닫기, 크기 변경 등의 기본 메시지를 처리하는 루프를 구현합니다.
- DirectX 11 초기화:
  - D3D11CreateDeviceAndSwapChain을 통해 Device, Device Context, Swap Chain을 생성합니다.
  - Render Target View와 Viewport를 설정하여 화면에 무언가를 그릴 준비를 마칩니다.
- 최소 렌더링 파이프라인 구축:
  - 셰이더 작성: 2D 좌표를 그대로 넘겨주는 최소 기능의 **정점 셰이더(Vertex Shader)**와 단색을 출력하는 **픽셀 셰이더(Pixel Shader)**를 HLSL로 작성합니다.
  - 정점 버퍼 생성: 사각형(삼각형 2개)을 그리기 위한 정점(위치, 텍스처 좌표) 데이터를 정의하고 버퍼를 생성합니다.
  - 상수 버퍼 설정: 2D 게임의 핵심인 **직교 투영 행렬(Orthographic Projection Matrix)**을 포함한 변환 행렬(WVP)을 CPU에서 GPU로 넘겨주기 위한 상수 버퍼를 설정합니다.
- 텍스처 렌더링 및 입력:
  - DirectXTex 라이브러리 등을 이용해 이미지 파일(.png, .bmp)을 로드하는 기능을 구현합니다.
  - 픽셀 셰이더가 텍스처를 샘플링하여 출력하도록 수정합니다.
  - 키보드 입력을 받아 사각형의 위치를 변경하는 간단한 입력 시스템을 구현합니다.

## 1. Visual Studio 설치

---

![251009-150821](/assets/img/post/works/maplestory-clone/251009-150821.png)

노트북으로는 VS Code로만 작업해와서, 따로 Visual Studio를 설치해두지 않았다. (컴퓨터에선 사용하는데, 추석 연휴 때 본가 내려와서 노트북으로 작업 중이다.)  
**Visual Studio 2026 Insider** 나왔길래 설치.  

![251009-150834](/assets/img/post/works/maplestory-clone/251009-150834.png)

아이콘 모양도 바뀐 것 같다. 기존보다 좀 더 색이 진해진 느낌.  

## 2. Win32 API HelloWorld

---

{% include embed/youtube.html id = "V9nwIepvPWc" %}
HelloWorld Win32 API 버전 코드 분석  

VS 설치되는 동안 Win32 API 입문 영상 하나 시청.  

HelloWorld 하나하나 씹어먹지 말고 필요한 것만 알아가라는 것이 인상적. 나라면 분명 코드 하나하나 분석하다가 일주일을 썼겠지. 소중한 가이드라인이다.  

## 3. Visual Studio Project 생성

---

![251009-152835](/assets/img/post/works/maplestory-clone/251009-152835.png)

뭔가 UI가 굉장히 심플해졌다. 나는 심플한 것이 좋다.  

![251009-152914](/assets/img/post/works/maplestory-clone/251009-152914.png)
![251009-153000](/assets/img/post/works/maplestory-clone/251009-153000.png)

이렇게 프로젝트 만들면 되는건지 잘 모르겠다. 일단 고.  

![251009-153100](/assets/img/post/works/maplestory-clone/251009-153100.png)

VS Code처럼 우측에 Copilot 창도 하나 생겼다. (원래 있었나? 최근엔 VS Code만 써와서 잘 모르겠다.) VS Code와 다르게 Copilot창을 떼서 창 왼쪽에 배치할 수도 있다.  
새로운 기능이 많아보인다. 다른 건 그런가보다 싶고, 적응형 붙여넣기는 좀 눈길이 간다. 코드 붙여 넣으면 정리하고, 오류 수정하고, 미구현된 것 구현하고, 언어 번역도 해준다는 것 같다. 유용하겠군.  

![251009-153809](/assets/img/post/works/maplestory-clone/251009-153809.png)

그래서 일단 프로젝트를 만들었는데, 어떻게 DirectX를 써야하는거지? 제미나이한테 물어봤다. (이 녀석 한본어를 쓰잖아 !!)  

![251009-154456](/assets/img/post/works/maplestory-clone/251009-154456.png)

아하, Windows SDK에 DirectX가 포함됐다고 한다. 다시 Installer 열어서 Windows SDK 설치.  

![251009-155504](/assets/img/post/works/maplestory-clone/251009-155504.png)

SDK 설치는 됐는데, 설명따라 프로젝트 속성에서 추가해보려고 해도 옵션이 보이지 않음. 알고보니 내가 처음 프로젝트 팔 때 '공유 항목 프로젝트' 템플릿으로 만들어서, C++ 프로젝트로 인식되고 있는 것이 아니였음. Windows 데스크톱 애플레케이션으로 프로젝트를 다시 생성.  

![251009-160056](/assets/img/post/works/maplestory-clone/251009-160056.png)

이후 `<d3d11.h>` 같은 DirectX11 헤더 사용할 수 있게 됨.  

![251009-163927](/assets/img/post/works/maplestory-clone/251009-163927.png)

간단히 DirectX11 기본적인 코드 생성한 후 단계 마무리. (테마와 폰트 변경은 덤)  

## 4. 간단한 네모 그리기

---

DirectX11 기본적인 코드 생성하면서 공부해야 할 것이 있어보이긴 하는데 (생성된 변수라든지, 스왑 체인이라든지), 일단 네모부터 그려보기.  

네모를 그리려면 다음 4단계를 거쳐야 한다고 한다.  

1. 정점(Vertex) 데이터 정의: 사각형을 구성할 4개의 꼭짓점 정보를 정의합니다.
2. 셰이더(Shader) 작성: GPU가 정점을 어떻게 처리하고(Vertex Shader) 각 픽셀을 어떤 색으로 칠할지(Pixel Shader) 결정하는 프로그램을 작성합니다.
3. GPU 리소스 생성: InitDevice 함수에서 정점 버퍼, 인덱스 버퍼, 셰이더, 입력 레이아웃 등 렌더링에 필요한 DirectX 리소스를 생성합니다.
4. 렌더링 명령: Render 함수에서 생성된 리소스를 사용하여 GPU에 그리기 명령을 내립니다.

### 4.1 전력 변수 추가

```cpp
// --- 사각형 렌더링을 위한 전역 변수 ---
ID3D11VertexShader*     g_pVertexShader = nullptr;      // 정점 셰이더
ID3D11PixelShader*      g_pPixelShader = nullptr;       // 픽셀 셰이더
ID3D11InputLayout*      g_pVertexLayout = nullptr;      // 정점 데이터의 구조를 설명
ID3D11Buffer*           g_pVertexBuffer = nullptr;      // 정점 데이터를 저장할 버퍼
ID3D11Buffer*           g_pIndexBuffer = nullptr;       // 인덱스 데이터를 저장할 버퍼
```

> 사각형 렌더링에 필요한 DirectX 인터페이스들을 저장할 전역 변수를 추가한다.  

정점 셰이더나 픽셀 셰이더는 셰이더 책 읽으면서 봤던 내용. 개념적으로만 알고 있던 내용을 코드로 보는 것은 처음이다.  

### 4.2 InitDevice 함수 설정

```cpp
// --- DirectX 11 초기화 함수 ---
HRESULT InitDevice(HWND hWnd)
{
    // 1. 스왑 체인(Swap Chain) 설정
    // ...

    // 2. 장치(Device)와 스왑 체인 생성
    // ...

    // 3. 렌더 타겟 뷰(Render Target View) 생성
    // ...

    // 4. 렌더 타겟 설정
    // ...

    // 5. 뷰포트(Viewport) 설정
    // ...

    // --- 사각형 그리기를 위한 초기화 시작 ---

    // 6. 셰이더 컴파일 및 생성
    ID3DBlob* pVSBlob = nullptr;
    ID3DBlob* pPSBlob = nullptr;
    ID3DBlob* pErrorBlob = nullptr;

    // 셰이더 코드를 정의합니다.
    const char* shaderCode = R"(
        struct VS_INPUT
        {
            float4 Pos : POSITION;
        };

        struct PS_INPUT
        {
            float4 Pos : SV_POSITION;
        };

        PS_INPUT VS( VS_INPUT input )
        {
            PS_INPUT output = (PS_INPUT)0;
            output.Pos = input.Pos;
            return output;
        }

        float4 PS( PS_INPUT input ) : SV_Target
        {
            // 노란색 사각형을 그립니다.
            return float4( 1.0f, 1.0f, 0.0f, 1.0f );
        }
    )";

    // Vertex Shader 컴파일
    hr = D3DCompile(shaderCode, strlen(shaderCode), NULL, NULL, NULL, "VS", "vs_4_0", 0, 0, &pVSBlob, &pErrorBlob);
    if (FAILED(hr)) {
        if (pErrorBlob) pErrorBlob->Release();
        return hr;
    }
    // Vertex Shader 생성
    hr = g_pd3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &g_pVertexShader);
    if (FAILED(hr)) {
        pVSBlob->Release();
        return hr;
    }

    // Pixel Shader 컴파일
    hr = D3DCompile(shaderCode, strlen(shaderCode), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pErrorBlob);
    if (FAILED(hr)) {
        if (pErrorBlob) pErrorBlob->Release();
        pVSBlob->Release(); // 이미 성공한 VS Blob도 해제
        return hr;
    }
    // Pixel Shader 생성
    hr = g_pd3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &g_pPixelShader);
    pPSBlob->Release();
    if (FAILED(hr)) {
        pVSBlob->Release();
        return hr;
    }

    // 7. 입력 레이아웃(Input Layout) 생성
    // 정점 데이터가 어떤 형식인지(예: 위치, 색상, 텍스처 좌표 등) GPU에 알려줍니다.
    D3D11_INPUT_ELEMENT_DESC layout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    UINT numElements = ARRAYSIZE(layout);
    hr = g_pd3dDevice->CreateInputLayout(layout, numElements, pVSBlob->GetBufferPointer(),
        pVSBlob->GetBufferSize(), &g_pVertexLayout);
    pVSBlob->Release(); // 이제 필요 없으므로 해제
    if (FAILED(hr))
        return hr;

    // 8. 정점 버퍼(Vertex Buffer) 생성
    // 화면 좌표계(-1.0 ~ 1.0)를 사용하는 사각형의 정점 4개 정의
    float vertices[] =
    {
        -0.5f,  0.5f, 0.5f, // Top-Left
         0.5f,  0.5f, 0.5f, // Top-Right
        -0.5f, -0.5f, 0.5f, // Bottom-Left
         0.5f, -0.5f, 0.5f, // Bottom-Right
    };

    D3D11_BUFFER_DESC bd = {};
    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(vertices);
    bd.BindFlags = D3D11_BIND_VERTEX_BUFFER;

    D3D11_SUBRESOURCE_DATA InitData = {};
    InitData.pSysMem = vertices;
    hr = g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pVertexBuffer);
    if (FAILED(hr))
        return hr;

    // 9. 인덱스 버퍼(Index Buffer) 생성
    // 4개의 정점을 사용하여 2개의 삼각형(사각형)을 만드는 순서 정의
    WORD indices[] =
    {
        0, 1, 2, // 첫 번째 삼각형 (TL, TR, BL)
        2, 1, 3, // 두 번째 삼각형 (BL, TR, BR)
    };

    bd.Usage = D3D11_USAGE_DEFAULT;
    bd.ByteWidth = sizeof(indices);
    bd.BindFlags = D3D11_BIND_INDEX_BUFFER;
    InitData.pSysMem = indices;
    hr = g_pd3dDevice->CreateBuffer(&bd, &InitData, &g_pIndexBuffer);
    if (FAILED(hr))
        return hr;

    return S_OK;
}
```

> 함수에서 사각형을 그리는 데 필요한 모든 리소스를 초기화합니다. 정점/인덱스 버퍼 생성, 셰이더 컴파일 및 생성, 입력 레이아웃 설정 코드가 추가됩니다.

기존에 생성됐던 `InitDevice()` 함수에 사각형 관련 코드 추가. 셰이더라든지, 좌표라든지 하드코딩된 것 같은데, 일단 대충 훑어 읽고 넘어간다.  

### 4.3 Render 함수 수정

```cpp
// --- 매 프레임 호출되는 렌더링 함수 ---
void Render()
{
    // 1. 렌더 타겟 지우기
    // 매 프레임 새로 그리기 위해 이전 프레임의 내용을 깨끗하게 지웁니다.
    float ClearColor[4] = { 0.0f, 0.125f, 0.3f, 1.0f }; // RGBA (진한 파란색)
    g_pImmediateContext->ClearRenderTargetView(g_pRenderTargetView, ClearColor);

    // --- 여기에 모든 2D 객체를 그리는 코드가 들어갑니다 ---
    
    // 2. 렌더링 파이프라인 설정
    // 입력 레이아웃 설정
    g_pImmediateContext->IASetInputLayout(g_pVertexLayout);
    // 정점 및 인덱스 버퍼 설정
    UINT stride = sizeof(float) * 3;
    UINT offset = 0;
    g_pImmediateContext->IASetVertexBuffers(0, 1, &g_pVertexBuffer, &stride, &offset);
    g_pImmediateContext->IASetIndexBuffer(g_pIndexBuffer, DXGI_FORMAT_R16_UINT, 0);
    // 기본 도형 타입 설정 (삼각형 리스트)
    g_pImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    // 셰이더 설정
    g_pImmediateContext->VSSetShader(g_pVertexShader, nullptr, 0);
    g_pImmediateContext->PSSetShader(g_pPixelShader, nullptr, 0);

    // 3. 사각형 그리기
    // 6개의 인덱스를 사용하여 그립니다.
    g_pImmediateContext->DrawIndexed(6, 0, 0);

    // 4. 결과물 표시
    // 후면 버퍼(Back Buffer)에 그려진 내용을 화면(Front Buffer)으로 넘깁니다.
    g_pSwapChain->Present(1, 0);
}
```

> Render 함수에서 매 프레임 사각형을 그리도록 명령합니다. 배경을 지운 후, 설정한 셰이더와 버퍼를 사용하여 DrawIndexed를 호출합니다.

마찬가지로 기존에 생성됐던 `Render()` 함수에 사각형 관련 코드 추가. 2번, 3번이 추가됐다. 기존에는 진한 파란색 배경만 그렸었는데, 여러 가지 설정을 추가해 사각형도 그려낸다.  

###

```cpp
// --- DirectX 리소스 해제 함수 ---
void CleanupDevice()
{
    // 생성된 순서의 역순으로 리소스를 해제하는 것이 안전합니다.
    // ...
    if (g_pIndexBuffer) g_pIndexBuffer->Release();
    if (g_pVertexBuffer) g_pVertexBuffer->Release();
    if (g_pVertexLayout) g_pVertexLayout->Release();
    if (g_pVertexShader) g_pVertexShader->Release();
    if (g_pPixelShader) g_pPixelShader->Release();
    // ...
}
```

> 프로그램 종료 시 새로 추가된 리소스들을 해제하도록 CleanupDevice 함수를 수정합니다.

음, C++은 언매니지드 언어였지. 사각형을 그리기 위해 추가했던 리소스들이 `CleanupDevice()`에서 해제되도록 코드를 추가한다.  

### 실행 결과

![251009-165625](/assets/img/post/works/maplestory-clone/251009-165625.gif)

노란 단무지 출력 성공 !  

## 5. 정리

---

일단 그리기는 성공했지만, 내가 짠 코드는 없기도 하고, 이해한 내용이 많지는 않아서 공부할 필요가 있다.  

{% include embed/youtube.html id = "NTvhVxSC_80" %}

[여기서 다운로드 하면 샘플 코드나 튜토리얼을 확인할 수 있다.](https://www.microsoft.com/en-us/download/details.aspx?id=6812)  

<https://megayuchi.com/2019/04/18/direct-x-프로그래밍-학습에-대한-조언/>

- [Direct X 프로그래밍 학습에 대한 조언](https://megayuchi.com/2019/04/18/direct-x-프로그래밍-학습에-대한-조언/)
