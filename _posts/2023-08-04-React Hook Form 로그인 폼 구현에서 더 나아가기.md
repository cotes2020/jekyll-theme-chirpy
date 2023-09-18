---
title: React Hook Form 로그인 폼 구현에서 더 나아가기
date: 2023-08-04 20:00:00 +0900
categories:
  - React
tags:
  - ReactHookForm
---

안녕하세요, 여러분! 오늘은 React Hook Form 라이브러리로 로그인 폼을 만들면서 더 많은 것들을 배워볼 거에요. 말로만 듣던 로그인 폼, 드디어 직접 만들어보는 날이 왔습니다! 🎉

## 설치부터 시작! 🚀

먼저, 터미널을 열고 아래 명령어를 쳐주세요.

```bash
$ npm install react-hook-form
```

이제 라이브러리가 설치되었으니, 코드를 작성할 준비가 끝났습니다!

## JSX로 로그인 폼 만들기 🛠️

아래는 기본적인 로그인 폼을 JSX로 어떻게 만드는지 보여주는 코드입니다.

```javascript
function LoginForm() {
  return (
    <form>
      <label htmlFor="email">이메일</label>
      <input id="email" type="email" placeholder="test@email.com" />
      <label htmlFor="password">비밀번호</label>
      <input id="password" type="password" placeholder="****************" />
      <button type="submit">로그인</button>
    </form>
  );
}
export default LoginForm;
```

별다를 것 없는 코드이죠? 그런데 여기서 시작입니다!

## React Hook Form 연동하기 🔄

이제 React Hook Form을 이 로그인 폼과 연결해 봅시다! 아래 코드는 `useForm()` 훅을 사용해서 로그인 폼을 업그레이드하는 방법입니다.

```javascript
import { useForm } from "react-hook-form";

function LoginForm() {
  const { register, handleSubmit } = useForm();
  return (
    <form onSubmit={handleSubmit((data) => alert(JSON.stringify(data)))}>
      <label htmlFor="email">이메일</label>
      <input
        id="email"
        type="email"
        placeholder="test@email.com"
        {...register("email")}
      />
      <label htmlFor="password">비밀번호</label>
      <input
        id="password"
        type="password"
        placeholder="****************"
        {...register("password")}
      />
      <button type="submit">로그인</button>
    </form>
  );
}
export default LoginForm;
```

## 중복 제출? 이젠 안녕! 👋

로그인 폼에서 중복으로 제출하는 문제를 해결해봅시다! 아래 코드에서 `isSubmitting`을 이용하여 버튼을 비활성화시켜 이 문제를 해결합니다.

```javascript
import { useForm } from "react-hook-form";

function LoginForm() {
  const {
    register,
    handleSubmit,
    formState: { isSubmitting },
  } = useForm();
  return (
    <form
      onSubmit={handleSubmit(async (data) => {
        await new Promise((r) => setTimeout(r, 1000));
        alert(JSON.stringify(data));
      })}
    >
      <label htmlFor="email">이메일</label>
      <input
        id="email"
        type="email"
        placeholder="test@email.com"
        {...register("email")}
      />
      <label htmlFor="password">비밀번호</label>
      <input
        id="password"
        type="password"
        placeholder="****************"
        {...register("password")}
      />
      <button type="submit" disabled={isSubmitting}>
        로그인
      </button>
    </form>
  );
}
export default LoginForm;
```

## 입력값 검증! 🛡️

이메일과 비밀번호를 꼼꼼하게 검사해 봅시다. 아래 코드는 `register()` 함수에서 어떻게 입력값을 검증하는지를 보여줍니다.

```javascript
{
  ...register("email", { required: true, pattern: /^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,4}$/ })
}
{
  ...register("password", { required: true, minLength: 8 })
}
```

이제 이메일과 비밀번호를 잘못 입력하면 사용자에게 알려줍니다!

## 나만의 로그인 폼, 완성! 🎊

여러분, 이제 나만의 로그인 폼을 만들었습니다! 👏 React Hook Form 라이브러리를 이용해서 멋진 로그인 폼을 만들어 본 것이 어떠셨나요? 더 배우고 싶다면, 공식 문서도 잊지마세요! 📚
