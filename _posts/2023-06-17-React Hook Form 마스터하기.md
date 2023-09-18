---
title: React Hook Form 마스터하기
date: 2023-06-17 20:00:00 +0900
categories:
  - JavaScript
tags:
  - React
  - HookForm
---

안녕하세요 여러분! 오늘은 빠르게 혁명을 일으키고 있는 `React Hook Form`을 함께 알아보도록 할게요. 요즘 개발자들 사이에서 핫한 이 주제, 과연 어떤 점에서 다들 찬사를 보내는지 한번 같이 파해쳐 보도록 합시다!

## 로그인 폼 만들기 🛠️

먼저, React Hook Form을 마스터하기 위해 기본적인 로그인 폼을 만들어볼까요? 아래와 같이 간단한 JSX 마크업을 시작으로 우리의 모험을 시작해봅시다!

```jsx
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
export default Form;
```

별도의 라이브러리 없이도 이 정도의 코드 작성은 가능하지 않을까요? 이제 여기에 `React Hook Form`의 힘을 더해보도록 합시다!

## React Hook Form 연결하기 🔗

자, 이제 우리의 폼에 `React Hook Form`을 연결해 볼 시간이에요! 그러려면 `useForm`이라는 훅을 사용해야 해요. 우리 로그인 폼에 이 라이브러리를 연결해볼까요?

```jsx
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
export default Form;
```

자, 이제 로그인 버튼을 누르면 입력한 데이터가 알림으로 표시되게 됩니다! 어떤가요, 생각보다 간단하지 않나요? 😊

## 중복 제출 방지하기 ⏳

때로는 사용자가 너무 빨리 버튼을 누르면서 폼이 여러 번 제출되는 문제가 발생할 수 있어요. 그래서 우리는 중복 제출을 방지하기 위한 작업을 추가해 보겠습니다!

```jsx
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
export default Form;
```

이제 로그인 버튼은 제출이 완료될 때까지 비활성화 됩니다. 사실 React Hook Form은 너무 빨라서 지연시간을 주지 않으면 버튼이 비활성화 되는 것을 볼 수 없어요! 😆

## 입력값 검증하기 🕵️‍♂️

마지막으로 우리는 입력값 검증을 추가해 볼 것입니다. 이메일과 비밀번호는 필수 항목이며, 각각 특정한 규칙을 충족해야 한다는 것을 우리 모두 알고 있죠?

```jsx
function LoginForm() {
  const {
    register,
    handleSubmit,
  } = useForm();
  return (
    <form
      onSubmit={handleSubmit((data) => {
        alert(JSON.stringify(data));
      })}
    >
      <label htmlFor="email">이메일</label>
      <input
        id="email"
        type="email"
        placeholder="test@email.com"
        {...register("email", { required: true, pattern: /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/ })}
      />
      <label htmlFor="password">비밀번호</label>
      <input
        id="password"
        type="password"
        placeholder="****************"
        {...register("password", { required: true, minLength: 8 })}
      />
      <button type="submit">
        로그인
      </button>
    </form>
  );
}
export default Form;
```

위 코드를 통해 이메일과 비밀번호 필드에 각각 필요한 검증 규칙을 추가했어요. 그러면 사용자가 유효하지 않은 데이터를 입력하려고 하면, 오류 메시지를 받게 됩니다! 

## 마무리 🎉

자, 여러분! 우리가 함께 React Hook Form 라이브러리를 이용하여 로그인 폼을 만들었습니다. 어떤가요? 생각보다 간단하고 재미있지 않나요? 🎉
