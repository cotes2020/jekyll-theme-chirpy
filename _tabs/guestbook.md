---
layout: page
title: GUEST
icon: fas fa-comments
permalink: /comments/
comments: true
---

여기에 여러분의 이야기를 남겨주세요! ✨

> 예시:
> - 오늘 방문했습니다!
> - 좋은 정보 감사합니다 😄

<style>
.comment-form {
  display: flex;
  flex-direction: column;
  gap: 0.75em;
  max-width: 500px;
}
.comment-form input,
.comment-form textarea {
  padding: 0.5em;
  border: 1px solid var(--border-color);
  border-radius: 0.5em;
  background-color: var(--bg-color);
  color: var(--text-color);
}
.comment-form button {
  padding: 0.5em;
  border: none;
  border-radius: 0.5em;
  background-color: var(--btn-bg-color);
  color: var(--btn-text-color);
  cursor: pointer;
}
.comment-form button:hover {
  background-color: var(--btn-hover-bg-color);
}
</style>

<form method="POST" action="https://comment-w-guestbook.lanitoous.workers.dev/api/handle/form">
  <input type="text" name="fields[name]" placeholder="Name" required>
  <textarea name="fields[message]" placeholder="Your Comment" required></textarea>
  <button type="submit">남기기</button>
</form>

<dr/>
한숨
<dr/>

<---댓글란--->

<style>
.comment-bubble {
  max-width: 70%;
  margin: 0.5em 0;
  padding: 0.75em 1em;
  border-radius: 1em;
  background-color: var(--card-bg);
  color: var(--text-color);
  position: relative;
  box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.comment-bubble::after {
  content: "";
  position: absolute;
  left: 1em;
  bottom: -10px;
  width: 0;
  height: 0;
  border: 10px solid transparent;
  border-top-color: var(--card-bg);
}
.comment-name {
  font-weight: bold;
}
.comment-date {
  font-size: 0.8em;
  color: var(--text-muted-color);
}
</style>

{% for entry in site.data.comments reversed %}
  <div class="comment-bubble">
    <div class="comment-name">{{ entry.name }}</div>
    <div class="comment-date">{{ entry.date | date: "%Y년 %m월 %d일 %H시 %M분" }}</div>
    <p>{{ entry.message }}</p>
  </div>
{% endfor %}
