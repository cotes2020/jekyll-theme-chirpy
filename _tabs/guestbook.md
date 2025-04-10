---
layout: page
title: GUEST
icon: fas fa-comments
permalink: /guestbook/
comments: true
---

여기에 여러분의 이야기를 남겨주세요! ✨

> 예시:
> - 오늘 방문했습니다!
> - 좋은 정보 감사합니다 😄

<form method="POST" action="https://your-subdomain.workers.dev/api/comment">
  <input type="text" name="nickname" placeholder="닉네임" required><br>
  <textarea name="message" placeholder="메시지를 입력하세요" required></textarea><br>
  <button type="submit">방명록 남기기</button>
</form>



{% for entry in site.data.guestbook reversed %}
  <div class="guestbook-entry">
    <strong>{{ entry.name }}</strong> - <small>{{ entry.date }}</small>
    <p>{{ entry.message }}</p>
  </div>
{% endfor %}
