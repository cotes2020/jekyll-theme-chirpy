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

<form method="POST" action="https://YOUR-STATICMAN-URL/v3/entry/github/lanitoous/lanitoous.github.io/guestbook">
  <label>닉네임</label><br>
  <input type="text" name="name" required><br><br>

  <label>메시지</label><br>
  <textarea name="message" rows="4" required></textarea><br><br>

  <button type="submit">방명록 남기기</button>
</form>

{% for entry in site.data.guestbook reversed %}
  <div class="guestbook-entry">
    <strong>{{ entry.name }}</strong> - <small>{{ entry.date }}</small>
    <p>{{ entry.message }}</p>
  </div>
{% endfor %}
