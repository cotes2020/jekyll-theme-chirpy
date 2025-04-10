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

<form method="POST" action="https://comment-w-guestbook.lanitoous.workers.dev/api/handle/form">
  <input type="text" name="fields[name]" required>
  <textarea name="fields[message]" required></textarea>
  <input type="hidden" name="fields[slug]" value="guestbook">
  <input type="hidden" name="options[origin]" value="https://lanitoous.github.io">
  <button type="submit">Submit</button>
</form>





{% for entry in site.data.guestbook reversed %}
  <div class="guestbook-entry">
    <strong>{{ entry.name }}</strong> - <small>{{ entry.date }}</small>
    <p>{{ entry.message }}</p>
  </div>
{% endfor %}
