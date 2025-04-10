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

<form method="POST" action="https://your-staticman-instance/v3/entry/github/your-username/your-repo/main/comments">
  <input type="text" name="fields[name]" placeholder="Your Name" required>
  <textarea name="fields[message]" placeholder="Your Comment" required></textarea>
  <input type="hidden" name="options[slug]" value="post-slug">
  <button type="submit">Submit</button>
</form>


{% for entry in site.data.guestbook reversed %}
  <div class="guestbook-entry">
    <strong>{{ entry.name }}</strong> - <small>{{ entry.date }}</small>
    <p>{{ entry.message }}</p>
  </div>
{% endfor %}
