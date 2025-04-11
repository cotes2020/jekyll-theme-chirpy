---
layout: page
title: GUEST
icon: fas fa-comments
permalink: /comments/
comments: true
<<<<<<< Updated upstream
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
=======
order: 5
---

방명록 ✨

> 주의사항 :
> - 글이 올라가는 데 시간이 오래 걸릴 수 있습니다



<!-- guestbook.md -->
<div style="margin-bottom: 1em;">
  <form method="POST" action="https://comment-w-guestbook.lanitoous.workers.dev/api/handle/form" style="display: flex; flex-direction: column; gap: 0.5em; max-width: 400px;">
    <input type="text" name="name" placeholder="Name" required style="padding: 0.5em; border: 1px solid #ccc; border-radius: 8px;">
    <textarea name="message" placeholder="Your Comment" required rows="3" style="padding: 0.5em; border: 1px solid #ccc; border-radius: 8px;"></textarea>
    <button type="submit" style="padding: 0.5em; background: #333; color: #fff; border: none; border-radius: 8px;">남기기</button>
  </form>
</div>

---

***
<br/>

<!---댓글란--->


{% assign all_comments = site.data.comments %}
{% for filename in all_comments %}
  {% assign comment = filename[1] %}
  <div class="comment-bubble">
    <strong>{{ comment.name }}</strong>
    <div class="comment-date">
      {{ comment.date | date: "%Y년 %m월 %d일 %H:%M" }}
    </div>
    <div class="comment-message">{{ comment.message }}</div>
  </div>
{% endfor %}



<style>
.comment-list {
  display: flex;
  flex-direction: column;
  gap: 1em;
  margin-top: 1.5em;
}

.comment-bubble {
  position: relative;
  max-width: 90%;
  padding: 1em;
  border-radius: 1em;
  background-color: var(--bubble-bg, #f1f1f1);
  color: var(--text-color, #333);
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.05);
}

>>>>>>> Stashed changes
.comment-bubble::after {
  content: "";
  position: absolute;
  left: 1em;
  bottom: -10px;
  width: 0;
  height: 0;
  border: 10px solid transparent;
<<<<<<< Updated upstream
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
=======
  border-top-color: var(--bubble-bg, #f1f1f1);
}

.comment-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 0.5em;
  font-size: 0.9em;
  color: #666;
}

.comment-name {
  font-weight: bold;
}

.comment-message {
  white-space: pre-wrap;
}

.comment-bubble {
  background-color: var(--card-bg, #f4f4f4);
  color:  var(--text-color, #000);
  padding: 1em;
  margin: 1em 0;
  border-radius: 15px;
  max-width: 500px;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
  transition: background-color 0.3s ease, color 0.3s ease;


  .comment-date {
    font-size: 0.85em;
    color: #555;
    margin-top: 0.25em;
  }

  .comment-message {
    margin-top: 0.5em;
    white-space: pre-wrap;
  }
}


}

</style>
>>>>>>> Stashed changes
