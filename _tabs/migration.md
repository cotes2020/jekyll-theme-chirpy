---
icon: fas fa-route
order: 8
mermaid: true
---
{% include lang.html %}

마이그레이션 - 타사 ERP 데이터 추출 가이드

<div class="category-root">
  {% assign tag_groups = "회계,전자결재" | split: "," %}
  {% for tag in tag_groups %}
    {% assign posts = site.posts | where_exp: "p", "p.tags contains '마이그레이션' and p.tags contains tag" %}
    {% if posts.size > 0 %}
      <div class="folder-box">
        <div class="folder-header open"
             onclick="this.classList.toggle('open'); this.nextElementSibling.classList.toggle('hidden')">
          <i class="far fa-folder-open fa-fw text-muted"></i>
          {{ tag }} 마이그레이션
          <small class="text-muted">{{ posts | size }} 포스트</small>
          <i class="fas fa-chevron-down arrow-icon"></i>
        </div>
        <ul class="folder-list">
          {% for post in posts %}
            <li class="folder-item">
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            </li>
          {% endfor %}
        </ul>
      </div>
    {% endif %}
  {% endfor %}
</div>

<style>
.category-root {
  margin-top: 1.5rem;
}
.folder-box {
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 6px;
  margin-bottom: 1rem;
  overflow: hidden;
}
.folder-header {
  display: flex;
  align-items: center;
  padding: 0.75rem 1rem;
  font-weight: 600;
  cursor: pointer;
  background: #2a2a2a;
  border-bottom: 1px solid #333;
}
.folder-header i {
  margin-right: 0.5rem;
}
.folder-header .text-muted {
  margin-left: 0.5rem;
  color: #aaa;
  font-size: 0.9rem;
}
.arrow-icon {
  margin-left: auto;
  color: #aaa;
  transition: transform 0.2s ease;
}
.folder-header.open .arrow-icon {
  transform: rotate(180deg);
}
.folder-list {
  list-style: none;
  margin: 0;
  padding: 0;
  border-top: 1px solid #333;
}
.folder-list.hidden {
  display: none;
}
.folder-item {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #333;
}
.folder-item:last-child {
  border-bottom: none;
}
.folder-item a {
  color: #59afff;
  text-decoration: none;
}
.folder-item a:hover {
  text-decoration: underline;
}
</style>
