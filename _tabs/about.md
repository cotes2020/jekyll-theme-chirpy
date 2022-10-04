---
title: About
icon: bi bi-info-circle
order: 4
---

> Add Markdown syntax content to file `_tabs/about.md`{: .filepath } and it will show up on this page.
> {: .prompt-tip }

<div class="row">
{% include about/skills.html title="Programming Skills" source=site.data.skills.programming %}
{% include about/skills.html title="Other Skills" source=site.data.skills.other %}
</div>

<div class="row">
{% include about/certs.html title="Certifications" source=site.data.certs %}
</div>

<div class="row">
{% include about/timeline.html %}
</div>
