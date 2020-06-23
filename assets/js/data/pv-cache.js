---
layout: compress
---

{%- capture pv_data -%}
  {%- if site.google_analytics.pv.cache and site.google_analytics.pv.enabled -%}
    {% include_relative _pageviews.json %}
  {%- endif -%}
{%- endcapture -%}

const pageviews = '{{ pv_data }}';
