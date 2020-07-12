/*!
  PV configuration and Javascript conversion.
*/

const proxyEndpoint = "{{ site.google_analytics.pv.proxy_endpoint }}";


{% if site.google_analytics.pv.cache and site.google_analytics.pv.enabled %}
  {% assign enabled = true %}
{% else %}
  {% assign enabled = false %}
{% endif %}

const pvCacheEnabled = {{ enabled }};
