---
layout: archive
title: "Publications"
author_profile: true
permalink: /publications/
---

{% include base_path %}

You can also find my articles on <a href="https://scholar.google.com/citations?user=GHVdrBgAAAAJ&hl=en">my Google Scholar profile</a>.

{% for category in site.publication_category %}
  {% assign cat = category[0] %}
  {% assign has_posts = false %}
  {% for post in site.publications %}
    {% if post.category == cat %}{% assign has_posts = true %}{% endif %}
  {% endfor %}
  {% if has_posts %}
<h2>{{ category[1].title }}</h2>
    {% for post in site.publications reversed %}
      {% if post.category == cat %}
        {% include archive-single.html %}
      {% endif %}
    {% endfor %}
  {% endif %}
{% endfor %}
