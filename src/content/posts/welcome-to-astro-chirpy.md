---
title: Welcome to Astro Chirpy
date: 2024-01-15T10:00:00+00:00
categories: [Blogging, Tutorial]
tags: [astro, chirpy, migration]
description: A minimal, responsive and feature-rich Astro theme for technical writing, migrated from Jekyll Chirpy.
pin: true
---

## Welcome! 🎉

This is **Astro Chirpy**, a migration of the popular Jekyll Chirpy theme to Astro. This theme maintains the clean, minimal design while leveraging Astro's modern features for blazing-fast performance.

## Features

### Performance
- ⚡ Lightning-fast page loads with Astro
- 🎯 Zero JS by default
- 📦 Optimized bundle sizes

### Design
- 🎨 Clean, minimal design
- 🌓 Dark/Light mode toggle
- 📱 Fully responsive
- ♿ Accessible

### Developer Experience
- 🚀 Easy to customize
- 📝 MDX support
- 🔧 TypeScript ready
- 🎭 Component-based architecture

## Getting Started

To create a new post, add a markdown file to `src/content/posts/` with the following frontmatter:

```yaml
---
title: Your Post Title
date: 2024-01-15T10:00:00+00:00
categories: [Category1, Category2]
tags: [tag1, tag2]
description: Brief description of your post
---
```

## Syntax Highlighting

The theme supports syntax highlighting out of the box:

```javascript
function greet(name) {
  console.log(`Hello, ${name}!`);
}

greet('Astro Chirpy');
```

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
```

## Next Steps

1. Customize your site configuration in `src/config.ts`
2. Add your own posts to `src/content/posts/`
3. Modify the tabs in `src/content/tabs/`
4. Update the styling in `src/styles/`

Happy blogging! 📝
