# Astro Chirpy

A minimal, responsive, and feature-rich Astro theme for technical writing, migrated from the popular [Jekyll Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy).

## 🚀 Features

- ⚡ **Lightning Fast** - Built with Astro for optimal performance
- 🎨 **Clean Design** - Minimal and elegant interface
- 🌓 **Dark Mode** - Automatic theme switching with manual toggle
- 📱 **Fully Responsive** - Works perfectly on all devices
- 🔍 **SEO Optimized** - Built-in SEO best practices
- 📝 **MDX Support** - Write content in Markdown or MDX
- 🏷️ **Tags & Categories** - Organize your content effectively
- 🎯 **TypeScript Ready** - Full TypeScript support
- ♿ **Accessible** - WCAG compliant

## 📦 Project Structure

```
astro-chirpy/
├── public/              # Static assets
├── src/
│   ├── components/      # Astro components
│   │   ├── Head.astro
│   │   ├── Sidebar.astro
│   │   ├── Topbar.astro
│   │   ├── Footer.astro
│   │   └── ...
│   ├── content/         # Content collections
│   │   ├── posts/       # Blog posts
│   │   └── tabs/        # Navigation tabs
│   ├── data/            # Data files (YAML)
│   │   ├── authors.yml
│   │   ├── contact.yml
│   │   ├── locales/     # 33 language translations
│   │   └── ...
│   ├── layouts/         # Astro layouts
│   │   ├── BaseLayout.astro
│   │   └── PostLayout.astro
│   ├── pages/           # Routes
│   │   ├── index.astro  # Home page
│   │   ├── [slug].astro # Tab pages
│   │   └── posts/
│   │       └── [...slug].astro  # Blog posts
│   ├── scripts/         # TypeScript modules
│   │   ├── main.ts
│   │   └── theme.ts
│   ├── styles/          # SCSS styles
│   │   ├── abstracts/
│   │   ├── base/
│   │   ├── components/
│   │   ├── layout/
│   │   ├── pages/
│   │   └── themes/
│   └── config.ts        # Site configuration
├── astro.config.mjs     # Astro configuration
├── tsconfig.json        # TypeScript configuration
└── package.json
```

## 🛠️ Getting Started

### Prerequisites

- Node.js 18+
- npm or yarn

### Installation

1. **Install dependencies:**

```bash
npm install
```

2. **Configure your site:**

Edit `src/config.ts` to customize your site settings:

```typescript
export const SITE = {
  title: 'Your Site Title',
  tagline: 'Your tagline',
  description: 'Site description',
  url: 'https://your-site.com',
  // ... more settings
};
```

3. **Run the development server:**

```bash
npm run dev
```

Visit `http://localhost:4321` to see your site!

## 📝 Creating Content

### Writing Posts

Create a new Markdown file in `src/content/posts/`:

```markdown
---
title: Your Post Title
date: 2024-01-15T10:00:00+00:00
categories: [Category1, Category2]
tags: [tag1, tag2, tag3]
description: Brief description of your post
pin: false
---

Your content here...
```

#### Frontmatter Options

- `title` (required) - Post title
- `date` (required) - Publication date
- `categories` - Array of categories
- `tags` - Array of tags
- `description` - Post description for SEO and previews
- `image` - Featured image object
  - `path` - Image path
  - `alt` - Alt text
  - `lqip` - Low Quality Image Placeholder
- `pin` - Pin post to top of home page
- `hidden` - Hide post from listings
- `toc` - Enable/disable table of contents
- `comments` - Enable/disable comments
- `author` - Author ID from authors.yml

### Creating Tabs

Add a new tab in `src/content/tabs/`:

```markdown
---
title: Tab Title
icon: fas fa-icon-name
order: 1
---

Tab content...
```

## 🎨 Customization

### Styling

Styles are organized in `src/styles/`:

- `abstracts/` - Variables, mixins, functions
- `base/` - Base styles and typography
- `components/` - Component styles
- `layout/` - Layout styles
- `pages/` - Page-specific styles
- `themes/` - Light and dark themes

### Configuration

Main site configuration is in `src/config.ts`:

```typescript
// Site settings
export const SITE = { ... };

// Social media
export const SOCIAL = { ... };

// Analytics
export const ANALYTICS = { ... };

// Comments
export const COMMENTS = { ... };

// Features
export const FEATURES = { ... };
```

## 🚢 Deployment

### Build for Production

```bash
npm run build
```

The built site will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

### Deploy

You can deploy to any static hosting service:

- **Vercel**: Connect your repo for automatic deployments
- **Netlify**: Drag and drop `dist/` folder or connect repo
- **GitHub Pages**: Use GitHub Actions
- **Cloudflare Pages**: Connect your repo

## 📚 Commands

| Command | Action |
|---------|--------|
| `npm run dev` | Start development server |
| `npm run build` | Build production site |
| `npm run preview` | Preview production build |
| `npm run astro` | Run Astro CLI commands |
| `npm run lint:js` | Lint JavaScript/TypeScript |
| `npm run lint:scss` | Lint SCSS files |

## 🔄 Migration from Jekyll Chirpy

### Key Differences

1. **No Ruby/Jekyll** - Pure JavaScript/TypeScript ecosystem
2. **Content Collections** - Type-safe content with Astro's content collections
3. **Component-based** - Reusable Astro components instead of Liquid includes
4. **Faster Builds** - Significantly faster build times with Astro
5. **Modern Tooling** - Vite, TypeScript, and modern JavaScript

### What's Migrated

✅ Core layouts and design
✅ Responsive sidebar navigation
✅ Dark/Light theme toggle
✅ Post listings with pagination support
✅ Categories and tags
✅ SCSS styles
✅ 33 language locales
✅ SEO optimization
✅ Analytics integration (Google Analytics, etc.)
✅ Social media integration

### What's Different

- Search functionality (to be implemented)
- PWA features (to be implemented)
- Comment systems (to be integrated)
- Some Jekyll-specific features

### Migration Steps

If you're migrating from Jekyll Chirpy:

1. Copy your posts from `_posts/` to `src/content/posts/`
2. Update post frontmatter (mostly compatible)
3. Copy custom tabs from `_tabs/` to `src/content/tabs/`
4. Update `src/config.ts` with your `_config.yml` settings
5. Copy any custom assets to `public/`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the original [Jekyll Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy) for details.

## 🙏 Credits

- Original theme: [Jekyll Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) by [Cotes Chung](https://github.com/cotes2020)
- Built with [Astro](https://astro.build)
- Icons by [Font Awesome](https://fontawesome.com)
- Styling with [Bootstrap](https://getbootstrap.com) and custom SCSS

## 📞 Support

- [Documentation](https://astro.build)
- [Astro Discord](https://astro.build/chat)
- [GitHub Issues](https://github.com/your-repo/issues)

---

Made with ❤️ using [Astro](https://astro.build)
