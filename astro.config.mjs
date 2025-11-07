import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';

// https://astro.build/config
export default defineConfig({
  site: 'https://example.com', // Update this to your site URL
  integrations: [
    mdx(),
    sitemap(),
  ],
  markdown: {
    syntaxHighlight: 'shiki',
    shikiConfig: {
      theme: 'github-dark',
      wrap: true
    }
  },
  vite: {
    resolve: {
      alias: {
        '@': '/src'
      }
    }
  }
});
