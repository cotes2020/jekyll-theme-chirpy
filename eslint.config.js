import { defineConfig, globalIgnores } from 'eslint/config';
import js from '@eslint/js';
import globals from 'globals';

export default defineConfig([
  globalIgnores(['assets/*', 'node_modules/*', '_site/*']),
  js.configs.recommended,
  {
    rules: {
      semi: ['error', 'always'],
      quotes: ['error', 'single']
    },
    languageOptions: {
      globals: {
        ...globals.browser,
        ...globals.node
      }
    }
  },
  {
    files: ['_javascript/**/*.js'],
    languageOptions: {
      globals: {
        ...globals.serviceworker,
        ClipboardJS: 'readonly',
        GLightbox: 'readonly',
        Theme: 'readonly',
        dayjs: 'readonly',
        mermaid: 'readonly',
        tocbot: 'readonly',
        swconf: 'readonly'
      }
    }
  }
]);
