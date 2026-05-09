import { defineConfig, globalIgnores } from 'eslint/config';
import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

// fork 확장 (KL-031 B1) — chirpy upstream v7.5.0 root config 위에 typescript-eslint
// 추가. 이유: 본 fork 는 _javascript/ 를 .ts 로 마이그레이션. upstream 의 .js 매칭 config
// 만으론 .ts 파일 lint 안 됨. tseslint.configs.recommended 가 .ts 파서 + 권장 룰.

export default defineConfig([
  globalIgnores(['assets/*', 'node_modules/*', '_site/*']),
  js.configs.recommended,
  ...tseslint.configs.recommended,
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
    files: ['_javascript/**/*.{js,ts}'],
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
