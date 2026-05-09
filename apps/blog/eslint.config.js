import { defineConfig, globalIgnores } from 'eslint/config';
import js from '@eslint/js';
import globals from 'globals';
import tseslint from 'typescript-eslint';

// fork 확장 (KL-031 B1 + B1.1) — chirpy upstream v7.5.0 root config 위에 typescript-eslint
// type-aware lint. 이유: 본 fork 는 _javascript/ 를 .ts 로 마이그레이션 + tsconfig.json 의
// strict: true. recommendedTypeChecked 가 noFloatingPromises / await-thenable / no-misused-promises
// 같은 type 정보 필요한 룰까지 포함.

export default defineConfig([
  globalIgnores(['assets/*', 'node_modules/*', '_site/*']),
  js.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
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
      parserOptions: {
        project: './tsconfig.json',
        tsconfigRootDir: import.meta.dirname
      },
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
    },
    rules: {
      // KL-031 B1.1 부분 적용 — no-floating-promises 만 활성 (PWA SW 등록 / clipboard
      // 같은 진짜 빚 detect). 나머지 type-aware 룰 (no-unsafe-*, no-redundant-type-constituents,
      // no-unnecessary-type-assertion) 은 chirpy 의 외부 lib (d3 등) 타입 부재로 누적 error
      // → 점진 fix backlog (KL-031 B1.1 sub).
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off'
      // KL-031 B1.3 ✅ no-redundant-type-constituents 활성 (GLightboxInstance: unknown → object).
      // KL-031 B1.4 ✅ no-unnecessary-type-assertion 활성 (auto-fix 적용 후).
    }
  },
  // root .js (rollup.config.js / purgecss.js / eslint.config.js 자체) 는 tsconfig 의 include
  // (_javascript/**/*) 밖 → type-aware lint 비활성화 (parserOptions.project 충돌 방지).
  {
    files: ['rollup.config.js', 'purgecss.js', 'eslint.config.js'],
    ...tseslint.configs.disableTypeChecked
  }
]);
