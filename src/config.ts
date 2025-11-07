export const SITE = {
  title: 'Chirpy',
  tagline: 'A text-focused Jekyll theme',
  description: 'A minimal, responsive and feature-rich Jekyll theme for technical writing.',
  url: 'https://example.com', // Update this to your site URL
  baseurl: '',
  lang: 'en',
  timezone: 'Asia/Shanghai',
  cdn: 'https://chirpy-img.netlify.app',
  avatar: '/commons/avatar.jpg',
  social_preview_image: '',
  social: {
    name: 'your_full_name',
    email: 'example@domain.com',
  },
  github: {
    username: 'github_username',
  },
  twitter: {
    username: 'twitter_username',
  },
};

export const SOCIAL = {
  name: 'your_full_name',
  email: 'example@domain.com',
  links: [
    'https://twitter.com/username',
    'https://github.com/username',
  ],
};

export const GITHUB = {
  username: 'github_username',
};

export const TWITTER = {
  username: 'twitter_username',
};

export const ANALYTICS = {
  google: {
    id: '',
  },
  goatcounter: {
    id: '',
  },
  umami: {
    id: '',
    domain: '',
  },
  matomo: {
    id: '',
    domain: '',
  },
  cloudflare: {
    id: '',
  },
  fathom: {
    id: '',
  },
};

export const PAGEVIEWS = {
  provider: '', // 'goatcounter'
};

export const COMMENTS = {
  provider: '', // 'disqus' | 'utterances' | 'giscus'
  disqus: {
    shortname: '',
  },
  utterances: {
    repo: '',
    issue_term: '',
  },
  giscus: {
    repo: '',
    repo_id: '',
    category: '',
    category_id: '',
    mapping: 'pathname',
    strict: '0',
    input_position: 'bottom',
    lang: 'en',
    reactions_enabled: '1',
  },
};

export const FEATURES = {
  toc: true,
  theme_mode: '', // 'light' | 'dark' | '' (empty for system default with toggle)
  pwa: {
    enabled: true,
    cache: {
      enabled: true,
      deny_paths: [],
    },
  },
};

export const PAGINATION = {
  postsPerPage: 10,
};

export const WEBMASTER_VERIFICATIONS = {
  google: '',
  bing: '',
  alexa: '',
  yandex: '',
  baidu: '',
  facebook: '',
};
