import { defineCollection, z } from 'astro:content';

const posts = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    categories: z.array(z.string()).optional(),
    tags: z.array(z.string()).optional(),
    author: z.string().optional(),
    description: z.string().optional(),
    image: z.object({
      path: z.string(),
      alt: z.string().optional(),
      lqip: z.string().optional(),
      no_bg: z.boolean().optional(),
    }).optional(),
    pin: z.boolean().optional(),
    hidden: z.boolean().optional(),
    toc: z.boolean().default(true),
    comments: z.boolean().default(true),
    math: z.boolean().optional(),
    mermaid: z.boolean().optional(),
    media_subpath: z.string().optional(),
    render_with_liquid: z.boolean().optional(),
  }),
});

const tabs = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    icon: z.string(),
    order: z.number(),
  }),
});

export const collections = { posts, tabs };
