---
id: plugin-development
title: Plugin Development
description: Documentation on Plugin Development
---

# backstage


- [backstage](#backstage)
  - [basic](#basic)
  - [Backstage plugins](#backstage-plugins)
    - [Intro to plugins](#intro-to-plugins)
    - [Existing plugins](#existing-plugins)
    - [Suggesting a plugin](#suggesting-a-plugin)
    - [Integrate into the Software Catalog](#integrate-into-the-software-catalog)
    - [Creating a plugin](#creating-a-plugin)
    - [Plugin Development](#plugin-development)
      - [Developing guidelines](#developing-guidelines)
      - [Plugin concepts / API](#plugin-concepts--api)
        - [Routing](#routing)
      - [structure-of-a-plugin](#structure-of-a-plugin)
        - [Folder structure](#folder-structure)
        - [Base files](#base-files)
        - [The plugin file](#the-plugin-file)
        - [Components](#components)
        - [Connecting the plugin to the Backstage app](#connecting-the-plugin-to-the-backstage-app)
        - [Talking to the outside world](#talking-to-the-outside-world)

ref:
- https://github.com/backstage/backstage


---

## basic


```bash

nvm install node


npx @backstage/create-app
cd my-backstage-app
yarn dev
```



---

## Backstage plugins

---

### Intro to plugins

- Backstage is a single-page application composed of a set of plugins.

- Our goal for the plugin ecosystem is that the definition of a plugin is flexible enough to allow you to expose pretty much any kind of infrastructure or software development tool as a plugin in Backstage.

- Backstage plugins provide features to a Backstage App.

- Each plugin is treated as a self-contained web app and can include almost any type of content. Plugins all use a common set of platform APIs and reusable UI components. Plugins can fetch data from external sources using the regular browser APIs or by depending on external modules to do the work.

![plugin](/assets/img/backstage-assets/my-plugin_screenshot.png)

### Existing plugins

- Open source plugins that you can add to the Backstage deployment can be found at: https://backstage.io/plugins


### Suggesting a plugin

- If start developing a plugin that you aim to release as open source, create a [new Issue](https://github.com/backstage/backstage/issues/new?labels=plugin&template=plugin_template.md&title=%5BPlugin%5D+THE+PLUGIN+NAME).

### Integrate into the Software Catalog

- If the plugin isn't supposed to live as a standalone page, but rather needs to be presented as a part of a Software Catalog (e.g. a separate tab or a card on an "Overview" tab), then check out
[the instruction]() on how to do it.

---

### Creating a plugin

- run `yarn install` and installed dependencies
- run `yarn new --select plugin`
  - a shortcut to invoking the [`backstage-cli new --select plugin`] from the root of the project.
  - This will create a new Backstage Plugin based on the ID that was provided. It will be built and added to the Backstage App automatically.

![](/assets/img/backstage-assets/getting-started/create-plugin_output.png)


> If the Backstage App is already running (with `yarn start` or `yarn dev`) you should be able to see the default page for the new plugin directly by navigating to `http://localhost:3000/my-plugin`.

![](/assets/img/backstage-assets/my-plugin_screenshot.png)

- You can also serve the plugin in isolation by
  - running `yarn start` in the plugin directory.
  - Or using the `yarn workspace` command

```bash
yarn workspace @backstage/plugin-my-plugin start
# Also supports --check
```

- This method of serving the plugin provides quicker iteration speed and a faster startup and hot reloads. It is only meant for local development, and the setup for it can be found inside the plugin's `dev/` directory.


---

### Plugin Development

---

#### Developing guidelines

- Consider writing plugins in `TypeScript`.
- Plan the directory structure of the plugin so that it becomes easy to manage.
- Prefer using the [Backstage components](https://backstage.io/storybook), otherwise go with [Material UI](https://material-ui.com/).
- Check out the shared Backstage APIs before building a new one.

---
---

#### Plugin concepts / API

##### Routing

- Each plugin can export routable extensions, which are then imported into the app and mounted at a path.

- need a `RouteRef` instance to serve as the mount point of the extensions.
  - This can be used within the own plugin to create a link to the extension page using `useRouteRef`, as well as for other plugins to link to the extension.

  - It is best to place these in a separate top-level `src/routes.ts` file, in order to avoid import cycles, for example like this:

```tsx
/* src/routes.ts */
import { createRouteRef } from '@backstage/core-plugin-api';

// Note: This route ref is for internal use only, don't export it from the plugin
export const rootRouteRef = createRouteRef({
  title: 'Example Page',
});
```

- Now that we have a `RouteRef`,
  - import it into `src/plugin.ts`,
  - create our plugin instance with `createPlugin`,
  - create and wrap routable extension using `createRoutableExtension` from `@backstage/core-plugin-api`:

```tsx
/* src/plugin.ts */
import { createPlugin, createRouteRef } from '@backstage/core-plugin-api';
import ExampleComponent from './components/ExampleComponent';

// Create a plugin instance
// and export this from the plugin package
export const examplePlugin = createPlugin({
  id: 'example',
  routes: {
    root: rootRouteRef,
    // This is where the route ref should be exported for usage in the app
  },
});

// This creates a routable extension, which are typically full pages of content.
// Each extension should also be exported from the plugin package.
export const ExamplePage = examplePlugin.provide(
  createRoutableExtension({
    name: 'ExamplePage',
    // The component needs to be lazy-loaded.
    // It's what will actually be rendered in the end.
    component: () =>
      import('./components/ExampleComponent').then(m => m.ExampleComponent),
    // This binds the extension to this route ref,
    // which allows for routing within and across plugin extensions
    mountPoint: rootRouteRef,
  }),
);
```

- This extension can then be imported and used in the app as follow, typically placed within the top-level `<FlatRoutes>`:

```tsx
<Route path="/any-path" element={<ExamplePage />} />
```

---

#### structure-of-a-plugin

##### Folder structure

The new plugin should look something like:

```bash
new-plugin/
    dev/
        index.ts
    node_modules/
    src/
        components/
            ExampleComponent/
                ExampleComponent.test.tsx
                ExampleComponent.tsx
                index.ts
            ExampleFetchComponent/
                ExampleFetchComponent.test.tsx
                ExampleFetchComponent.tsx
                index.ts
        index.ts
        plugin.test.ts
        plugin.ts
        routes.ts
        setupTests.ts
    .eslintrc.js
    package.json
    README.md
```

- a plugin looks like a mini project on it's own with a `package.json` and a `src` folder. this is because we want plugins to be separate packages. This makes it possible to ship plugins on npm and it lets you work on a plugin in isolation, without loading all the other plugins in a potentially big Backstage app.

- The `index.ts` files are there to let us import from the folder path and not specific files. It's a way to have control over the exports in one file per folder.

##### Base files

- You get a readme to populate with info about the plugin and a
package.json to declare the plugin dependencies, metadata and scripts.

##### The plugin file

- In the `src` folder, Check out the `plugin.ts`:

```jsx
import {
  createPlugin,
  createRoutableExtension,
} from '@backstage/core-plugin-api';

import { rootRouteRef } from './routes';

export const examplePlugin = createPlugin({
  id: 'example',
  routes: {
    root: rootRouteRef,
  },
});

export const ExamplePage = examplePlugin.provide(
  createRoutableExtension({
    name: 'ExamplePage',
    component: () =>
      import('./components/ExampleComponent').then(m => m.ExampleComponent),
    mountPoint: rootRouteRef,
  }),
);
```

This is where the plugin is created and where it creates and exports extensions
that can be imported and used the app. See reference docs for
[`createPlugin`](../reference/core-plugin-api.createplugin.md) or introduction to
the new [Composability System](./composability.md).

##### Components

The generated plugin includes two example components to showcase how we
structure our plugins. There are usually one or multiple page components and
next to them you can split up the UI in as many components as you feel like.

We have the `ExampleComponent` to show an example Backstage page component. The
`ExampleFetchComponent` showcases the common task of making an async request to
a public API and plot the response data in a table using Material UI components.

You may tweak these components, rename them and/or replace them completely.

##### Connecting the plugin to the Backstage app

There are two things needed for a Backstage app to start making use of a plugin.

1. Add plugin as dependency in `app/package.json`
2. Import and use one or more plugin extensions, for example in
   `app/src/App.tsx`.

Luckily both of these steps happen automatically when you create a plugin with
the Backstage CLI.

##### Talking to the outside world

If the plugin needs to communicate with services outside the Backstage
environment you will probably face challenges like CORS policies and/or
backend-side authorization. To smooth this process out you can use proxy -
either the one you already have (like Nginx, HAProxy, etc.) or the proxy-backend
plugin that we provide for the Backstage backend.
[Read more](https://github.com/backstage/backstage/blob/master/plugins/proxy-backend/README.md)








.
