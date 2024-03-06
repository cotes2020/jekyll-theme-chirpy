---
title: DevPortal - Backstage
date: 2021-01-11 11:11:11 -0400
categories: [90DevPortal]
tags: [Backstage]
toc: true
image:
---

# Backstage

- [Backstage](#backstage)
  - [basic](#basic)
  - [architecture-overview](#architecture-overview)
    - [Terminology](#terminology)
    - [Overview](#overview)
      - [User Interface](#user-interface)
      - [Plugins and plugin backends](#plugins-and-plugin-backends)
      - [Installing plugins](#installing-plugins)
      - [Plugin architecture](#plugin-architecture)
        - [Standalone plugins](#standalone-plugins)
        - [Service backed plugins](#service-backed-plugins)
        - [Third-party backed plugins](#third-party-backed-plugins)
    - [Package Architecture](#package-architecture)
      - [Overview](#overview-1)
      - [Plugin Packages](#plugin-packages)
      - [Frontend Packages](#frontend-packages)
      - [Backend Packages](#backend-packages)
      - [Common Packages](#common-packages)
      - [Deciding where you place your code](#deciding-where-you-place-your-code)
    - [Databases](#databases)
    - [Cache](#cache)
      - [Use memory for cache](#use-memory-for-cache)
    - [Use memcache for cache](#use-memcache-for-cache)
    - [Use Redis for cache](#use-redis-for-cache)
    - [Containerization](#containerization)
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

Backstage is an open platform for building developer portals. Powered by a centralized software catalog, Backstage restores order to your microservices and infrastructure and enables your product teams to ship high-quality code quickly — without compromising autonomy.

Backstage unifies all your infrastructure tooling, services, and documentation to create a streamlined development environment from end to end.



```bash

nvm install node


npx @backstage/create-app
cd my-backstage-app
yarn dev
```

---

## architecture-overview

### Terminology

Backstage is constructed out of three parts. We separate Backstage in this way because we see three groups of contributors that work with Backstage in three different ways.

- **Core**
  - Base functionality built by core developers in the open source project.

- **App**
  - The app is an instance of a Backstage app that is deployed and tweaked.
  - The app ties together core functionality with additional plugins. The app is built and maintained by app developers, usually a productivity team within a company.

- **Plugins**
  - Additional functionality to make your Backstage app useful for your company.
  - Plugins can be specific to a company or open sourced and reusable. A
  - t Spotify we have over 100 plugins built by over 50 different teams. It has been very powerful to get contributions from various infrastructure teams added into a single unified developer experience.

### Overview

> The following diagram shows how Backstage might look when deployed inside a company which uses the `Tech Radar` plugin, the `Lighthouse` plugin, the `CircleCI` plugin and the `software catalog`.

3 main components in this architecture:

1. The **core** Backstage UI
2. The UI **plugins** and their backing services
3. **Databases**

Running this architecture in a real environment typically involves containerising the components. Various commands are provided for accomplishing this.

![The architecture of a basic Backstage application](/assets/img/backstage-assets/architecture-overview/backstage-typical-architecture.png)

#### User Interface

- The UI is a thin, client-side wrapper around a set of plugins. It provides some core UI components and libraries for shared activities such as config management. [[live demo](https://demo.backstage.io/catalog)]

![UI with different components highlighted](/assets/img/backstage-assets/architecture-overview/core-vs-plugin-components-highlighted.png)

Each plugin typically makes itself available in the UI on a dedicated URL. For example,
- the Lighthouse plugin is registered with the UI on `/lighthouse`. [[learn more](https://backstage.io/blog/2020/04/06/lighthouse-plugin)]
  - ![The lighthouse plugin UI](/assets/img/backstage-assets/architecture-overview/lighthouse-plugin.png)

- The CircleCI plugin is available on `/circleci`.
  - ![CircleCI Plugin UI](/assets/img/backstage-assets/architecture-overview/circle-ci.png)


#### Plugins and plugin backends

Each plugin is a client side application which mounts itself on the UI.
- Plugins are written in TypeScript or JavaScript.
- They each live in their own directory in `backstage/plugins`.
- For example, the source code for the lighthouse plugin is available at [backstage/plugins/lighthouse](https://github.com/backstage/backstage/tree/master/plugins/lighthouse).


#### Installing plugins

Plugins are typically installed as React components in your Backstage application.
- For example, [here](https://github.com/backstage/backstage/blob/master/packages/app/src/App.tsx) is a file that imports many full-page plugins in the Backstage sample app.

An example of one of these plugin components is the `CatalogIndexPage`
- a full-page view that allows you to browse entities in the Backstage catalog.
- It is installed in the app by importing it and adding it as an element like this:


```tsx
import { CatalogIndexPage } from '@backstage/plugin-catalog';

...

const routes = (
  <FlatRoutes>
    ...
    <Route path="/catalog" element={<CatalogIndexPage />} />
    ...
  </FlatRoutes>
);
```

Note that we use `"/catalog"` as our path to this plugin page, but we can choose any route we want for the page, as long as it doesn't collide with the routes that we choose for the other plugins in the app.

These components that are exported from plugins are referred to as `"Plugin Extension Components", or "Extension Components"`.
- They are regular React components, but in addition to being able to be rendered by React, they also contain various pieces of metadata that is used to wire together the entire app.
- Extension components are created using `create*Extension` methods [composability documentation].

As of this moment, there is no config based install procedure for plugins. Some code changes are required.


---

#### Plugin architecture

Architecturally, plugins can take 3 forms:

1. Standalone
2. Service backed
3. Third-party backed

---

##### Standalone plugins

**Standalone plugins**
- run entirely in the browser
- simply renders hard-coded information
- doesn't make any API requests to other services

for example
- [The Tech Radar plugin](https://demo.backstage.io/tech-radar),

![tech radar plugin ui](/assets/img/backstage-assets/architecture-overview/tech-radar-plugin.png)

- The architecture of the Tech Radar installed into a Backstage app is very simple.

![ui and tech radar plugin connected together](/assets/img/backstage-assets/architecture-overview/tech-radar-plugin-architecture.png)

---

##### Service backed plugins

**Service backed plugins**
- make API requests to a service which is within the purview of the organisation running Backstage.

for example:

- The Lighthouse plugin
  - makes requests to the [lighthouse-audit-service](https://github.com/spotify/lighthouse-audit-service).
  - The `lighthouse-audit-service` is a microservice which runs a copy of Google's [Lighthouse library](https://github.com/GoogleChrome/lighthouse/) and stores the results in a PostgreSQL database.

  - Its architecture looks like this:

  - ![lighthouse plugin backed to microservice and database](/assets/img/backstage-assets/architecture-overview/lighthouse-plugin-architecture.png)

- The software catalog
  - another example of a service backed plugin.
  - It retrieves a list of services, or "entities", from the Backstage Backend service and renders them in a table for the user.

---

##### Third-party backed plugins

Third-party backed plugins are similar to service backed plugins. The main
difference is that the service which backs the plugin is hosted outside of the
ecosystem of the company hosting Backstage.

The CircleCI plugin is an example of a third-party backed plugin. CircleCI is a
SaaS service which can be used without any knowledge of Backstage. It has an API
which a Backstage plugin consumes to display content.

Requests going to CircleCI from the user's browser are passed through a proxy
service that Backstage provides. Without this, the requests would be blocked by
Cross Origin Resource Sharing policies which prevent a browser page served at
[https://example.com](https://example.com) from serving resources hosted at
https://circleci.com.

![CircleCI plugin talking to proxy talking to SaaS Circle CI](/assets/img/backstage-assets/architecture-overview/circle-ci-plugin-architecture.png)

### Package Architecture

Backstage relies heavily on NPM packages, both for distribution of libraries,
and structuring of code within projects. While the way you structure your
Backstage project is up to you, there is a set of established patterns that we
encourage you to follow. These patterns can help set up a sound project
structure as well as provide familiarity between different Backstage projects.

The following diagram shows an overview of the package architecture of
Backstage. It takes the point of view of an individual plugin and all of the
packages that it may contain, indicated by the thicker border and italic text.
Surrounding the plugin are different package groups which are the different
possible interface points of the plugin. Note that not all library package lists
are complete as packages have been omitted for brevity.

![Package architecture](/assets/img/backstage-assets/architecture-overview/package-architecture.drawio.svg)

#### Overview

The arrows in the diagram above indicate a runtime dependency on the code of the
target package. This strict dependency graph only applies to runtime
`dependencies`, and there may be `devDependencies` that break the rules of this
table for the purpose of testing. While there are some arrows that show a
dependency on a collection of frontend, backend and isomorphic packages, those
still have to abide by important compatibility rules shown in the bottom left.

The `app` and `backend` packages are the entry points of a Backstage project.
The `app` package is the frontend application that brings together a collection
of frontend plugins and customizes them to fit an organization, while the
`backend` package is the backend service that powers the Backstage application.
Worth noting is that there can be more than one instance of each of these
packages within a project. Particularly the `backend` packages can benefit from
being split up into smaller deployment units that each serve their own purpose
with a smaller collection of plugins.

#### Plugin Packages

A typical plugin consists of up to five packages, two frontend ones, two
backend, and one isomorphic package. All packages within the plugin must share a
common prefix, typically of the form `@<scope>/plugin-<plugin-id>`, but
alternatives like `backstage-plugin-<plugin-id>` or
`@<scope>/backstage-plugin-<plugin-id>` are also valid. Along with this prefix,
each of the packages have their own unique suffix that denotes their role. In
addition to these five plugin packages it's also possible for a plugin to have
additional frontend and backend modules that can be installed to enable optional
features. For a full list of suffixes and their roles, see the
[Plugin Package Structure ADR](../architecture-decisions/adr011-plugin-package-structure.md).

The `-react`, `-common`, and `-node` plugin packages together form the external
library of a plugin. The plugin library enables other plugins to build on top of
and extend a plugin, and likewise allows the plugin to depend on and extend
other plugins. Because of this, it is preferable that plugin library packages
allow duplicate installations of themselves, as you may end up with a mix of
versions being installed as dependencies of various plugins. It is also
forbidden for plugins to directly import non-library packages from other
plugins, all communication between plugins must be handled through libraries and
the application itself.

#### Frontend Packages

The frontend packages are grouped into two main groups. The first one is
"Frontend App Core", which is the set of packages that are only used by the
`app` package itself. These packages help build up the core structure of the app
as well as provide a foundation for the plugin libraries to rely upon.

The second group is the rest of the shared packages, further divided into
"Frontend Plugin Core" and "Frontend Libraries". The core packages are
considered particularly stable and form the core of the frontend framework.
Their most important role is to form the boundary around each plugin and provide
a set of tools that helps you combine a collection of plugins into a running
application. The rest of the frontend packages are more traditional libraries
that serve as building blocks to create plugins.

#### Backend Packages

The backend library packages do not currently share a similar plugin
architecture as the frontend packages. They are instead simply a collection of
building blocks and patterns that help you build backend services. This is
however likely to change in the future.

#### Common Packages

The common packages are the packages effectively depended on by all other pages.
This is a much smaller set of packages but they are also very pervasive. Because
the common packages are isomorphic and must execute both in the frontend and
backend, they are never allowed to depend on any of the frontend or backend
packages.

The Backstage CLI is in a category of its own and is depended on by virtually
all other packages. It's not a library in itself though, and must always be a
development dependency only.

#### Deciding where you place your code

It can sometimes be difficult to decide where to place your plugin code. For example
should it go directly in the `-backend` plugin package or in the `-node` package?
As a general guideline you should try to keep the exposure of your code as low
as possible. If it doesn't need to be public API, it's best to avoid. If you don't
need it to be used by other plugins, then keep it directly in the plugin packages.

Below is a chart to help you decide where to place your code.

![Package decision](/assets/img/backstage-assets/architecture-overview/package-decision.drawio.svg)

### Databases

As we have seen, both the `lighthouse-audit-service` and `catalog-backend`
require a database to work with.

The Backstage backend and its built-in plugins are based on the
[Knex](https://knexjs.org/) library, and set up a separate logical database per
plugin. This gives great isolation and lets them perform migrations and evolve
separate from each other.

The Knex library supports a multitude of databases, but Backstage is at the time
of writing tested primarily against two of them: SQLite, which is mainly used as
an in-memory mock/test database, and PostgreSQL, which is the preferred
production database. Other databases such as the MySQL variants are reported to
work but
[aren't tested as fully](https://github.com/backstage/backstage/issues/2460)
yet.

### Cache

The Backstage backend and its built-in plugins are also able to leverage cache
stores as a means of improving performance or reliability. Similar to how
databases are supported, plugins receive logically separated cache connections,
which are powered by [Keyv](https://github.com/lukechilds/keyv) under the hood.

At this time of writing, Backstage can be configured to use one of three cache
stores: memory, which is mainly used for local testing, memcache or Redis,
which are cache stores better suited for production deployment. The right cache
store for your Backstage instance will depend on your own run-time constraints
and those required of the plugins you're running.

#### Use memory for cache

```yaml
backend:
  cache:
    store: memory
```

### Use memcache for cache

```yaml
backend:
  cache:
    store: memcache
    connection: user:pass@cache.example.com:11211
```

### Use Redis for cache

```yaml
backend:
  cache:
    store: redis
    connection: redis://user:pass@cache.example.com:6379
```

Contributions supporting other cache stores are welcome!

### Containerization

The example Backstage architecture shown above would Dockerize into three
separate Docker images.

1. The frontend container
2. The backend container
3. The Lighthouse audit service container

![Boxes around the architecture to indicate how it is containerised](/assets/img/backstage-assets/architecture-overview/containerised.png)

The backend container can be built by running the following command:

```bash
yarn run build
yarn run build-image
```

This will create a container called `example-backend`.

The lighthouse-audit-service container is already publicly available in Docker
Hub and can be downloaded and run with

```bash
docker run spotify/lighthouse-audit-service:latest
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


> If the Backstage App is already running (with `yarn start` or `yarn dev`) you should be able to see the default page for the new plugin directly by navigating to `https://localhost:3000/my-plugin`.

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

- This is where the plugin is created and where it creates and exports extensions that can be imported and used the app. See reference docs for [`createPlugin`] or introduction to
the new [Composability System].

##### Components

- The generated plugin includes two example components to showcase how we structure our plugins. There are usually one or multiple page components and next to them you can split up the UI in as many components as you feel like.

- We have the `ExampleComponent` to show an example Backstage page component. The `ExampleFetchComponent` showcases the common task of making an async request to a public API and plot the response data in a table using Material UI components.

- You may tweak these components, rename them and/or replace them completely.

##### Connecting the plugin to the Backstage app

There are two things needed for a Backstage app to start making use of a plugin.

1. Add plugin as dependency in `app/package.json`
2. Import and use one or more plugin extensions, for example in `app/src/App.tsx`.

Luckily both of these steps happen automatically when you create a plugin with the Backstage CLI.

##### Talking to the outside world

If the plugin needs to communicate with services outside the Backstage environment you will probably face challenges like `CORS policies and/or backend-side authorization`.

To smooth this process out you can use `proxy` - either the one you already have (like Nginx, HAProxy, etc.) or the `proxy-backend plugin` that we provide for the Backstage backend. [Read more](https://github.com/backstage/backstage/blob/master/plugins/proxy-backend/README.md)








.
