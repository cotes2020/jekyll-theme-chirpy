---
published: true
date: 2023-06-02
title: How to import specific component of Bootstrap with Vite
---
Install Bootstrap:

    npm i --save bootstrap @popperjs/core
    npm i --save-dev sass
    

Create _bootstrap.scss_ with the below content:

    // Required
    @import "~bootstrap/scss/functions";
    @import "~bootstrap/scss/variables";
    @import "~bootstrap/scss/variables-dark";
    @import "~bootstrap/scss/maps";
    @import "~bootstrap/scss/mixins";
    @import "~bootstrap/scss/root";
    
    // Components
    @import "~bootstrap/scss/_buttons.scss";
    @import "~bootstrap/scss/_tables.scss";
    @import "~bootstrap/scss/_alert.scss";
    @import "~bootstrap/scss/forms";
    // ...
    

The list of components could find at `node_modules/bootstrap/scss`

Then import `bootstrap.scss` to `main.js`:

    import 'path/to/bootstrap.scss';