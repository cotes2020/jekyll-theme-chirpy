---
published: true
date: 2024-02-23
title: How to install Angular Material in Angular 17
---
Modify `angular.json` from root folder:

    {
      "projects": [
        "[application-name]": {
          "architect": {
            "build": {
              // "builder": "ngx-build-plus:browser",
              "builder": "@angular-devkit/build-angular:browser",
              "options": {
                // "main": "path/to/[application-name]/src/main.ts",
                "main": "path/to/[application-name]/src/bootstrap.ts",
              }
            }
          }
        }
      ]
    }
    

Let’s install Angular Material:

    ng add @angular/material
    

Optional steps:

*   Move all Angular Material auto generated styles, providers to root module/component of the application.
    
*   In the root component which imports Angular Material styles, add this config for sharing css via Module Federation:

```
@Component({
  //...
  encapsulation: ViewEncapsulation.None,
  //...
})
```

Finally, revert above `angular.json` changed configs.