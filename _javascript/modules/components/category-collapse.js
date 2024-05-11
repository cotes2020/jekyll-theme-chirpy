/**
 * Tab 'Categories' expand/close effect.
 */

import 'bootstrap/js/src/collapse.js';

const childPrefix = 'l_';
const parentPrefix = 'h_';
const children = document.getElementsByClassName('collapse');

export function categoryCollapse() {
  [...children].forEach((elem) => {
    const id = parentPrefix + elem.id.substring(childPrefix.length);
    const parent = document.getElementById(id);

    // collapse sub-categories
    elem.addEventListener('hide.bs.collapse', () => {
      if (parent) {
        parent.querySelector('.far.fa-folder-open').className =
          'far fa-folder fa-fw';
        parent.querySelector('.fas.fa-angle-down').classList.add('rotate');
        parent.classList.remove('hide-border-bottom');
      }
    });

    // expand sub-categories
    elem.addEventListener('show.bs.collapse', () => {
      if (parent) {
        parent.querySelector('.far.fa-folder').className =
          'far fa-folder-open fa-fw';
        parent.querySelector('.fas.fa-angle-down').classList.remove('rotate');
        parent.classList.add('hide-border-bottom');
      }
    });
  });
}
