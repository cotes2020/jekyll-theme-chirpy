/**
 * Tab 'Categories' expand/close effect.
 */

import 'bootstrap/js/src/collapse.js';

const childPrefix = 'l_';
const parentPrefix = 'h_';
const children = document.getElementsByClassName('collapse');

export function categoryCollapse(): void {
  [...children].forEach((elem) => {
    if (!(elem instanceof HTMLElement)) return;
    const id = parentPrefix + elem.id.substring(childPrefix.length);
    const parent = document.getElementById(id);

    // collapse sub-categories
    elem.addEventListener('hide.bs.collapse', () => {
      if (parent) {
        parent.querySelector<HTMLElement>('.far.fa-folder-open')?.setAttribute(
          'class',
          'far fa-folder fa-fw'
        );
        parent
          .querySelector<HTMLElement>('.fas.fa-angle-down')
          ?.classList.add('rotate');
        parent.classList.remove('hide-border-bottom');
      }
    });

    // expand sub-categories
    elem.addEventListener('show.bs.collapse', () => {
      if (parent) {
        parent.querySelector<HTMLElement>('.far.fa-folder')?.setAttribute(
          'class',
          'far fa-folder-open fa-fw'
        );
        parent
          .querySelector<HTMLElement>('.fas.fa-angle-down')
          ?.classList.remove('rotate');
        parent.classList.add('hide-border-bottom');
      }
    });
  });
}
