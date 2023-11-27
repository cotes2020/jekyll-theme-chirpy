export function toc() {
  if (document.querySelector('#core-wrapper h2,#core-wrapper h3')) {
    // see: https://github.com/tscanlin/tocbot#usage
    tocbot.init({
      tocSelector: '#toc',
      contentSelector: '.post-content',
      ignoreSelector: '[data-toc-skip]',
      headingSelector: 'h2, h3',
      orderedList: false,
      scrollSmooth: false
    });
  }
}
