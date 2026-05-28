(function () {
  'use strict';

  /* ---------- theme toggle ---------- */
  var root = document.documentElement;
  var toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) { /* ignore */ }
    });
  }

  /* ---------- scrollspy: highlight nav link of the section in view ---------- */
  var navLinks = document.querySelectorAll('[data-nav]');
  if (!navLinks.length) return;

  var sections = [];
  navLinks.forEach(function (a) {
    var href = a.getAttribute('href');
    if (href && href.indexOf('#') === 0) {
      var el = document.querySelector(href);
      if (el) sections.push({ id: href, el: el, link: a });
    }
  });

  if (!sections.length) return;

  var topbar = document.getElementById('topbar');
  var topbarH = topbar ? topbar.offsetHeight : 0;

  function setActive(id) {
    sections.forEach(function (s) {
      if (s.id === id) s.link.classList.add('is-active');
      else s.link.classList.remove('is-active');
    });
  }

  function onScroll() {
    var pos = window.scrollY + topbarH + 32;
    var current = sections[0].id;
    for (var i = 0; i < sections.length; i++) {
      if (sections[i].el.offsetTop <= pos) {
        current = sections[i].id;
      } else {
        break;
      }
    }
    setActive(current);
  }

  var ticking = false;
  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(function () {
        onScroll();
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });

  window.addEventListener('resize', function () {
    topbarH = topbar ? topbar.offsetHeight : 0;
    onScroll();
  });

  onScroll();
})();
