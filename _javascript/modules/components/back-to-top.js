/**
 * Reference: https://bootsnipp.com/snippets/featured/link-to-top-page
 */

export function back2top() {
  const btn = document.getElementById('back-to-top');

  window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
      btn.classList.add('show');

      const circumference = 2 * 3.14 * 20;
      const circle = document.querySelector('#progress-circle circle');
      const scrollTop = window.scrollY;
      const docHeight = document.documentElement.scrollHeight - window.innerHeight;
      const scrollFraction = scrollTop / docHeight;
      const drawLength = circumference * scrollFraction;

      circle.style.strokeDashoffset = circumference - drawLength;
    } else {
      btn.classList.remove('show');
    }
  });

  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0 });
  });
}
