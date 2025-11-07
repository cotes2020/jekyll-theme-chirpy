// Main application JavaScript
import './theme';

// Back to top button
function initBackToTop() {
  const backToTopBtn = document.getElementById('back-to-top');
  if (!backToTopBtn) return;

  const toggleVisibility = () => {
    if (window.scrollY > 300) {
      backToTopBtn.style.display = 'block';
    } else {
      backToTopBtn.style.display = 'none';
    }
  };

  window.addEventListener('scroll', toggleVisibility);
  toggleVisibility(); // Initial check

  backToTopBtn.addEventListener('click', () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    });
  });
}

// Sidebar mobile toggle
function initSidebar() {
  const sidebar = document.getElementById('sidebar');
  const mask = document.getElementById('mask');
  const sidebarTrigger = document.getElementById('sidebar-trigger');

  if (!sidebar || !mask || !sidebarTrigger) return;

  sidebarTrigger.addEventListener('click', () => {
    sidebar.classList.toggle('shown');
    mask.classList.toggle('d-none');
  });

  mask.addEventListener('click', () => {
    sidebar.classList.remove('shown');
    mask.classList.add('d-none');
  });
}

// Image lazy loading with fallback
function initImageLoading() {
  // Add loading error handlers
  const images = document.querySelectorAll('img[loading="lazy"]');
  images.forEach((img) => {
    img.addEventListener('error', function(this: HTMLImageElement) {
      this.style.display = 'none';
    });
  });
}

// Initialize on DOM ready
if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    initBackToTop();
    initSidebar();
    initImageLoading();
  });

  // Re-initialize on page navigation (for view transitions)
  document.addEventListener('astro:after-swap', () => {
    initBackToTop();
    initSidebar();
    initImageLoading();
  });
}

// Export for use in other modules
export { initBackToTop, initSidebar, initImageLoading };
