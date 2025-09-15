// Iframe visibility control using Intersection Observer
document.addEventListener('DOMContentLoaded', function () {
  const iframe = document.getElementById('portfolio-banner');
  const container = document.getElementById('portfolio-container');
  
  if (!iframe || !container) return;
  
  let iframeLoaded = false;

  // Safe postMessage function with error handling
  function safePostMessage(message) {
    if (!iframe || !iframeLoaded) return;
    
    try {
      // Check if iframe and contentWindow exist
      if (iframe.contentWindow && typeof iframe.contentWindow.postMessage === 'function') {
        iframe.contentWindow.postMessage(message, 'https://ounols.github.io');
      }
    } catch (error) {
      console.warn('PostMessage failed:', error);
    }
  }

  // Safe visibility control with !important
  function setIframeVisibility(visible) {
    if (!iframe) return;
    
    if (visible) {
      iframe.style.setProperty('display', 'block', 'important');
      iframe.style.setProperty('visibility', 'visible', 'important');
      iframe.style.setProperty('opacity', '1', 'important');
    } else {
      iframe.style.setProperty('display', 'none', 'important');
      iframe.style.setProperty('visibility', 'hidden', 'important');
      iframe.style.setProperty('opacity', '0', 'important');
    }
  }

  // Wait for iframe to load completely
  iframe.addEventListener('load', function() {
    iframeLoaded = true;
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          // Header is visible - activate iframe and resume Emscripten loop
          setIframeVisibility(true);
          safePostMessage({action: 'resume'});
        } else {
          // Header is not visible - pause Emscripten loop and hide iframe
          safePostMessage({action: 'pause'});
          setIframeVisibility(false);
        }
      });
    },
    {
      threshold: 0.01 // Trigger when 1% of the container is visible
    }
  );

  observer.observe(container);
});