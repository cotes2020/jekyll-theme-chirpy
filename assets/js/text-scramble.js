class TextScramble {
  constructor(el) {
    this.el = el;
    this.chars = '!<>-_\\/[]{}—=+*^?#________';
    this.revealSpeed = 0.3;
    this.changeFrequency = 0.2;
    this.highlightColor = '#00ff00';
    this.glowIntensity = 10;
    this.activeGlowIntensity = 12;
    this.update = this.update.bind(this);
  }

  setText(newText) {
    const oldText = this.el.innerText;
    const length = Math.max(oldText.length, newText.length);
    const promise = new Promise(resolve => this.resolve = resolve);
    this.queue = [];

    for (let i = 0; i < length; i++) {
      const from = oldText[i] || '';
      const to = newText[i] || '';
      const start = Math.floor(Math.random() * (40 / this.revealSpeed));
      const end = start + Math.floor(Math.random() * (40 / this.revealSpeed));
      this.queue.push({ from, to, start, end });
    }

    cancelAnimationFrame(this.frameRequest);
    this.startTime = performance.now();
    this.frameRequest = requestAnimationFrame(this.update);

    return promise;
  }

  update(time) {
    const elapsed = time - this.startTime;
    let output = '';
    let complete = 0;

    for (let i = 0, n = this.queue.length; i < n; i++) {
      let { from, to, start, end, char } = this.queue[i];
      const t = elapsed / 16;

      if (t >= end) {
        complete++;
        output += to;
      } else if (t >= start) {
        if (!char || Math.random() < this.changeFrequency) {
          char = this.randomChar();
          this.queue[i].char = char;
        }
        output += `<span style="color: ${this.highlightColor}; text-shadow: 0 0 ${this.activeGlowIntensity}px currentColor;">${char}</span>`;
      } else {
        output += from;
      }
    }

    this.el.innerHTML = output;

    if (complete === this.queue.length) {
      this.resolve();
    } else {
      this.frameRequest = requestAnimationFrame(this.update);
    }
  }

  randomChar() {
    return this.chars[Math.floor(Math.random() * this.chars.length)];
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".card-title").forEach(el => {
    if (!el.dataset.originalText) {
      el.dataset.originalText = el.textContent;
    }
    const fx = new TextScramble(el);
    fx.setText(el.dataset.originalText);
  });
});
