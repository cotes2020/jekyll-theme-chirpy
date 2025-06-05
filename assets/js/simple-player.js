// ... your existing selectors
const audio = document.getElementById('audio');
const playBtn = document.getElementById('play-btn');
const loopBtn = document.getElementById('loop-btn');
const seek = document.getElementById('seek');
const time = document.getElementById('time');
const volume = document.getElementById('volume');
const songTitle = document.getElementById('song-title');



// === Load volume from localStorage if available ===
const savedVolume = localStorage.getItem('player-volume');
if (savedVolume !== null) {
  volume.value = savedVolume;
  audio.volume = savedVolume / 100;
}

// === Restore position and song ===
let song;
fetch('/assets/music/song.json')
  .then(res => res.json())
  .then(data => {
    const today = new Date().getDate();
    song = data[today % data.length];
    audio.src = song.audio;
    songTitle.textContent = `${song.title} — ${song.author}`;

    const savedTime = localStorage.getItem('player-time');
    const savedSrc = localStorage.getItem('player-src');

    if (savedSrc === song.audio && savedTime) {
      audio.currentTime = parseFloat(savedTime);
    }

    //  Only try to autoplay if the browser allows it
    const shouldResume = localStorage.getItem('player-was-playing') === 'true';

    if (shouldResume) {
      audio.play().then(() => {
        playBtn.textContent = '⏸️';
      }).catch(() => {
        // Auto-play blocked or failed — reset button
        playBtn.textContent = '▶️';
        localStorage.setItem('player-was-playing', 'false');
      });
    } else {
      playBtn.textContent = '▶️';
    }
  });


playBtn.addEventListener('click', () => {
  if (audio.paused) {
    audio.play();
    playBtn.textContent = '⏸️';
    localStorage.setItem('player-was-playing', 'true');
  } else {
    audio.pause();
    playBtn.textContent = '▶️';
    localStorage.setItem('player-was-playing', 'false');
  }
});
// Load loop state on startup
const wasLooping = localStorage.getItem('player-loop') === 'true';
audio.loop = wasLooping;
loopBtn.textContent = audio.loop ? '🔂' : '🔁';

// Save loop toggle
loopBtn.addEventListener('click', () => {
  audio.loop = !audio.loop;
  loopBtn.textContent = audio.loop ? '🔂' : '🔁';
  localStorage.setItem('player-loop', audio.loop);
});



audio.addEventListener('timeupdate', () => {
  seek.value = audio.currentTime;
  time.textContent = `${format(audio.currentTime)} / ${format(audio.duration)}`;
  localStorage.setItem('player-time', audio.currentTime);
});

audio.addEventListener('loadedmetadata', () => {
  seek.max = audio.duration;
  localStorage.setItem('player-src', audio.src);
});

seek.addEventListener('input', () => {
  audio.currentTime = seek.value;
});

volume.addEventListener('input', () => {
  audio.volume = volume.value / 100;
  localStorage.setItem('player-volume', volume.value);
});

function format(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}
