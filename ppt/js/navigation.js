// Slide navigation + interactivity engine
let currentSlide = 1;
let totalSlides = 0;

function initNavigation() {
  totalSlides = document.querySelectorAll('.slide').length;
  updateCounter();
}

function showSlide(n) {
  if (n < 1 || n > totalSlides) return;
  document.querySelector('.slide.active').classList.remove('active');
  document.getElementById('slide-' + n).classList.add('active');
  currentSlide = n;
  updateCounter();
}

function updateCounter() {
  document.getElementById('slide-counter').textContent = currentSlide + ' / ' + totalSlides;
}

document.addEventListener('keydown', function (e) {
  if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); showSlide(currentSlide + 1); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); showSlide(currentSlide - 1); }
});

// click navigation — ignore interactive elements
document.addEventListener('click', function (e) {
  if (e.target.closest('.interactive, .bitstring, canvas, img, button, .stat-card, table')) return;
  if (e.clientX > window.innerWidth / 2) showSlide(currentSlide + 1);
  else showSlide(currentSlide - 1);
});
