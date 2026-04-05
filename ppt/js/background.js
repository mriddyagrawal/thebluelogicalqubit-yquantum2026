// p5.js — floating quantum-particle background (light theme)
let particles = [];
const NUM_PARTICLES = 40;

function setup() {
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.style('z-index', '0');
  for (let i = 0; i < NUM_PARTICLES; i++) {
    particles.push({
      x: random(width),
      y: random(height),
      size: random(3, 8),
      speedX: random(-0.3, 0.3),
      speedY: random(-0.3, 0.3),
      alpha: random(30, 80),
      hue: random(200, 260)
    });
  }
}

function draw() {
  background(245, 247, 250, 40);
  stroke(180, 200, 230, 25);
  strokeWeight(0.8);
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      let d = dist(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
      if (d < 150) {
        let a = map(d, 0, 150, 40, 0);
        stroke(180, 200, 230, a);
        line(particles[i].x, particles[i].y, particles[j].x, particles[j].y);
      }
    }
  }
  noStroke();
  for (let p of particles) {
    fill(p.hue - 100, p.hue - 60, 255, p.alpha);
    ellipse(p.x, p.y, p.size);
    fill(p.hue - 100, p.hue - 60, 255, p.alpha * 0.3);
    ellipse(p.x, p.y, p.size * 2.5);
    p.x += p.speedX;
    p.y += p.speedY;
    if (p.x < -10) p.x = width + 10;
    if (p.x > width + 10) p.x = -10;
    if (p.y < -10) p.y = height + 10;
    if (p.y > height + 10) p.y = -10;
  }
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}
