import * as THREE from 'three';
import { SceneManager } from './sceneManager.js';
import { initUI } from './utils/ui.js';

const canvas = document.getElementById('webgl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
const DPR = Math.min(window.devicePixelRatio || 1, 1.5);
renderer.setPixelRatio(DPR);
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

let sceneManager;
const clock = new THREE.Clock();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1e9);
camera.position.set(0, 0, 6);
const scene = new THREE.Scene();
scene.background = new THREE.Color('#05060A');
const pmrem = new THREE.PMREMGenerator(renderer);

let running = false;
const loop = () => {
  if (!running) return;
  const dt = clock.getDelta();
  sceneManager.update(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
};

const { ui, loader } = initUI({
  onPrev: () => sceneManager.prev(),
  onNext: () => sceneManager.next(),
  onSkip: () => sceneManager.skip(),
  onDot: (i) => sceneManager.goTo(i),
  onStart: async () => {
    loader.hide();
    running = true;
    loop();
    await sceneManager.start();
    ui.setCaption(sceneManager.caption());
  }
});

sceneManager = new SceneManager({ renderer, scene, camera, pmrem, ui });

window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  sceneManager.resize(w, h);
});
