/**
 * Interactive viewer for the reconstructed meshes: drag to rotate, wheel to zoom, right-drag to pan.
 *
 * One WebGL context is shared by all objects (browsers cap the number of live contexts at ~16, so a
 * canvas per object is not an option) and each `.glb` is fetched the first time its chip is picked.
 *
 * The vertex colours in the meshes are the diffuse albedo the material network predicts, written
 * sRGB-encoded by `trainer/mesh_export.py`; they are converted to linear here so that the lighting
 * maths and the sRGB output conversion of the renderer do not brighten them a second time.
 */
import * as THREE from 'three';
import { OrbitControls } from './vendor/three/OrbitControls.js';
import { GLTFLoader } from './vendor/three/GLTFLoader.js';

const holder = document.getElementById('mesh-viewer');
const status = document.getElementById('viewer-status');
const info = document.getElementById('viewer-info');
const meshes = JSON.parse(document.getElementById('mesh-data').textContent);
const byName = new Map(meshes.map((entry) => [entry.name, entry]));

const BACKGROUND = 0x0d0f16;
const UNLIT_GREY = 0xb9bfcc;

if (!meshes.length) {
  status.textContent = 'No meshes were exported.';
  throw new Error('no meshes');
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
holder.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BACKGROUND);

const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
scene.add(camera);

// Lights ride with the camera, so an object whose "up" is unknown is never left in the dark.
const key = new THREE.DirectionalLight(0xffffff, 2.0);
key.position.set(0.6, 0.9, 1.0);
camera.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.6);
fill.position.set(-1.0, -0.3, 0.6);
camera.add(fill);
scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x2a2f3d, 0.9));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotateSpeed = 1.1;
controls.minDistance = 0.6;
controls.maxDistance = 20;

const loader = new GLTFLoader();
const cache = new Map();
let current = null;
let showAlbedo = true;

/**
 * Rewrite an sRGB-encoded COLOR_0 attribute as a linear float attribute.
 *
 * @param {THREE.BufferGeometry} geometry geometry whose 'color' attribute is replaced in place.
 */
function colorsToLinear(geometry) {
  const attribute = geometry.getAttribute('color');
  if (!attribute) return;
  const linear = new Float32Array(attribute.count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < attribute.count; i++) {
    color.setRGB(attribute.getX(i), attribute.getY(i), attribute.getZ(i), THREE.SRGBColorSpace);
    linear[3 * i] = color.r;
    linear[3 * i + 1] = color.g;
    linear[3 * i + 2] = color.b;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(linear, 3));
}

/**
 * Place the camera so the whole object is in frame, seen from a three-quarter view.
 *
 * @param {THREE.Mesh} mesh the mesh to frame.
 */
function frame(mesh) {
  mesh.geometry.computeBoundingSphere();
  const radius = mesh.geometry.boundingSphere.radius || 1;
  const distance = 1.25 * radius / Math.sin(0.5 * THREE.MathUtils.degToRad(camera.fov));
  const direction = new THREE.Vector3(0.62, 0.42, 1.0).normalize();
  camera.position.copy(direction.multiplyScalar(distance));
  camera.near = Math.max(0.01, distance - 3 * radius);
  camera.far = distance + 5 * radius;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

/**
 * Describe the object under the canvas: what was decimated, and whether "up" is meaningful.
 *
 * @param {object} entry the entry of `meshes` being shown.
 */
function describe(entry) {
  const kib = (entry.bytes / 1024).toFixed(0);
  const mib = (entry.bytes_source / 1024 / 1024).toFixed(0);
  const upright = entry.upright
    ? 'gravity-aligned from the Blender ground truth'
    : 'no ground truth for "up", shown in the network’s canonical frame';
  const metrics = entry.metrics_3d
    ? ` &middot; CD-L1 ${(100 * entry.metrics_3d.chamfer_l1_relative).toFixed(3)}% of the GT diagonal`
      + ` &middot; F@1% ${entry.metrics_3d['f_score@0.01'].toFixed(3)}`
    : '';
  info.innerHTML = `<b>${entry.name}</b> &middot; ${entry.faces_source.toLocaleString()} faces`
    + ` (${mib} MB ply) decimated to ${entry.faces.toLocaleString()} (${kib} KiB glb)`
    + ` &middot; ${upright}${metrics}`;
}

/**
 * Show one object, fetching and preparing its glb the first time it is asked for.
 *
 * @param {string} name object name, a key of `byName`.
 */
function show(name) {
  const entry = byName.get(name);
  if (!entry) return;
  for (const chip of document.querySelectorAll('#mesh-picker a')) {
    chip.classList.toggle('active', chip.dataset.object === name);
  }
  describe(entry);

  if (cache.has(name)) {
    swapIn(cache.get(name));
    return;
  }
  status.textContent = `loading ${name}…`;
  loader.load(entry.glb, (gltf) => {
    let mesh = null;
    gltf.scene.traverse((node) => {
      if (!mesh && node.isMesh) mesh = node;
    });
    if (!mesh) {
      status.textContent = `${name}: the glb has no mesh`;
      return;
    }
    colorsToLinear(mesh.geometry);
    mesh.material = new THREE.MeshStandardMaterial({
      vertexColors: showAlbedo, color: showAlbedo ? 0xffffff : UNLIT_GREY,
      roughness: 0.72, metalness: 0.0, side: THREE.DoubleSide,
    });
    cache.set(name, mesh);
    swapIn(mesh);
  }, undefined, (error) => {
    status.textContent = `${name}: could not load ${entry.glb} (${error.message || error})`;
  });
}

/**
 * Put a prepared mesh on screen in place of the one before it.
 *
 * @param {THREE.Mesh} mesh the mesh to show.
 */
function swapIn(mesh) {
  if (current) scene.remove(current);
  current = mesh;
  applyMaterialMode();
  scene.add(mesh);
  frame(mesh);
  status.textContent = '';
}

/** Apply the albedo/plain-grey choice to the mesh on screen. */
function applyMaterialMode() {
  if (!current) return;
  current.material.vertexColors = showAlbedo;
  current.material.color.set(showAlbedo ? 0xffffff : UNLIT_GREY);
  current.material.needsUpdate = true;
}

/** Match the drawing buffer to the size the layout gave the canvas. */
function resize() {
  const width = holder.clientWidth;
  const height = holder.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(holder);
resize();

document.getElementById('mesh-picker').addEventListener('click', (event) => {
  const chip = event.target.closest('a[data-object]');
  if (!chip) return;
  event.preventDefault();
  show(chip.dataset.object);
});

for (const link of document.querySelectorAll('a.view-mesh[data-object]')) {
  link.addEventListener('click', () => show(link.dataset.object));
}

document.getElementById('viewer-spin').addEventListener('change', (event) => {
  controls.autoRotate = event.target.checked;
});
document.getElementById('viewer-albedo').addEventListener('change', (event) => {
  showAlbedo = event.target.checked;
  applyMaterialMode();
});
document.getElementById('viewer-reset').addEventListener('click', () => {
  if (current) frame(current);
});

controls.autoRotate = document.getElementById('viewer-spin').checked;
show(meshes[0].name);

renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
