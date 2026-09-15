---
name: threejs-3d
command: /three
description: Three.js 3D scenes — PBR materials, GLTF loading, cameras, animation mixer, postprocessing. Use for 3D visualizations, WebGL scenes, or WebXR prototypes.
tags: [threejs, 3d, webgl, gltf, animation]
---

# threejs-3d

Modern Three.js (r160+). ES modules from a CDN when the project doesn't have a bundler.

## When to use

- User asks for 3D, a 3D viewer, a WebGL scene, "a rotating cube", "load this GLTF".
- User has a game idea that's more than 2D but doesn't need a full engine.

## Boilerplate

```html
<canvas id="c" style="width:100%; height:100vh; display:block;"></canvas>
<script type="importmap">{ "imports": {
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}</script>
<script type="module">
  import * as THREE from 'three';
  import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

  const canvas = document.getElementById('c');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e0f13);

  const camera = new THREE.PerspectiveCamera(60, canvas.clientWidth / canvas.clientHeight, 0.1, 100);
  camera.position.set(3, 2, 5);
  camera.lookAt(0, 0, 0);

  const light = new THREE.DirectionalLight(0xffffff, 3);
  light.position.set(5, 10, 7);
  scene.add(light, new THREE.AmbientLight(0xffffff, 0.4));

  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(),
    new THREE.MeshStandardMaterial({ color: 0x66d9ef, roughness: 0.4 })
  );
  scene.add(cube);

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  addEventListener('resize', resize);

  function frame(t) {
    cube.rotation.y = t * 0.001;
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
</script>
```

## Patterns

- **Load a GLTF**: `new GLTFLoader().load('model.glb', gltf => scene.add(gltf.scene))`. Animations are in `gltf.animations`.
- **AnimationMixer**: `const mixer = new THREE.AnimationMixer(gltf.scene); mixer.clipAction(gltf.animations[0]).play();` — call `mixer.update(dt)` each frame.
- **PBR materials**: `MeshStandardMaterial` for most, `MeshPhysicalMaterial` when you need clearcoat/transmission/iridescence.
- **Orbit controls**: `import { OrbitControls } from 'three/addons/controls/OrbitControls.js';` then `new OrbitControls(camera, canvas)`.
- **Postprocessing**: `EffectComposer` + `RenderPass` + `UnrealBloomPass` for bloom. Only when the user asks — it costs frames.
- **Instancing**: `InstancedMesh` for grass, crowds, particle-style geometry. Set matrices via `mesh.setMatrixAt(i, m)` then `mesh.instanceMatrix.needsUpdate = true`.

## Avoid

- Adding lights per frame — set them up once in the setup block.
- Creating geometry per frame — reuse and mutate.
- `renderer.setPixelRatio(devicePixelRatio)` on integrated GPUs — quality bump costs 4× fill. Skip it for the reference box.
