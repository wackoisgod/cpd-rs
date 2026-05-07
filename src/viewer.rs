use crate::decomp::Primitive;
use crate::mesh::Mesh;
use crate::prim;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_viewer(path: &Path, mesh: &Mesh, prims: &[Primitive]) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let mut data = String::new();
    data.push_str("{\n");

    // input mesh
    data.push_str("  \"input\": {\n    \"vertices\": [");
    for (i, v) in mesh.verts.iter().enumerate() {
        if i > 0 {
            data.push(',');
        }
        data.push_str(&format!("{},{},{}", v.x, v.y, v.z));
    }
    data.push_str("],\n    \"indices\": [");
    for (i, t) in mesh.tris.iter().enumerate() {
        if i > 0 {
            data.push(',');
        }
        data.push_str(&format!("{},{},{}", t[0], t[1], t[2]));
    }
    data.push_str("]\n  },\n");

    // primitives
    data.push_str("  \"primitives\": [\n");
    let mut first = true;
    for (pi, p) in prims.iter().enumerate() {
        if !p.alive {
            continue;
        }
        if !first {
            data.push_str(",\n");
        }
        first = false;
        let (verts, tris) = prim::tessellate(&p.prim);
        let kind = match p.prim.kind() {
            prim::PrimKind::Obb => "obb",
            prim::PrimKind::Sphere => "sphere",
            prim::PrimKind::Cylinder => "cylinder",
            prim::PrimKind::Capsule => "capsule",
            prim::PrimKind::Frustum => "frustum",
            prim::PrimKind::Prism => "prism",
        };
        data.push_str(&format!(
            "    {{\"id\":{},\"kind\":\"{}\",\"volume\":{},\"vertices\":[",
            pi, kind, p.volume
        ));
        for (i, v) in verts.iter().enumerate() {
            if i > 0 {
                data.push(',');
            }
            data.push_str(&format!("{},{},{}", v[0], v[1], v[2]));
        }
        data.push_str("],\"indices\":[");
        for (i, t) in tris.iter().enumerate() {
            if i > 0 {
                data.push(',');
            }
            data.push_str(&format!("{},{},{}", t[0], t[1], t[2]));
        }
        data.push_str("]}");
    }
    data.push_str("\n  ]\n}\n");

    w.write_all(HTML_PREFIX.as_bytes())?;
    w.write_all(data.as_bytes())?;
    w.write_all(HTML_SUFFIX.as_bytes())?;
    eprintln!("wrote viewer {}", path.display());
    Ok(())
}

const HTML_PREFIX: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CPD viewer</title>
<style>
  html, body { margin: 0; height: 100%; background: #1a1a1a; color: #e0e0e0; font: 13px -apple-system, system-ui, sans-serif; overflow: hidden; }
  canvas { display: block; }
  #ui {
    position: absolute; top: 8px; left: 8px;
    background: rgba(20,20,20,0.85); padding: 10px 12px;
    border: 1px solid #333; border-radius: 4px; min-width: 220px;
    user-select: none; z-index: 10;
  }
  #ui h2 { margin: 0 0 8px; font-size: 12px; font-weight: 600; opacity: 0.6; text-transform: uppercase; letter-spacing: 0.5px; }
  label { display: flex; align-items: center; padding: 3px 0; cursor: pointer; }
  label input { margin-right: 8px; }
  .swatch { display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.2); }
  hr { border: none; border-top: 1px solid #333; margin: 8px 0; }
  input[type=range] { width: 100%; }
  #stats { margin-top: 6px; opacity: 0.6; font-size: 11px; }
  #help { position: absolute; bottom: 8px; left: 8px; opacity: 0.4; font-size: 11px; z-index: 10; }
  .pane-label {
    position: absolute; top: 10px;
    background: rgba(20,20,20,0.8); padding: 4px 10px;
    border: 1px solid #333; border-radius: 3px;
    font-size: 11px; opacity: 0.7; pointer-events: none;
    text-transform: uppercase; letter-spacing: 0.5px;
    z-index: 5;
  }
  #pane-label-left  { display: none; left: 50%; transform: translateX(-100%) translateX(-12px); }
  #pane-label-right { display: none; left: 50%; transform: translateX(12px); }
  #divider {
    display: none;
    position: absolute; top: 0; bottom: 0; left: 50%;
    width: 1px; background: #333; pointer-events: none; z-index: 5;
  }
  body.split #divider, body.split #pane-label-left, body.split #pane-label-right { display: block; }
</style>
</head>
<body>
<div id="ui">
  <h2>convex primitive decomposition</h2>
  <label><input type="radio" name="layout" id="layout-overlay" value="overlay"> overlay</label>
  <label><input type="radio" name="layout" id="layout-split" value="split" checked> side-by-side</label>
  <hr>
  <label><input type="checkbox" id="toggle-input" checked> input mesh</label>
  <div style="margin: 0 0 4px 22px;">
    <label style="font-size: 12px; opacity: 0.7;">opacity
      <input type="range" id="input-opacity" min="0" max="100" value="100">
    </label>
  </div>
  <label><input type="checkbox" id="toggle-input-wire"> wireframe</label>
  <hr>
  <label><input type="checkbox" id="toggle-prims" checked> primitives</label>
  <div style="margin: 0 0 4px 22px;">
    <label style="font-size: 12px; opacity: 0.7;">opacity
      <input type="range" id="prim-opacity" min="0" max="100" value="90">
    </label>
  </div>
  <hr>
  <div id="kind-toggles"></div>
  <hr>
  <div id="stats"></div>
</div>
<div id="pane-label-left">input</div>
<div id="pane-label-right">primitives</div>
<div id="divider"></div>
<div id="help">drag = orbit · right-drag = pan · wheel = zoom · R = reset</div>

<script id="cpd-data" type="application/json">"#;

const HTML_SUFFIX: &str = r#"</script>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const data = JSON.parse(document.getElementById('cpd-data').textContent);

const KIND_COLORS = {
  obb: 0x4ade80,      // green
  sphere: 0x60a5fa,   // light blue
  cylinder: 0xfbbf24, // yellow
  capsule: 0xf87171,  // red
  frustum: 0xa78bfa,  // purple
  prism: 0x38bdf8,    // dark blue
};

const scene = new THREE.Scene();
scene.background = null;

let layoutMode = 'split'; // 'split' | 'overlay'
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 5000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setScissorTest(true);
renderer.setClearColor(0x1a1a1a, 1.0);
document.body.appendChild(renderer.domElement);
document.body.classList.add('split');

const controls = new OrbitControls(camera, renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(1, 2, 1);
scene.add(dir);

// Top-level groups per "pane" so we can toggle visibility per render pass in
// side-by-side mode. inputGroup goes on the left; primGroup on the right.
const inputGroup = new THREE.Group();
const primGroup = new THREE.Group();
scene.add(inputGroup);
scene.add(primGroup);

// build input mesh
const inputGeom = new THREE.BufferGeometry();
inputGeom.setAttribute('position', new THREE.Float32BufferAttribute(data.input.vertices, 3));
inputGeom.setIndex(data.input.indices);
inputGeom.computeVertexNormals();
const inputMat = new THREE.MeshStandardMaterial({
  color: 0xcccccc, transparent: true, opacity: 1.0,
  side: THREE.DoubleSide,
});
const inputMesh = new THREE.Mesh(inputGeom, inputMat);
inputGroup.add(inputMesh);

const wireMat = new THREE.LineBasicMaterial({ color: 0x666666, transparent: true, opacity: 0.5 });
const wireGeom = new THREE.WireframeGeometry(inputGeom);
const wire = new THREE.LineSegments(wireGeom, wireMat);
wire.visible = false;
inputGroup.add(wire);

// frame the view. URL param `?angle=iso|front|back|left|right|top|bottom`
// lets headless screenshotters capture the model from different sides.
inputGeom.computeBoundingBox();
const bb = inputGeom.boundingBox;
const center = bb.getCenter(new THREE.Vector3());
const size = bb.getSize(new THREE.Vector3());
const maxDim = Math.max(size.x, size.y, size.z);
const angle = new URLSearchParams(window.location.search).get('angle') || 'iso';
const angles = {
  iso:    new THREE.Vector3(1.5, 0.8, 1.5),
  front:  new THREE.Vector3(0.0, 0.2, 1.8),
  back:   new THREE.Vector3(0.0, 0.2, -1.8),
  right:  new THREE.Vector3(1.8, 0.2, 0.0),
  left:   new THREE.Vector3(-1.8, 0.2, 0.0),
  top:    new THREE.Vector3(0.001, 1.8, 0.001),
  bottom: new THREE.Vector3(0.001, -1.8, 0.001),
};
const offset = (angles[angle] || angles.iso).clone().multiplyScalar(maxDim);
camera.position.copy(center).add(offset);
camera.lookAt(center);
controls.target.copy(center);
const homePos = camera.position.clone();
const homeTarget = controls.target.clone();
controls.update();

// build primitive meshes, grouped by kind (under primGroup so we can hide
// the whole right pane in side-by-side mode)
const kindGroups = {};
for (const k of Object.keys(KIND_COLORS)) {
  kindGroups[k] = new THREE.Group();
  primGroup.add(kindGroups[k]);
}

const primKindCounts = {};
for (const p of data.primitives) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(p.vertices, 3));
  g.setIndex(p.indices);
  g.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({
    color: KIND_COLORS[p.kind] ?? 0xffffff,
    transparent: true, opacity: 0.8, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(g, mat);
  mesh.userData = { id: p.id, kind: p.kind, volume: p.volume };
  kindGroups[p.kind].add(mesh);
  primKindCounts[p.kind] = (primKindCounts[p.kind] ?? 0) + 1;
}

// build kind toggle UI
const kindUi = document.getElementById('kind-toggles');
for (const k of Object.keys(KIND_COLORS)) {
  const count = primKindCounts[k] ?? 0;
  const lbl = document.createElement('label');
  lbl.style.fontSize = '12px';
  lbl.innerHTML = `<input type="checkbox" data-kind="${k}" checked><span class="swatch" style="background:#${KIND_COLORS[k].toString(16).padStart(6,'0')}"></span>${k} (${count})`;
  if (count === 0) {
    lbl.style.opacity = '0.4';
    lbl.querySelector('input').disabled = true;
  }
  kindUi.appendChild(lbl);
  lbl.querySelector('input').addEventListener('change', e => {
    kindGroups[k].visible = e.target.checked;
  });
}

// other UI bindings — toggle at group-level so split-mode per-pane override
// can capture user intent cleanly.
document.getElementById('toggle-input').addEventListener('change', e => {
  inputGroup.visible = e.target.checked;
});
document.getElementById('toggle-input-wire').addEventListener('change', e => {
  wire.visible = e.target.checked;
});
document.getElementById('toggle-prims').addEventListener('change', e => {
  primGroup.visible = e.target.checked;
});
document.getElementById('input-opacity').addEventListener('input', e => {
  inputMat.opacity = e.target.value / 100;
});
document.getElementById('prim-opacity').addEventListener('input', e => {
  for (const k of Object.keys(KIND_COLORS)) {
    for (const m of kindGroups[k].children) m.material.opacity = e.target.value / 100;
  }
});

// stats line
const stats = document.getElementById('stats');
const triCount = data.input.indices.length / 3;
const primTotal = data.primitives.length;
stats.textContent = `${data.input.vertices.length / 3} verts · ${triCount} tris · ${primTotal} primitives`;

// keyboard: R resets the camera
window.addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') {
    camera.position.copy(homePos);
    controls.target.copy(homeTarget);
    controls.update();
  }
});

function applyLayout() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  if (layoutMode === 'split') {
    camera.aspect = (w * 0.5) / h;
  } else {
    camera.aspect = w / h;
  }
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  document.body.classList.toggle('split', layoutMode === 'split');
}
applyLayout();

window.addEventListener('resize', applyLayout);

document.getElementById('layout-overlay').addEventListener('change', e => {
  if (e.target.checked) { layoutMode = 'overlay'; applyLayout(); }
});
document.getElementById('layout-split').addEventListener('change', e => {
  if (e.target.checked) { layoutMode = 'split'; applyLayout(); }
});

// In side-by-side mode the user-facing toggles control "would this be visible
// in overlay mode" — but during render we override per-pane.
function renderSplit() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  const halfW = Math.floor(w * 0.5);

  // remember user-toggled visibilities and force per-pane.
  const inputWanted = inputGroup.visible;
  const primWanted = primGroup.visible;

  // left pane: input only
  inputGroup.visible = inputWanted;
  primGroup.visible = false;
  renderer.setViewport(0, 0, halfW, h);
  renderer.setScissor(0, 0, halfW, h);
  renderer.render(scene, camera);

  // right pane: primitives only
  inputGroup.visible = false;
  primGroup.visible = primWanted;
  renderer.setViewport(halfW, 0, w - halfW, h);
  renderer.setScissor(halfW, 0, w - halfW, h);
  renderer.render(scene, camera);

  inputGroup.visible = inputWanted;
  primGroup.visible = primWanted;
}

function renderOverlay() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setViewport(0, 0, w, h);
  renderer.setScissor(0, 0, w, h);
  renderer.render(scene, camera);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (layoutMode === 'split') renderSplit();
  else renderOverlay();
}
animate();
</script>
</body>
</html>
"#;
