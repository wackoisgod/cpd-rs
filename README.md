# cpd-rs

Rust implementation of **Convex Primitive Decomposition for Collision Detection**
(Knodt & Gao, Eurographics 2026,
[arXiv:2602.07369](https://arxiv.org/abs/2602.07369)) plus a few research
extensions and a self-contained eval/visualisation workflow.

The algorithm fits a small set of parametric convex primitives — oriented
bounding boxes, spheres, capped cylinders, capsules, frustums, and isosceles
trapezoidal prisms — to an input mesh by greedy bottom-up merging driven by
quadric error metrics. It targets game-collider use cases where a tight,
performant, and artist-modifiable decomposition matters more than the raw
hull count of approximate convex decomposition.

## Status

Working end-to-end on real `.glb` meshes. Output metrics on tested models
match the paper's reported one-way Hausdorff/Chamfer ranges (Sec. 4.4).

| mesh | tris | N | volume / AABB | one-way Hausdorff (% of diag) | Chamfer (% of diag) | merge time |
|---|---|---|---|---|---|---|
| `blink-visual.glb` | 35,680 | 256 | 17.6% | 4.03% | 0.44% | 714 ms |
| `ram.glb` | 2,626 | 256 | 12.7% | 4.41% | 0.37% | 44 ms |
| `ram.glb` | 2,626 | 512 | 4.0% | 4.41% | 0.21% | 38 ms |
| `ram-visual.glb` | 69,001 | 512 | 39.4% | 6.15% | 0.59% | 1.4 s |

Paper's reported mean across 60+ models: 4.45% Hausdorff, 0.70% Chamfer.

## Build

```
cargo build --release
```

Tested on macOS aarch64 with Rust 1.95.

## CLI

```
cpd <mesh.glb> <target_n> [out.obj]
    [--viewer <viewer.html>]
    [--metrics] [--metrics-json <path>]
    [--volume-threshold <fraction-of-AABB-volume>]
    [--obb-only]
    [--no-cull]                    disable redundant-primitive cull
    [--no-refine]                  disable post-merge orientation refit
    [--empty-space]                hard-reject merges bridging open space
        [--empty-space-fraction <0..1>]   default 0.25
        [--empty-space-distance <frac-of-diag>]   default 0.01
    [--no-sphere | --no-cylinder | --no-capsule | --no-frustum | --no-prism]
```

Examples:

```
# Tight collision mesh with metrics
cargo run --release -- input.glb 256 colliders.obj --metrics

# Side-by-side viewer
cargo run --release -- input.glb 256 colliders.obj \
    --viewer viewer.html --metrics --metrics-json metrics.json

# Hole-preserving (stairwells, mazes, vents)
cargo run --release -- input.glb 256 colliders.obj --empty-space
```

The viewer is a single self-contained HTML file using three.js from a CDN.
Drag = orbit, right-drag = pan, wheel = zoom, R = reset. Side-by-side and
overlay layouts are toggleable; per-kind primitive visibility is toggleable.

## Eval workflow

A fast iterate-and-validate loop:

```
# 1. Build + run + collect metrics + write viewer
./target/release/cpd <mesh.glb> <N> out.obj \
    --viewer viewer.html --metrics --metrics-json metrics.json

# 2. Headless screenshot of the viewer (uses macOS Chrome)
./scripts/screenshot.sh viewer.html out.png 1920 1080
```

`metrics.json` carries per-run quantitative numbers (Hausdorff, Chamfer,
volume ratio, by-kind counts, worst-primitive locator). `out.png` is a
side-by-side render. The PNG is small enough to read in any tool that
handles images (including agentic tooling).

## What's implemented (vs. the paper)

| Section | Feature | Status |
|---|---|---|
| §3.1 | Per-face quadric `Q = area·(nnᵀ + ε·ttᵀ)` | ✓ |
| §3.4 | Quad-aware tangent for coplanar regions | ✓ |
| §3.2 | OBB / Sphere / Cylinder / Capsule / Frustum / Prism (Algs 2 & 3) | ✓ |
| §3.3 | Excess-volume cost, per-shape weights, volume threshold | ✓ |
| §3.4 | Vertex welding | ✓ |
| §3.4 | Redundant-primitive cull | ✓ (constrained — see below) |
| §3.4 | DSU + cyclic linked list face bookkeeping | ✓ |
| §3.4 | Pairwise component fallback | ✓ |
| §4.4 | One-way Hausdorff / Chamfer evaluation metrics | ✓ |

### Extensions beyond the paper

- **Multi-orientation post-merge refit.** On every accepted merge, refit the
  merged primitive against three orientation candidates and pick the lowest
  weighted volume:
    1. Q's eigenbasis (paper default).
    2. Vertex PCA — captures elongated geometric extent.
    3. Tangent-plane PCA — fixes the normal from Q, runs 2D PCA in the
       perpendicular plane. Auto-handles coplanar degeneracy.
    4. Sharp-edge PCA — covariance of unit edge directions on edges where
       adjacent face normals deviate by > 30°. Targets feature-aligned
       primitives (building corners, hood lines, stair treads).

  Cost: ~10% more merge time, ~1.5–11% tighter total volume on tested meshes.

- **Constrained redundant cull.** The paper drops primitive A if every
  vertex of A is enclosed by some other primitive B. This deletes
  legitimate per-component primitives once the all-pairs phase has wrapped
  unrelated components together. We additionally require A and B to share
  at least one mesh vertex — i.e., they were topologically related at some
  point during the merge — which preserves tight per-component fits.

- **Empty-space preservation (toggleable).** `--empty-space` adds a hard
  reject: sample 27 stratified points inside the candidate primitive's
  AABB, reject the merge if more than `--empty-space-fraction` of the
  in-primitive samples sit further than `--empty-space-distance` *outside*
  the input mesh (signed-distance via BVH). Disables the all-pairs
  fallback so rejections don't trigger combinatorial candidate explosion.
  Effective on hole-dominant geometries (stairwells, mazes); for curved
  surface meshes it will reject some legitimate merges — use with care.

- **Degenerate-axis guard for cylinder/capsule fits.** When a point cloud
  is near-coplanar, fitting a cylinder along the normal collapses to a
  thin disk: `h` clamps to `MIN_HALF_EXTENT` and `r` is full in-plane
  radius. The volume formula gives a misleadingly small number, so the
  disk wins selection despite extending far beyond the cloud. We skip
  axes whose axial extent is < 5% of the largest axial extent.

- **Self-evaluation loop.** `--metrics` computes paper-§4.4 distances
  against the input mesh via a BVH-accelerated nearest-face query.
  `scripts/screenshot.sh` headlessly renders the viewer to PNG so an
  agent can visually validate without a GUI.

### Performance

- Sphere skipped when OBB also enabled (provably loses on non-degenerate
  inputs).
- Fancy primitive fits skipped on tiny vertex counts (`< 8`) — early
  merges of singleton triangles are too small for cylinder/capsule/
  frustum/prism to differ from OBB.
- Push-time fit cached in priority queue entries so valid pops don't redo
  `fit_best`.
- Initial 50k-pair PQ build and the all-pairs N² fallback parallelised
  with rayon. Post-merge candidate generation is sequential (per-pop
  candidate count is small enough that par_iter overhead dominates).

## Layout

```
src/
  main.rs        — CLI, argv parsing, top-level orchestration
  mesh.rs        — .glb load, vertex welding, adjacency, sharp-edge detect
  prim.rs        — primitive enum, fit/volume/contains/tessellate/world_aabb
  decomp.rs      — quadric, eigendecomp, PCA variants, merge loop, cull
  bvh.rs         — AABB BVH on input mesh, nearest-face query
  dsu.rs         — disjoint set union (path compression, union-by-rank)
  metrics.rs     — area-weighted surface sampling, Hausdorff/Chamfer
  io_obj.rs      — write primitives as colliders.obj
  viewer.rs      — write self-contained three.js viewer.html
scripts/
  screenshot.sh  — headless Chrome render of viewer.html → png
```

## License

Personal research repo. No license claimed; not open-sourced.

## Citation

```
@article{knodt2026cpd,
  title  = {Convex Primitive Decomposition for Collision Detection},
  author = {Knodt, Julian and Gao, Xifeng},
  journal= {Computer Graphics Forum (Eurographics)},
  volume = {45},
  number = {2},
  year   = {2026}
}
```
