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
    [--quality <beta>]             experimental Hausdorff-aware refit
    [--shell]                      experimental shell-aware orientation
    [--proximity]                  spatial-proximity replaces all-pairs fallback
    [--weighted-cost]              PQ ordering uses weighted volume (organic-friendly)
    [--rebalance]                  Lloyd-style face migration after greedy (experimental)
    [--reject-pancakes]            penalise 1mm × N-metre slab merges (architecture)
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

# Multi-angle (iso/front/back/left/right/top/bottom, montaged 4×2;
# requires ImageMagick)
./scripts/screenshot.sh viewer.html out.png 1280 720 --multi
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

- **Hausdorff-aware refit (experimental, `--quality <beta>`).** When `beta
  > 0` the post-merge refit ranks candidates by
  `weighted_volume * (1 + beta · h/diag)` instead of pure
  `weighted_volume`, where `h` is sampled from the candidate primitive's
  surface to the input mesh via the BVH. Sphere becomes a real candidate
  in this mode (it's normally skipped because it always loses on raw
  volume). **Mixed results across meshes:** helps notably on the rock-kit
  at N=128 (4.26% → 3.74% Hausdorff) but can *worsen* at N=256 because
  greedy refit decisions don't always reduce the global max-Hausdorff.
  Try `--quality 1` to `--quality 5`; default `0` (off).

- **Weighted merge cost (`--weighted-cost`).** Default cost for the
  priority queue is `ΔV` (excess unweighted volume). With this flag,
  cost becomes `ΔweightedV` — every merge candidate is judged using
  the per-shape weights (OBB/sphere/capsule = 1.0, cylinder = 1.05,
  prism = 1.4, frustum = 2.1). This is the runtime/memory-cost ranking
  the paper's weights are designed to express, but it shifts the
  selection pressure away from prism / frustum primitives that often
  catch corner outliers in detail-heavy meshes.

  Measured impact on the rock kit (1 component, near-convex):

  | N    | default | --weighted-cost |
  | ---- | ------- | --------------- |
  | 64   | 6.39%   | **5.68%** (-11%)|
  | 128  | 4.26%   | **3.93%** (-8%) |
  | 256  | 2.56%   | **2.05%** (-20%)|

  But on detail-heavy meshes the effect reverses (blink N=128: 5.04%
  → 6.72%; ram-visual N=256: 7.74% → 8.54%). Default off; turn on
  for organic / near-convex inputs (rocks, terrain, sculpted props).

- **Pancake-merge penalty (`--reject-pancakes`).** Multiplies the merge
  cost by 1000 when the resulting primitive's smallest half-extent has
  clamped to `MIN_HALF_EXTENT` *and* its aspect ratio is < 0.001. This
  pushes "1mm-thick × Nm-wide slab" merges to the bottom of the PQ —
  any non-degenerate alternative is preferred. Targets the failure
  mode where many disparate near-coplanar faces (rooftops across
  buildings, walkways across an environment) all merge into one giant
  flat primitive whose tessellated surface drifts metres past the
  actual input geometry.

  Measured on a fortified-building test mesh (architectural, large
  flat horizontal surfaces):

  | N    | default | --reject-pancakes |
  | ---- | ------- | ----------------- |
  | 64   | Haus 23.60% | **20.44%** (-13%), 55 prims |
  | 128  | Haus 23.60% | **20.44%** (-13%), 113 prims |
  | 256  | Haus 23.60% | **14.93%** (-37%), 217 prims |

  Trade-off: primitive count drops (rejected merges starve the
  algorithm), and detail-heavy meshes with long thin panels at the
  same threshold can regress (blink: +75% Hausdorff). Default off; opt
  in for buildings / architecture / environment art with prominent
  flat collisions.

- **Iterative face-rebalance (experimental, `--rebalance`).** After the
  greedy merge completes, run Lloyd-style face migration: for each
  boundary face, try moving it to each adjacent primitive; accept the
  move that most reduces a combined `weighted_volume × (1 + 5·h/diag)`
  cost. Keeps `N` constant (a primitive's last face can't migrate
  away). Iterates until no moves are accepted, capped at
  `--rebalance-passes` (default 5).

  **Marginal and inconsistent in practice.** Counts and volumes are
  preserved, but the metric improvements are small at low N and
  *regress* at higher N:

  | mesh | N | Hausdorff (off → on) |
  | ---- | -- | ------------------- |
  | rock | 64 | 6.39% → 6.29% (-1.6%) |
  | rock | 128 | 4.26% → 5.18% (worse) |
  | rock | 256 | 2.56% → 3.59% (worse) |
  | blink | 128 | 5.04% → 4.73% (-6%) |
  | ram-s | 128 | 5.60% → 8.32% (worse) |

  Why it underperforms: **Lloyd minimises the *sum* of per-primitive
  scores; Hausdorff is a *max* metric.** Local moves can improve a
  pair's combined score while making the global worst-case worse. To
  actually reduce Hausdorff you'd need to specifically target the
  worst-fit primitive each iteration (split-and-redistribute), which
  is a different algorithm. The current implementation is kept as an
  experimental option and as scaffolding (DSU + linked-list rebuild)
  for that future direction. Default off.

- **Adaptive sharp-edge threshold.** Replaces the fixed 30° dihedral
  threshold for sharp-edge feature detection with a per-mesh value:
  the 95th percentile of the actual dihedral distribution, clamped to
  [30°, 60°]. Prevents over-flagging tiny ridges on organic meshes
  (rocks at 43°) while leaving angular meshes near the cap (vehicles
  at 60°).

- **Spatial-proximity merges (`--proximity`).** Replaces the all-pairs
  fallback (paper §3.4) with a spatially-filtered version: when the
  topology PQ drains, candidate edges are pushed only between live
  primitives whose AABBs are within `--proximity-r` (fraction of input
  diagonal, default 0.05) and whose dominant Q-normals differ by at
  most `--proximity-angle` degrees (default 45). Each primitive gets
  at most `--proximity-k` neighbours (default 2). Post-merge candidate
  generation also uses these guards.

  Compared to all-pairs:
  - Avoids the "monster primitive" failure mode where unrelated
    distant components get wrapped into a single OBB.
  - Honest target N: refuses guard-failing merges, so the algorithm
    may stop short of the requested N (treat the request as a
    *minimum* primitive count).
  - 3× faster on heavily-fragmented inputs (ram-visual: 2.9 s → 0.9 s).

  Measured impact:

  | mesh         | N    | metric    | baseline (all-pairs) | --proximity |
  | ------------ | ---- | --------- | -------------------- | ----------- |
  | blink        | 64   | Hausdorff | 27.0% (forced merge) | **22.4%**   |
  | blink        | 64   | reached N | 64                   | 64          |
  | ram-visual   | 64   | reached N | 64 (forced)          | 237 (honest)|
  | ram-visual   | 64   | wall-time | 2940 ms              | **905 ms**  |

  When a forced low-N is required, leave `--proximity` off so the
  default all-pairs fallback fires. When quality matters more than the
  exact N, `--proximity` is the better default.

- **Shell-aware orientation (experimental, `--shell`).** Pre-computes
  per-face ambient-occlusion exposure: cast 32 stratified rays in each
  face's outward hemisphere via the BVH and score the fraction
  unobstructed. Faces below 5% AO are flagged as "interior". The Q
  matrix gets multiplied by exposure (so interior faces don't bias
  the area-weighted normal in `axes_from_q`), and PCA / tangent-plane
  PCA / sharp-edge construction are filtered to shell-only vertices.
  Containment fits still use every subsumed vertex, so the paper's
  enclosure guarantee is preserved.

  **Modest gains, mostly at low N.** Detection works correctly
  (rock kit: 100% shell, blink: 80%, ram-visual: 77%). Measured impact:

  | mesh         | N    | Hausdorff (off→on) | Chamfer (off→on) | volume (off→on) |
  | ------------ | ---- | ------------------ | ---------------- | --------------- |
  | rock kit     | 64   | 6.39% → **6.05%**  | 0.67% → 0.65%    | 11.8% → 11.4%   |
  | rock kit     | 128  | 4.26% (identical)  | 0.343% → 0.337%  | 5.6% (same)     |
  | blink        | 256  | 4.03% → 4.28%      | 0.44% → 0.46%    | 17.7% → 19.6%   |
  | ram-visual   | 256  | 7.74% → 8.73%      | 0.96% → 0.95%    | 67.4% → 68.3%   |

  Where it helps (low N on solid meshes), the gain is ~5% Hausdorff
  reduction. On vehicle meshes where interior is *contained* by the
  shell hull, the OBB extents are still pinned by shell vertices; what
  shell-awareness shifts is primitive-type selection (more cylinders,
  fewer prisms — shell-only Q is more cleanly rank-deficient). That
  shuffle doesn't reduce worst-case surface drift on those meshes.
  Where it would shine: scanner data with floating debris, or
  kitbashed assets with internal geometry poking outside the natural
  shell hull. The meshes here don't exhibit that.

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
