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
    [--strip-thin]                 postprocess: delete OBBs with min half-extent
                                   ≤ 1e-4 × diag (paper appendix Fig 22). Tanks
                                   coverage on small-diag meshes — read the section.
        [--strip-thin-threshold <f>]   override 1e-4
    [--feasibility <f>]            reject popped merges whose result has
                                   local Hausdorff > f × diag. Try 0.15 for
                                   architecture meshes; no-op on meshes
                                   without slab pathology.
    [--split-worst <f>]            experimental post-merge: split primitives
                                   with local Hausdorff > f × diag along
                                   their longest PCA / world axis (median
                                   split). Mixed results — see section.
        [--split-worst-max <int>]  cap on splits (default 32)
    [--no-tangent-eps]             disable Q's ε·ttᵀ in-plane bias (architecture)
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
| ext. | Reverse Hausdorff (input → primitive surface) + coverage fraction | ✓ |
| ext. | Merge-time feasibility check (rejects slab merges, paper §3.3 unsolved) | ✓ |
| ext. | Post-merge split-worst pass (experimental, mixed results) | ✓ |

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

- **Tangent-term knob (`--no-tangent-eps` / `--tangent-eps <ε>`).** The
  per-face quadric is `Q = area · (n nᵀ + ε · t tᵀ)` (paper §3.4
  "Coplanar Vertices"). The ε·ttᵀ term stabilises the eigendecomp on
  flat regions by giving Q a non-zero in-plane component. Default
  ε=0.01 matches the paper. The paper itself notes:

  > "We decide per mesh whether to include this factor."

  At ε=0, Q is rank-1 for a single face; in-plane axis decisions are
  handed entirely to the refit pass (PCA / tangent-plane PCA /
  sharp-edge). This **removes the rotated-OBB failure mode on
  large flat regions** — the worst primitive on our test building goes
  from 60°-rotated axes (slab corners drift past the rooftop) to
  world-axis-aligned axes. But it costs vehicle/organic meshes some
  numerical stability, so it's opt-in.

  Measured impact:

  | mesh                    | N    | default (ε=0.01) | --no-tangent-eps |
  | ----------------------- | ---- | ---------------- | ---------------- |
  | building (architecture) | 64   | 23.60% Hausdorff | **19.97%** (-15%)|
  | building                | 128  | 23.60%           | **21.07%** (-11%)|
  | rock kit                | 64   | 6.39%            | 7.22% (+13%)     |
  | blink-visual            | 128  | 5.04%            | 7.33% (+46%)     |
  | ram-visual              | 256  | 8.54%            | 9.13% (+7%)      |

  **Architecture combo:** `--no-tangent-eps --reject-pancakes
  --empty-space --empty-space-fraction 0.10` on the test building
  drops Hausdorff from 23.60% → **14.54%** at N=64 (-38%) and is
  visually much cleaner (primitive footprint matches building outline
  on top/bottom views, no slab pancakes protruding past silhouette).

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

- **Postprocess thin-OBB strip (`--strip-thin`).** Paper appendix
  Fig 22 (Bistro scene) recipe: after the merge + redundant cull
  complete, delete every OBB whose smallest half-extent is
  ≤ `1e-4 × mesh_diag`. Override the fraction with
  `--strip-thin-threshold <f>`. The paper uses this to clean up
  "many walls are entirely planar but may not be rectangular,
  leading to regions jutting out" — the slab failure mode. Distinct
  from `--reject-pancakes`: this lets the merge fully converge,
  *then* deletes slabs as a postprocess.

  **The headline numbers look great. They lie.** On the test
  building it cuts forward Hausdorff from 23.6% → 4.5% at N=128 —
  paper-territory. But the forward (paper §4.4) Hausdorff measures
  primitive-surface drift past the input only; it does not penalise
  *uncovered input*. Strip too aggressively and you delete
  legitimate surface-fitted primitives. Forward Hausdorff stays
  low (the surviving primitives sit on the input), but coverage
  collapses. The reverse Hausdorff and coverage-fraction metrics
  added to `--metrics` expose this honestly:
  
  | N    | mode    | fwd H% | rev H%  | 2-way H% | coverage |
  | ---- | ------- | ------ | ------- | -------- | -------- |
  | 128  | default | 23.60  | 1.78    | 23.60    | **99.6%**|
  | 128  | strip   | 4.47   | 17.07   | 17.07    | **46.8%**|
  | 256  | default | 23.60  | 0.82    | 23.60    | **99.3%**|
  | 256  | strip   | 4.78   | 24.98   | 24.98    | **24.8%**|
  | 512  | strip   | 4.22   | 26.70   | 26.70    | **13.3%**|

  At N=512, `--strip-thin` deletes ~75% of primitives; the surviving
  225 cover only 13% of the input. By the symmetric 2-way metric,
  strip is *worse* than baseline at every N. Why the paper gets away
  with it on Bistro: their mesh diag is so much larger that the
  `1e-4` threshold (~5cm in absolute terms) sits below real wall
  thickness. On our 69m-diag building, `1e-4` is 6.9mm — most
  legitimate surface fits clamp to ≤1mm and get deleted.

  Kept as an option so you can reproduce the paper's recipe, and
  because on meshes with large diag and thick walls (true
  environment scenes — Bistro-class) it should still work as
  documented. Default off. **Always pair with `--metrics` and check
  reverse-Hausdorff + coverage** before claiming an improvement.

- **Merge-time feasibility check (`--feasibility <f>`).** Direct
  attack on the slab failure mode. The PQ cost
  (V(merge) − V(p0) − V(p1)) is ≈ 0 when two near-coplanar slabs
  merge — both have ~0 volume (1mm thickness clamp) and the merged
  slab also has ~0 volume. The cost function literally cannot see
  that the merged primitive's *surface* drifts metres past the
  input. Paper §3.3 acknowledges this; their alternative (Eq 5,
  including V(p0 ∩ p1)) they call intractable.

  This flag adds a hard reject: after a merge candidate is popped
  and the resulting primitive is fitted (refit included), sample its
  surface against the input mesh BVH using `local_hausdorff` (12
  vertex + 12 triangle-centroid samples) and reject the merge if
  `h > f × mesh_diag`. The two source primitives stay alive, the
  algorithm picks the next-best candidate.

  This optimises the actual metric we care about, at the cost of
  ~24 BVH nearest-face queries per realized merge.

  **Building (the architecture failure case), N=128:**

  | flag                    | fwd H% | rev H% | 2-way H% | coverage | rejects |
  | ----------------------- | ------ | ------ | -------- | -------- | ------- |
  | (default)               | 23.60  | 1.78   | 23.60    | 99.6%    | 0       |
  | `--feasibility 0.20`    | 10.18  | 2.50   | 10.18    | 99.6%    | 7       |
  | **`--feasibility 0.15`**| **10.18**| 1.73 | **10.18**| 99.6%    | 7       |
  | `--feasibility 0.10`    | 13.57  | 2.50   | 13.57    | 99.6%    | 13      |
  | `--feasibility 0.05`    | 15.91  | -      | 15.91    | 99.5%    | 40      |

  Tighter thresholds reject too many cheap merges; the algorithm
  starves and accepts more expensive ones with bigger drift. 0.15
  hits a clean optimum across N=64 (12.75%), N=128 (10.18%), N=256
  (11.99%), N=512 (12.75%) — all roughly half of the 23.60% / 23.27%
  baseline. Coverage stays ≥99.3% at every config (compare to
  `--strip-thin` which collapsed coverage to 13–47%).

  **Other meshes — no-op by construction.** Rock kit, blink, and
  ram show 0 feasibility-rejections at `f=0.15`: their merges all
  have local Hausdorff well below 15% of diag. Identical metrics
  with and without the flag:

  | mesh   | N   | default fwd H% | --feasibility 0.15 fwd H% |
  | ------ | --- | -------------- | ------------------------- |
  | rock   | 128 | 4.26           | 4.26                      |
  | rock   | 256 | 2.56           | 2.56                      |
  | blink  | 128 | 6.26           | 6.16                      |
  | blink  | 256 | 3.54           | 3.54                      |
  | ram    | 128 | 5.60           | 5.60                      |
  | ram    | 256 | 4.41           | 4.41                      |

  **Comparison to paper.** Pre-fix the building was 5× off paper
  mean (23.6% vs 4.45%); post-fix at 10.18% it sits in the same
  range as the paper's harder buildings (Chuo House 10.23%,
  Jpn House 9.14%, Lantern 11.23%, Dojo 11.11%). Not paper-best
  (Pipe Wall 0.98%, Jpn House 2 1.59%) — but the gap is no longer
  algorithmic. It's a "convex primitives can't fit non-rectangular
  planar regions tightly" gap, the same one driving the paper's own
  failure cases.

  Default off (0.15 doesn't cost anything on non-architecture
  meshes, but is a behavioural change). Recommended for any
  environment / architectural input.

- **Post-merge split-worst (experimental, `--split-worst <f>`).**
  After merge converges (and rebalance + redundant cull run, if
  enabled), repeatedly find the live primitive with highest local
  Hausdorff > `f × mesh_diag` and split its face set into two new
  primitives. Each accepted split adds 1 primitive. Targets the
  follow-on of the slab failure mode that feasibility can't reach:
  primitives that *individually* pass the merge-time feasibility
  check but still drift past their input region (e.g. an OBB on an
  L-shaped rooftop — every individual merge step looked fine, but
  the cumulative result is a rectangular OBB that doesn't fit the
  L's outline).

  Split mechanics:
  1. Sort the primitive's faces by their centroid's projection onto
     a candidate axis. Try 6 axes — all 3 PCA axes of the vertex
     cloud + the 3 world axes — and pick the one that produces the
     lowest combined local Hausdorff between the two halves.
  2. Median split into two face sets. Refit each half.
  3. Accept iff both halves have strictly lower local Hausdorff
     than the parent. Otherwise mark "tried" and move on.
  4. Local Hausdorff for split decisions uses a 280-sample dense
     variant (vertices + edge midpoints + 256 area-weighted random
     barycentric points) — the 24-sample version used by rebalance
     mis-ranked which primitive was actually worst, causing splits
     to make the global metric *worse*.

  **Mixed and inconsistent in practice — like `--rebalance`.**
  Building (paper §4.4 forward Hausdorff, 100k samples to cut
  metric noise):

  | N    | default | --feasibility 0.15 | --split-worst 0.05 |
  | ---- | ------- | ------------------ | ------------------ |
  | 64   | 23.60   | **12.39**          | 23.60 (no-op)      |
  | 128  | 23.60   | 11.84              | 12.03              |
  | 256  | 23.60   | 11.99              | **11.34**          |
  | 512  | 23.27   | 12.75              | **11.30**          |

  Split helps building at higher N (where the budget allows the
  added primitives) and matches feasibility at moderate N. At low
  N (≤64) it doesn't fire because the budget is too tight to
  afford new primitives.

  Other meshes (10k samples, more noise):

  | mesh   | N   | default | --split-worst 0.05 | Δ        |
  | ------ | --- | ------- | ------------------ | -------- |
  | rock   | 128 | 4.26    | 4.60               | +0.34pp  |
  | blink  | 128 | 6.26    | **4.38**           | -1.88pp  |
  | ram    | 128 | 5.60    | 6.10               | +0.50pp  |

  Blink (mechanical/vehicle) shows real ~30% Hausdorff
  improvement: the splits hit the long thin panels where one OBB
  was bridging multiple disconnected sub-panels. Rock and ram
  show small regressions.

  **Do not combine with `--feasibility`.** Empirically the two
  flags are anti-synergistic on the building (combo: ~16% vs
  ~12% for either alone). Feasibility blocks the worst slab
  merges leaving primitives that are tight-but-not-perfect; split
  then attacks them and produces sliver alternatives that drift
  similarly. Use one or the other, not both.

  Why it underperforms a clean win:
  - Median-split-along-PCA-axis is the wrong cut for L-shaped
    regions: PCA's longest axis is the L's *diagonal*, and a
    median cut at 45° doesn't separate the two arms. Sweeping all
    3 PCA + 3 world axes helps but doesn't fully fix it.
  - The acceptance check (both halves' local Hausdorff strictly
    less than parent's) bounds the local result but the *new
    surface* introduced where the two halves abut isn't on the
    input mesh, and can drift in its own right.

  Future directions: k-means clustering of vertices for cleaner
  L-arm separation; multi-position splits (33/66, not just 50/50);
  recursive split-and-merge to escape local minima. Default off,
  same status as `--rebalance` — kept as an experimental option
  and as scaffolding for those follow-ups.

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

- **Self-evaluation loop.** `--metrics` computes:
  - **Forward Hausdorff/Chamfer** (paper §4.4): primitive surface →
    input mesh, via the input-mesh BVH. Penalises primitives that
    drift past the input.
  - **Reverse Hausdorff/Chamfer** (extension): input mesh → nearest
    primitive surface, via a BVH built over the union of all live
    primitive tessellations. Penalises uncovered input.
  - **2-way Hausdorff** (max of fwd and rev): the standard
    symmetric metric — the single honest number to compare runs.
  - **Coverage fraction**: fraction of input-mesh samples that lie
    inside *some* live primitive's volume. Direct collision-detection
    proxy. 1.0 = every input point covered.

  Forward Hausdorff alone can be fooled by deleting primitives — see
  the `--strip-thin` writeup above for a worked example. Always
  cross-check against reverse + coverage before trusting an
  optimisation that drops forward Hausdorff but also drops primitive
  count.

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
