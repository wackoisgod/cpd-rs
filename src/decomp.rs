use crate::bvh::Bvh;
use crate::dsu::Dsu;
use crate::mesh::{Adjacency, Mesh, SharpEdges};
use crate::prim::{self, Prim, PrimMask};
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

const TANGENT_EPS: f32 = 0.01;

pub fn face_quadric(p0: Point3<f32>, p1: Point3<f32>, p2: Point3<f32>) -> Matrix3<f32> {
    let e0 = p1 - p0;
    let e1 = p2 - p1;
    let e2 = p0 - p2;
    let cross = e0.cross(&(p2 - p0));
    let area2 = cross.norm();
    if area2 < 1e-20 {
        return Matrix3::zeros();
    }
    let area = 0.5 * area2;
    let n = cross / area2;

    // Quad-aware tangent (paper §3.4 "Coplanar Vertices"). Sort the three
    // edges by length; treat the longest as if it were a quad's diagonal,
    // and synthesize the "halfway" tangent t0 = ½(e0 - e1 + e2). This is
    // what the paper does when triangulating a quad.
    let mut edges = [e0, e1, e2];
    edges.sort_by(|a, b| {
        a.norm_squared()
            .partial_cmp(&b.norm_squared())
            .unwrap_or(Ordering::Equal)
    });
    let t_raw = 0.5 * (edges[0] - edges[1] + edges[2]);
    // project onto the face plane and normalize
    let t_in_plane = t_raw - n * n.dot(&t_raw);
    let t = if t_in_plane.norm() > 1e-8 {
        t_in_plane.normalize()
    } else {
        // fallback: longest edge direction projected out of normal
        let fallback = edges[2].normalize();
        let in_plane = fallback - n * n.dot(&fallback);
        if in_plane.norm() > 1e-8 {
            in_plane.normalize()
        } else {
            // mesh is fully degenerate; pick anything orthogonal
            let helper = if n.x.abs() < 0.9 {
                Vector3::new(1.0, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            n.cross(&helper).normalize()
        }
    };

    area * (n * n.transpose() + TANGENT_EPS * t * t.transpose())
}

pub fn axes_from_q(q: Matrix3<f32>) -> [Vector3<f32>; 3] {
    let sym = (q + q.transpose()) * 0.5;
    let dec = SymmetricEigen::new(sym);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        dec.eigenvalues[j]
            .abs()
            .partial_cmp(&dec.eigenvalues[i].abs())
            .unwrap_or(Ordering::Equal)
    });
    let mut axes = [
        dec.eigenvectors.column(idx[0]).into_owned(),
        dec.eigenvectors.column(idx[1]).into_owned(),
        dec.eigenvectors.column(idx[2]).into_owned(),
    ];
    axes[0] = axes[0].normalize();
    axes[1] = (axes[1] - axes[0] * axes[0].dot(&axes[1])).normalize();
    axes[2] = axes[0].cross(&axes[1]).normalize();
    axes
}

/// Build an orthonormal basis whose first axis is `axis0`, with the other
/// two derived from the principal direction of `axis_seed` projected into
/// the plane perpendicular to `axis0`. Returns axes ordered as
/// `[axis0, in_plane_primary, in_plane_secondary]`.
fn orthonormal_basis_from_seed(
    axis0: Vector3<f32>,
    axis_seed: Vector3<f32>,
) -> [Vector3<f32>; 3] {
    let a0 = axis0.normalize();
    let mut a1 = axis_seed - a0 * a0.dot(&axis_seed);
    if a1.norm_squared() < 1e-12 {
        let helper = if a0.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        a1 = helper - a0 * a0.dot(&helper);
    }
    let a1 = a1.normalize();
    let a2 = a0.cross(&a1).normalize();
    [a0, a1, a2]
}

/// Principal axes from the centered covariance of a vertex set. Unlike
/// `axes_from_q` (driven by area-weighted face normals), PCA captures the
/// geometric extent of the merged region, which gives better orientations
/// for elongated, ridged, or co-planar point clouds. Used as a second
/// orientation candidate during post-merge refinement.
pub fn pca_axes(points: &[Point3<f32>]) -> [Vector3<f32>; 3] {
    if points.len() < 2 {
        return [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
    }
    let n = points.len() as f32;
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    centroid /= n;
    let mut cov = Matrix3::zeros();
    for p in points {
        let d = p.coords - centroid;
        cov += d * d.transpose();
    }
    cov /= n;
    let sym = (cov + cov.transpose()) * 0.5;
    let dec = SymmetricEigen::new(sym);
    // Sort by eigenvalue descending — axes[0] is the longest direction.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        dec.eigenvalues[j]
            .partial_cmp(&dec.eigenvalues[i])
            .unwrap_or(Ordering::Equal)
    });
    let v0 = dec.eigenvectors.column(idx[0]).into_owned();
    let v1 = dec.eigenvectors.column(idx[1]).into_owned();
    let mut axes = [v0, v1, Vector3::zeros()];
    axes[0] = axes[0].normalize();
    if axes[1].norm_squared() < 1e-20 {
        // pick any vector orthogonal to axes[0]
        let helper = if axes[0].x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        axes[1] = (helper - axes[0] * axes[0].dot(&helper)).normalize();
    } else {
        axes[1] = (axes[1] - axes[0] * axes[0].dot(&axes[1])).normalize();
    }
    axes[2] = axes[0].cross(&axes[1]).normalize();
    axes
}

/// Tangent-plane PCA: fix the normal direction (Q's largest eigenvector),
/// then find the in-plane principal direction by projecting points onto the
/// plane perpendicular to that normal and doing 2D PCA there. Targets the
/// "auto-tangent-weight" failure case the paper notes — coplanar regions
/// where the normal is well-defined but in-plane axes from `axes_from_q`
/// degenerate.
pub fn tangent_plane_pca_axes(q: Matrix3<f32>, points: &[Point3<f32>]) -> [Vector3<f32>; 3] {
    let q_axes = axes_from_q(q);
    let normal = q_axes[0];
    if points.len() < 2 {
        return q_axes;
    }
    let n = points.len() as f32;
    let mut centroid = Vector3::zeros();
    for p in points {
        centroid += p.coords;
    }
    centroid /= n;
    let mut cov = Matrix3::zeros();
    for p in points {
        let mut d = p.coords - centroid;
        d -= normal * normal.dot(&d); // project into plane
        cov += d * d.transpose();
    }
    cov /= n;
    let sym = (cov + cov.transpose()) * 0.5;
    let dec = SymmetricEigen::new(sym);
    // The two largest eigenvalues give the two in-plane principal
    // directions; the smallest is along `normal` and is ~0.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        dec.eigenvalues[j]
            .partial_cmp(&dec.eigenvalues[i])
            .unwrap_or(Ordering::Equal)
    });
    let in_plane_seed = dec.eigenvectors.column(idx[0]).into_owned();
    orthonormal_basis_from_seed(normal, in_plane_seed)
}

/// Sharp-edge PCA: build a covariance from the unit directions of all
/// sharp (high-dihedral) edges in the merged primitive's face set, then
/// extract the dominant direction. Returns `None` if too few sharp edges
/// to form a meaningful axis. Targets feature-aligned geometry: building
/// corners, ridge lines, stair treads.
pub fn sharp_edge_axes(
    sharp: &SharpEdges,
    face_iter: impl Iterator<Item = u32>,
) -> Option<[Vector3<f32>; 3]> {
    let mut cov = Matrix3::zeros();
    let mut count = 0usize;
    for fi in face_iter {
        for d in &sharp.per_face[fi as usize] {
            // outer product is sign-invariant; counting an edge from each
            // adjacent face just doubles the weight uniformly, no bias.
            cov += d * d.transpose();
            count += 1;
        }
    }
    if count < 3 {
        return None;
    }
    let sym = (cov + cov.transpose()) * 0.5;
    let dec = SymmetricEigen::new(sym);
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        dec.eigenvalues[j]
            .partial_cmp(&dec.eigenvalues[i])
            .unwrap_or(Ordering::Equal)
    });
    let v0 = dec.eigenvectors.column(idx[0]).into_owned();
    let v1 = dec.eigenvectors.column(idx[1]).into_owned();
    Some(orthonormal_basis_from_seed(v0, v1))
}

/// Sample both tessellation vertices (catch corner protrusions on OBBs/
/// prisms) and triangle centroids (catch face-bulge cases on smooth
/// primitives), and return the max distance to the input mesh. Combining
/// both gives stabler ranking than either alone — corners alone over-
/// penalises OBBs at the merger granularity where their corners aren't
/// the actual fit problem; centroids alone miss real OBB-corner outliers.
fn local_hausdorff(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
    const K_VERT: usize = 12;
    const K_TRI: usize = 12;
    let (verts, tris) = prim::tessellate(p);
    let mut max_d = 0.0f32;

    if !verts.is_empty() {
        let stride = (verts.len() / K_VERT).max(1);
        let mut vi = 0usize;
        let mut count = 0usize;
        while vi < verts.len() && count < K_VERT {
            let v = verts[vi];
            let q = Point3::new(v[0], v[1], v[2]);
            let (_pt, _n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, q);
            let d = signed.abs();
            if d > max_d {
                max_d = d;
            }
            count += 1;
            vi += stride;
        }
    }
    if !tris.is_empty() {
        let stride = (tris.len() / K_TRI).max(1);
        let mut ti = 0usize;
        let mut count = 0usize;
        while ti < tris.len() && count < K_TRI {
            let t = tris[ti];
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let q = Point3::new(
                (a[0] + b[0] + c[0]) / 3.0,
                (a[1] + b[1] + c[1]) / 3.0,
                (a[2] + b[2] + c[2]) / 3.0,
            );
            let (_pt, _n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, q);
            let d = signed.abs();
            if d > max_d {
                max_d = d;
            }
            count += 1;
            ti += stride;
        }
    }
    max_d
}

/// Walk the cyclic linked list of faces for a primitive starting at `start`,
/// yielding `count` distinct face indices.
fn walk_faces<'a>(start: u32, count: u32, face_next: &'a [u32]) -> impl Iterator<Item = u32> + 'a {
    let mut current = start;
    let mut remaining = count;
    std::iter::from_fn(move || {
        if remaining == 0 {
            return None;
        }
        let f = current;
        current = face_next[current as usize];
        remaining -= 1;
        Some(f)
    })
}

#[derive(Clone)]
pub struct Primitive {
    pub alive: bool, // == dsu.is_root(self_id) for live primitives
    pub version: u64,
    pub q: Matrix3<f32>,
    pub prim: Prim,
    pub volume: f32,
    pub weighted_volume: f32,
    /// Number of mesh faces this primitive subsumes. Walk `face_next` from
    /// any face known to be in this primitive (e.g., the DSU root face) for
    /// `face_count` steps to enumerate them.
    pub face_count: u32,
    pub vertex_indices: Vec<u32>, // sorted, unique
    pub neighbors: HashSet<u32>,
}

#[derive(Clone)]
struct PqEntry {
    cost: f32,
    a: u32,
    b: u32,
    va: u64,
    vb: u64,
    /// The fitted primitive computed at push time. Cached here so that valid
    /// pops don't redo `fit_best` — saves ~15% of merge_pair calls.
    prim: Prim,
    volume: f32,
    weighted_volume: f32,
}

impl PartialEq for PqEntry {
    fn eq(&self, o: &Self) -> bool {
        self.cost == o.cost
    }
}
impl Eq for PqEntry {}
impl PartialOrd for PqEntry {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        o.cost.partial_cmp(&self.cost)
    }
}
impl Ord for PqEntry {
    fn cmp(&self, o: &Self) -> Ordering {
        self.partial_cmp(o).unwrap_or(Ordering::Equal)
    }
}

fn merge_sorted_unique(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
            Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn gather(verts: &[Point3<f32>], idx: &[u32]) -> Vec<Point3<f32>> {
    idx.iter().map(|&i| verts[i as usize]).collect()
}

fn merge_pair(
    a: &Primitive,
    b: &Primitive,
    mesh_verts: &[Point3<f32>],
    enabled: PrimMask,
) -> (Matrix3<f32>, Prim, f32, f32, Vec<u32>) {
    let q = a.q + b.q;
    let axes = axes_from_q(q);
    let vidx = merge_sorted_unique(&a.vertex_indices, &b.vertex_indices);
    let pts = gather(mesh_verts, &vidx);
    let prim_fit = prim::fit_best(axes, &pts, enabled);
    let vol = prim_fit.volume();
    let wvol = prim_fit.weighted_volume();
    (q, prim_fit, vol, wvol, vidx)
}

fn live_indices(prims: &[Primitive]) -> Vec<u32> {
    prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive)
        .map(|(i, _)| i as u32)
        .collect()
}

/// Per-primitive bookkeeping for the proximity fallback.
struct LiveSummary {
    aabbs: Vec<(Point3<f32>, Point3<f32>)>,
    normals: Vec<Vector3<f32>>,
}

fn live_summary(prims: &[Primitive], live: &[u32]) -> LiveSummary {
    let mut aabbs = Vec::with_capacity(live.len());
    let mut normals = Vec::with_capacity(live.len());
    for &i in live {
        let (lo, hi) = prim::world_aabb(&prims[i as usize].prim);
        aabbs.push((Point3::new(lo[0], lo[1], lo[2]), Point3::new(hi[0], hi[1], hi[2])));
        let q = prims[i as usize].q;
        let sym = (q + q.transpose()) * 0.5;
        let dec = SymmetricEigen::new(sym);
        let mut max_i = 0;
        for k in 1..3 {
            if dec.eigenvalues[k].abs() > dec.eigenvalues[max_i].abs() {
                max_i = k;
            }
        }
        let v = dec.eigenvectors.column(max_i).into_owned();
        let nv = if v.norm_squared() > 1e-20 {
            v.normalize()
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        normals.push(nv);
    }
    LiveSummary { aabbs, normals }
}

fn aabb_to_aabb_dist(
    a: &(Point3<f32>, Point3<f32>),
    b: &(Point3<f32>, Point3<f32>),
) -> f32 {
    let mut d2 = 0.0f32;
    for i in 0..3 {
        let gap = if a.1[i] < b.0[i] {
            b.0[i] - a.1[i]
        } else if b.1[i] < a.0[i] {
            a.0[i] - b.1[i]
        } else {
            0.0
        };
        d2 += gap * gap;
    }
    d2.sqrt()
}

/// Spatial-proximity fallback: for each live primitive, push k candidate
/// edges to its closest neighbours by AABB distance, dropping pairs whose
/// dominant Q-normals differ by more than `max_angle_rad`. Compared to
/// the brute all-pairs phase this dramatically narrows the candidate set
/// on heavily fragmented meshes.
fn push_proximity_pairs(
    prims: &[Primitive],
    mesh_verts: &[Point3<f32>],
    pq: &mut BinaryHeap<PqEntry>,
    volume_threshold: f32,
    enabled: PrimMask,
    max_dist: f32,
    k: usize,
    max_angle_rad: f32,
    weighted_cost: bool,
    reject_pancakes: bool,
) -> usize {
    let live = live_indices(prims);
    if live.len() < 2 {
        return 0;
    }
    let summary = live_summary(prims, &live);
    let cos_min = max_angle_rad.cos();

    let mut pair_keys: HashSet<(u32, u32)> = HashSet::new();
    for i in 0..live.len() {
        let mut dists: Vec<(usize, f32)> = (0..live.len())
            .filter(|&j| j != i)
            .map(|j| (j, aabb_to_aabb_dist(&summary.aabbs[i], &summary.aabbs[j])))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        for &(j, d) in dists.iter().take(k) {
            if d > max_dist {
                continue;
            }
            let cos = summary.normals[i].dot(&summary.normals[j]).abs();
            if cos < cos_min {
                continue;
            }
            let a = live[i];
            let b = live[j];
            if prims[a as usize].neighbors.contains(&b) {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            pair_keys.insert(key);
        }
    }

    let pairs: Vec<(u32, u32)> = pair_keys.into_iter().collect();
    let entries: Vec<PqEntry> = pairs
        .par_iter()
        .filter_map(|&(a, b)| {
            let pa = &prims[a as usize];
            let pb = &prims[b as usize];
            let (_q, prim_fit, vol, wvol, _vidx) =
                merge_pair(pa, pb, mesh_verts, enabled);
            let mut cost = if weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            if reject_pancakes && prim_fit.is_pancake() {
                // Push pancake-producing merges far down the PQ. Not infinite,
                // so they can still be popped as a last resort if every
                // candidate is degenerate.
                cost = cost.abs() * 1000.0 + 1e6;
            }
            if cost > volume_threshold {
                return None;
            }
            Some(PqEntry {
                cost,
                a,
                b,
                va: pa.version,
                vb: pb.version,
                prim: prim_fit,
                volume: vol,
                weighted_volume: wvol,
            })
        })
        .collect();
    let pushed = entries.len();
    for e in entries {
        pq.push(e);
    }
    pushed
}

fn push_all_pairs(
    prims: &[Primitive],
    mesh_verts: &[Point3<f32>],
    pq: &mut BinaryHeap<PqEntry>,
    volume_threshold: f32,
    enabled: PrimMask,
    weighted_cost: bool,
    reject_pancakes: bool,
) -> usize {
    let live = live_indices(prims);
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..live.len() {
        for j in (i + 1)..live.len() {
            let a = live[i] as usize;
            let b = live[j] as usize;
            if prims[a].neighbors.contains(&(b as u32)) {
                continue;
            }
            pairs.push((live[i], live[j]));
        }
    }
    let entries: Vec<PqEntry> = pairs
        .par_iter()
        .filter_map(|&(a, b)| {
            let pa = &prims[a as usize];
            let pb = &prims[b as usize];
            let (_q, prim_fit, vol, wvol, _vidx) = merge_pair(pa, pb, mesh_verts, enabled);
            let mut cost = if weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            if reject_pancakes && prim_fit.is_pancake() {
                // Push pancake-producing merges far down the PQ. Not infinite,
                // so they can still be popped as a last resort if every
                // candidate is degenerate.
                cost = cost.abs() * 1000.0 + 1e6;
            }
            if cost > volume_threshold {
                return None;
            }
            Some(PqEntry {
                cost,
                a,
                b,
                va: pa.version,
                vb: pb.version,
                prim: prim_fit,
                volume: vol,
                weighted_volume: wvol,
            })
        })
        .collect();
    let pushed = entries.len();
    for e in entries {
        pq.push(e);
    }
    pushed
}

pub struct DecompResult {
    pub primitives: Vec<Primitive>,
    pub merges_done: usize,
    pub merges_skipped_stale: usize,
    pub merges_rejected_empty: usize,
    pub all_pairs_used: bool,
    pub redundant_culled: usize,
    pub rebalance_moves: usize,
}

pub struct DecompOpts {
    pub target_n: usize,
    pub volume_threshold: f32,
    pub enabled: PrimMask,
    pub cull_redundant: bool,
    /// Empty-space preservation. None disables the check entirely.
    /// Some((max_bridge_fraction, signed_dist_threshold)) enables it.
    pub empty_space: Option<(f32, f32)>,
    /// If true, refit the merged primitive against multiple orientation
    /// candidates (Q eigenbasis + vertex PCA) on each valid pop. Adds a
    /// little cost per realized merge but tightens fits on elongated /
    /// near-coplanar regions where Q's axes can be biased.
    pub refine_orient: bool,
    /// Hausdorff-aware refit. When > 0, the post-merge refit picks the
    /// primitive minimising `weighted_volume * (1 + beta * h/diag)` where
    /// `h` is sampled from the candidate primitive's surface to the input
    /// mesh via a BVH nearest-face query. Sphere becomes a real candidate
    /// in this mode (it usually loses on raw volume but can win on
    /// Hausdorff for near-spherical regions). 0 disables.
    pub quality_beta: f32,
    /// Shell-aware orientation. When true, per-face ambient-occlusion
    /// exposure is computed up-front and used to weight Q (so interior
    /// faces don't bias the area-weighted normal) and to filter PCA /
    /// tangent-plane PCA / sharp-edge inputs to outer-shell vertices
    /// only. Containment fitting still uses every subsumed vertex, so
    /// the paper's enclosure guarantee is preserved.
    pub shell_aware: bool,
    /// Spatial-proximity candidate merges. None disables; Some(...)
    /// enables. The (max_dist_frac, k_nearest, max_angle_rad) tuple
    /// adds, before merging starts, candidate edges between components
    /// whose AABBs are within `max_dist_frac * scene_diag`, capping at
    /// `k_nearest` neighbours per component, and rejecting pairs whose
    /// dominant normals differ by more than `max_angle_rad`.
    pub proximity: Option<(f32, usize, f32)>,
    /// Use the weighted volume in priority-queue ordering (cost = ΔwV
    /// instead of ΔV). Trades surface fit for runtime/memory cost
    /// (matches what the paper's per-shape weights represent). Helps
    /// near-convex / organic meshes (rocks: ~10-20% Hausdorff drop) at
    /// the cost of detail-heavy meshes (vehicles: 10-37% Hausdorff
    /// regression).
    pub weighted_cost: bool,
    /// Lloyd-style face-migration rebalance after the greedy merge
    /// completes. None disables; Some(max_passes) iterates that many
    /// passes (early-exits when no moves happen). Each pass: for every
    /// boundary face, try moving it to each adjacent primitive, accept
    /// the move that most reduces summed local Hausdorff. Keeps N
    /// constant. Targets greedy local minima at low N.
    pub rebalance: Option<usize>,
    /// When true, merges that produce a "pancake-degenerate" primitive
    /// (smallest half-extent at MIN_HALF_EXTENT clamp AND aspect ratio <
    /// 0.001) get a 1000× cost multiplier in the priority queue, pushing
    /// them to the bottom. Targets the failure mode where many disparate
    /// near-coplanar faces from across the mesh merge into one giant
    /// 1mm-thick slab whose surface drifts metres from the input. Fixes
    /// big-architecture meshes (-13% to -37% Hausdorff on the test
    /// building) but can regress vehicles whose long-narrow panels fall
    /// on the same side of the threshold (blink: +75% Hausdorff).
    pub reject_pancakes: bool,
}

/// Fraction of stratified-grid samples inside `prim` that are deeper than
/// `signed_dist_threshold` *outside* the input mesh. Used to reject merges
/// that bridge open regions (stairwells, holes, vents, slots).
///
/// "Outside" here is determined by the sign of the dot product of the face
/// normal with (sample − closest_point). For non-watertight meshes this is
/// more robust than generalized winding number, which can flicker near
/// boundaries.
fn empty_space_fraction(
    p: &Prim,
    mesh: &Mesh,
    bvh: &Bvh,
    signed_dist_threshold: f32,
) -> f32 {
    const GRID: usize = 3; // 3x3x3 = 27 candidate samples
    let (lo, hi) = prim::world_aabb(p);
    let mut inside = 0u32;
    let mut bridged = 0u32;
    for ix in 0..GRID {
        for iy in 0..GRID {
            for iz in 0..GRID {
                let tx = (ix as f32 + 0.5) / GRID as f32;
                let ty = (iy as f32 + 0.5) / GRID as f32;
                let tz = (iz as f32 + 0.5) / GRID as f32;
                let q = Point3::new(
                    lo[0] + tx * (hi[0] - lo[0]),
                    lo[1] + ty * (hi[1] - lo[1]),
                    lo[2] + tz * (hi[2] - lo[2]),
                );
                if !p.contains(q, 0.0) {
                    continue;
                }
                inside += 1;
                let (_pt, _n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, q);
                if signed > signed_dist_threshold {
                    bridged += 1;
                }
            }
        }
    }
    if inside == 0 {
        return 0.0;
    }
    bridged as f32 / inside as f32
}

pub fn run(mesh: &Mesh, adj: &Adjacency, opts: DecompOpts) -> DecompResult {
    let nf = mesh.tris.len();

    // Build BVH up-front (needed by exposure / quality / empty-space /
    // rebalance).
    let bvh: Option<Bvh> = if opts.empty_space.is_some()
        || opts.quality_beta > 0.0
        || opts.shell_aware
        || opts.rebalance.is_some()
    {
        Some(Bvh::build(&mesh.verts, &mesh.tris))
    } else {
        None
    };
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);

    let face_exposure: Option<Vec<f32>> = if opts.shell_aware {
        let bvh_ref = bvh.as_ref().expect("bvh built when shell_aware");
        let exp = crate::mesh::compute_face_exposure(mesh, bvh_ref, 32);
        let n_shell = exp.iter().filter(|&&e| e > 0.05).count();
        eprintln!(
            "shell-aware: {} of {} faces are exposed (>5% AO)",
            n_shell,
            exp.len()
        );
        Some(exp)
    } else {
        None
    };
    let shell_vertex_mask: Option<Vec<bool>> = face_exposure.as_ref().map(|exp| {
        let mut mask = vec![false; mesh.verts.len()];
        for (fi, t) in mesh.tris.iter().enumerate() {
            if exp[fi] > 0.05 {
                mask[t[0] as usize] = true;
                mask[t[1] as usize] = true;
                mask[t[2] as usize] = true;
            }
        }
        mask
    });

    let mut prims: Vec<Primitive> = Vec::with_capacity(nf);
    for (fi, tri) in mesh.tris.iter().enumerate() {
        let p0 = mesh.verts[tri[0] as usize];
        let p1 = mesh.verts[tri[1] as usize];
        let p2 = mesh.verts[tri[2] as usize];
        // Q is linear in face area, so multiplying the per-face quadric
        // by exposure simply down-weights interior faces in any later
        // Q_a + Q_b sum during merging.
        let q_unit = face_quadric(p0, p1, p2);
        let q = match &face_exposure {
            Some(exp) => q_unit * exp[fi],
            None => q_unit,
        };
        let axes = axes_from_q(q);
        let mut vidx = [tri[0], tri[1], tri[2]];
        vidx.sort();
        let pts = [p0, p1, p2];
        let prim_fit = prim::fit_best(axes, &pts, opts.enabled);
        let volume = prim_fit.volume();
        let weighted_volume = prim_fit.weighted_volume();
        let neighbors: HashSet<u32> = adj.neighbors[fi].iter().copied().collect();
        prims.push(Primitive {
            alive: true,
            version: 0,
            q,
            prim: prim_fit,
            volume,
            weighted_volume,
            face_count: 1,
            vertex_indices: vidx.to_vec(),
            neighbors,
        });
    }
    // Cyclic linked list of faces per primitive (paper §3.4). Each face's
    // `next` initially points to itself (singleton list).
    let mut face_next: Vec<u32> = (0..nf as u32).collect();
    // DSU mapping face_idx → primitive root face_idx. Used for fast stale
    // detection on PQ pop.
    let mut dsu = Dsu::new(nf);


    // Pre-compute sharp edges if orientation refinement is on. ~30° dihedral
    // threshold catches creases on architecture/CAD-style meshes without
    // false-flagging slightly-curved smooth surfaces.
    let sharp_edges: Option<SharpEdges> = if opts.refine_orient {
        // Build a shell-only face mask if shell-aware is on, so creases
        // between two interior faces don't pollute the sharp-edge axes.
        let face_shell_mask: Option<Vec<bool>> = face_exposure.as_ref().map(|exp| {
            exp.iter().map(|&e| e > 0.05).collect()
        });
        let mask_ref = face_shell_mask.as_deref();
        // Per-mesh adaptive threshold: 95th-percentile dihedral, clamped
        // to [30°, 60°]. Avoids over-flagging small ridges on organic
        // meshes (rocks, terrain) where most dihedrals are tiny.
        let thresh = crate::mesh::adaptive_sharp_threshold(mesh);
        eprintln!(
            "sharp-edge threshold: {:.1}° (adaptive 95th-percentile dihedral)",
            thresh.to_degrees()
        );
        Some(crate::mesh::build_sharp_edges(mesh, thresh, mask_ref))
    } else {
        None
    };

    // Build the initial set of (f, n) pairs once, then evaluate merge costs
    // for them in parallel.
    let mut initial_pairs: Vec<(u32, u32)> = Vec::new();
    for f in 0..nf {
        for &n in &adj.neighbors[f] {
            if (n as usize) <= f {
                continue;
            }
            initial_pairs.push((f as u32, n));
        }
    }
    let initial_entries: Vec<PqEntry> = initial_pairs
        .par_iter()
        .filter_map(|&(f, n)| {
            let pa = &prims[f as usize];
            let pb = &prims[n as usize];
            let (_q, prim_fit, vol, wvol, _vidx) =
                merge_pair(pa, pb, &mesh.verts, opts.enabled);
            let mut cost = if opts.weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            if opts.reject_pancakes && prim_fit.is_pancake() {
                cost = cost.abs() * 1000.0 + 1e6;
            }
            if cost > opts.volume_threshold {
                return None;
            }
            Some(PqEntry {
                cost,
                a: f,
                b: n,
                va: pa.version,
                vb: pb.version,
                prim: prim_fit,
                volume: vol,
                weighted_volume: wvol,
            })
        })
        .collect();
    let mut pq: BinaryHeap<PqEntry> = BinaryHeap::with_capacity(initial_entries.len() * 2);
    for e in initial_entries {
        pq.push(e);
    }

    let mut alive_count = nf;
    let mut merges_done = 0usize;
    let mut merges_skipped_stale = 0usize;
    let mut merges_rejected_empty = 0usize;
    let mut all_pairs_used = false;
    // Memoize rejected pairs so we don't re-evaluate the BVH for the same
    // pair every time a stale/cheaper entry of theirs gets popped. Key
    // includes versions so a fresh post-merge primitive re-checks.
    let mut rejected_pairs: HashSet<(u32, u64, u32, u64)> = HashSet::new();

    while alive_count > opts.target_n {
        let entry = match pq.pop() {
            Some(e) => e,
            None => {
                if all_pairs_used {
                    break;
                }
                // The all-pairs fallback is fundamentally at odds with
                // empty-space preservation: it generates O(N²) candidates
                // that wrap disjoint components into giant primitives, and
                // the empty-space check rejects almost all of them. With
                // many live components that's >100M rejections worth of
                // BVH queries. Accept the current count instead.
                let pushed = if let Some((r_frac, k, angle_rad)) = opts.proximity {
                    // Proximity is spatially bounded (O(N·k) candidates,
                    // not O(N²)). Safe to run even when --empty-space is
                    // on — the empty-space hard reject still fires on
                    // every pop and rejects any candidate that bridges
                    // open volume. We only skip the brute all-pairs
                    // fallback in empty-space mode.
                    let max_dist = r_frac * mesh_diag;
                    let p = push_proximity_pairs(
                        &prims,
                        &mesh.verts,
                        &mut pq,
                        opts.volume_threshold,
                        opts.enabled,
                        max_dist,
                        k,
                        angle_rad,
                        opts.weighted_cost,
                        opts.reject_pancakes,
                    );
                    eprintln!(
                        "topology PQ drained at {} primitives; pushed {} proximity candidates (k={}, r={:.3}, angle<={:.0}°)",
                        alive_count, p, k, max_dist, angle_rad.to_degrees(),
                    );
                    p
                } else if opts.empty_space.is_some() {
                    // Brute all-pairs would generate O(N²) candidates
                    // that the empty-space check rejects almost all of.
                    eprintln!(
                        "topology PQ drained at {} primitives; skipping brute all-pairs (--empty-space active, no --proximity)",
                        alive_count
                    );
                    break;
                } else {
                    let p = push_all_pairs(
                        &prims,
                        &mesh.verts,
                        &mut pq,
                        opts.volume_threshold,
                        opts.enabled,
                        opts.weighted_cost,
                        opts.reject_pancakes,
                    );
                    eprintln!(
                        "topology PQ drained at {} primitives; pushed {} all-pairs candidates",
                        alive_count, p
                    );
                    p
                };
                all_pairs_used = true;
                if pushed == 0 {
                    break;
                }
                continue;
            }
        };
        let a = entry.a as usize;
        let b = entry.b as usize;
        // Stale check: use the cheap inline `alive` flag — equivalent to
        // dsu.is_root() since we maintain both, and avoids an extra pointer
        // chase into the DSU `parent` array on the hot pop path.
        if !prims[a].alive || !prims[b].alive {
            merges_skipped_stale += 1;
            continue;
        }
        if prims[a].version != entry.va || prims[b].version != entry.vb {
            merges_skipped_stale += 1;
            continue;
        }

        // Empty-space preservation (paper §3.3-style hard reject, but
        // measured against the input mesh rather than against an excess
        // volume threshold).
        if let (Some((max_frac, dist_thresh)), Some(bvh_ref)) = (opts.empty_space, &bvh) {
            let key = if (a as u32) < (b as u32) {
                (a as u32, entry.va, b as u32, entry.vb)
            } else {
                (b as u32, entry.vb, a as u32, entry.va)
            };
            if rejected_pairs.contains(&key) {
                merges_rejected_empty += 1;
                continue;
            }
            let frac = empty_space_fraction(&entry.prim, mesh, bvh_ref, dist_thresh);
            if frac > max_frac {
                rejected_pairs.insert(key);
                merges_rejected_empty += 1;
                continue;
            }
        }

        // Use the cached fit from push time. Q is just the sum of the two
        // primitives' quadrics by linearity.
        let new_q = prims[a].q + prims[b].q;
        let new_vidx = merge_sorted_unique(&prims[a].vertex_indices, &prims[b].vertex_indices);

        // Optional post-merge orientation refinement. The Q-eigenbasis
        // orientation is what the cached prim was fit against, so the refit
        // can only equal-or-improve the cached primitive — no need to
        // re-push the candidate to the priority queue.
        let (new_prim, new_vol, new_wvol) = if opts.refine_orient {
            // Containment fits use every subsumed vertex (paper guarantee).
            let pts: Vec<Point3<f32>> = new_vidx
                .iter()
                .map(|&i| mesh.verts[i as usize])
                .collect();
            // Orientation fits prefer shell vertices when shell-awareness
            // is on. Fall back to the full set if too few shell verts.
            let pts_orient: Vec<Point3<f32>> = match &shell_vertex_mask {
                Some(mask) => {
                    let v: Vec<Point3<f32>> = new_vidx
                        .iter()
                        .filter(|&&i| mask[i as usize])
                        .map(|&i| mesh.verts[i as usize])
                        .collect();
                    if v.len() < 4 {
                        pts.clone()
                    } else {
                        v
                    }
                }
                None => pts.clone(),
            };

            // Selection criterion: when quality_beta == 0, just minimise
            // weighted volume (paper-faithful). When > 0, minimise
            // `weighted_volume * (1 + beta * h/diag)` where h is a cheap
            // sampled max-distance from the primitive's surface to the
            // input mesh via BVH. Sphere is included when quality is on.
            let use_quality = opts.quality_beta > 0.0 && bvh.is_some();
            let bvh_ref = bvh.as_ref();

            let score = |p: &Prim| -> f32 {
                if use_quality {
                    let h = local_hausdorff(p, bvh_ref.unwrap(), mesh);
                    p.weighted_volume() * (1.0 + opts.quality_beta * h / mesh_diag)
                } else {
                    p.weighted_volume()
                }
            };

            let mut best_prim = entry.prim;
            let mut best_score = score(&best_prim);

            let mut try_axes = |axes: [Vector3<f32>; 3], best: &mut Prim, best_score: &mut f32| {
                if use_quality {
                    // try every primitive type for this orientation, score
                    // each individually.
                    let mask = if opts.enabled.obb || opts.enabled.sphere {
                        let mut m = opts.enabled;
                        m.sphere = true; // unconditional sphere in quality mode
                        m
                    } else {
                        opts.enabled
                    };
                    for cand in prim::fit_all(axes, &pts, mask) {
                        let s = score(&cand);
                        if s < *best_score {
                            *best_score = s;
                            *best = cand;
                        }
                    }
                } else {
                    let cand = prim::fit_best(axes, &pts, opts.enabled);
                    let s = cand.weighted_volume();
                    if s < *best_score {
                        *best_score = s;
                        *best = cand;
                    }
                }
            };

            // Candidate 1: vertex PCA (shell-only when shell-aware is on).
            try_axes(pca_axes(&pts_orient), &mut best_prim, &mut best_score);
            // Candidate 2: tangent-plane PCA (shell-only when on).
            try_axes(tangent_plane_pca_axes(new_q, &pts_orient), &mut best_prim, &mut best_score);
            // Candidate 3: sharp-edge directions.
            if let Some(sharp_ref) = &sharp_edges {
                let face_iter = walk_faces(a as u32, prims[a].face_count, &face_next)
                    .chain(walk_faces(b as u32, prims[b].face_count, &face_next));
                if let Some(axes) = sharp_edge_axes(sharp_ref, face_iter) {
                    try_axes(axes, &mut best_prim, &mut best_score);
                }
            }

            let v = best_prim.volume();
            let w = best_prim.weighted_volume();
            (best_prim, v, w)
        } else {
            (entry.prim, entry.volume, entry.weighted_volume)
        };
        // O(1) face-list splice (paper §3.4): swap `next` pointers at the
        // two roots to merge the cyclic linked lists into one.
        let tmp = face_next[a];
        face_next[a] = face_next[b];
        face_next[b] = tmp;
        let new_face_count = prims[a].face_count + prims[b].face_count;
        dsu.link(a as u32, b as u32);

        let mut new_neighbors: HashSet<u32> = prims[a]
            .neighbors
            .iter()
            .chain(prims[b].neighbors.iter())
            .copied()
            .collect();
        new_neighbors.remove(&(a as u32));
        new_neighbors.remove(&(b as u32));

        let neighbor_list: Vec<u32> = new_neighbors.iter().copied().collect();
        for &n in &neighbor_list {
            let nrefs = &mut prims[n as usize].neighbors;
            nrefs.remove(&(b as u32));
            nrefs.insert(a as u32);
        }

        prims[a].q = new_q;
        prims[a].prim = new_prim;
        prims[a].volume = new_vol;
        prims[a].weighted_volume = new_wvol;
        prims[a].face_count = new_face_count;
        prims[a].vertex_indices = new_vidx;
        prims[a].neighbors = new_neighbors;
        prims[a].version += 1;

        prims[b].alive = false;
        prims[b].neighbors.clear();
        // free the now-stale vertex/face data on the loser slot to reclaim mem
        prims[b].vertex_indices = Vec::new();
        prims[b].face_count = 0;

        alive_count -= 1;
        merges_done += 1;

        let candidates: Vec<u32> = if all_pairs_used {
            // After the topology drain we operate on logical neighbours.
            // In proximity mode we want only spatially-close, similarly-
            // oriented prims; in all-pairs mode every live primitive.
            if let Some((r_frac, k, angle_rad)) = opts.proximity {
                let live = live_indices(&prims);
                let summary = live_summary(&prims, &live);
                // find this primitive's index in `live`
                let me_in_live = live.iter().position(|&x| x as usize == a);
                let max_dist = r_frac * mesh_diag;
                let cos_min = angle_rad.cos();
                if let Some(i) = me_in_live {
                    let mut dists: Vec<(usize, f32)> = (0..live.len())
                        .filter(|&j| j != i)
                        .map(|j| (j, aabb_to_aabb_dist(&summary.aabbs[i], &summary.aabbs[j])))
                        .collect();
                    dists.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(Ordering::Equal));
                    dists
                        .into_iter()
                        .take(k)
                        .filter_map(|(j, d)| {
                            if d > max_dist {
                                return None;
                            }
                            if summary.normals[i].dot(&summary.normals[j]).abs() < cos_min {
                                return None;
                            }
                            Some(live[j])
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                prims
                    .iter()
                    .enumerate()
                    .filter(|(i, p)| p.alive && *i != a)
                    .map(|(i, _)| i as u32)
                    .collect()
            }
        } else {
            prims[a].neighbors.iter().copied().collect()
        };
        // Sequential: post-merge candidate count is small (typically 3–6),
        // and par_iter overhead dominates for that workload.
        let pa = &prims[a];
        for &n in &candidates {
            let pn = &prims[n as usize];
            let (_q, prim_fit, vol, wvol, _vidx) =
                merge_pair(pa, pn, &mesh.verts, opts.enabled);
            let mut cost = if opts.weighted_cost {
                wvol - (pa.weighted_volume + pn.weighted_volume)
            } else {
                vol - (pa.volume + pn.volume)
            };
            if opts.reject_pancakes && prim_fit.is_pancake() {
                cost = cost.abs() * 1000.0 + 1e6;
            }
            if cost > opts.volume_threshold {
                continue;
            }
            pq.push(PqEntry {
                cost,
                a: a as u32,
                b: n,
                va: pa.version,
                vb: pn.version,
                prim: prim_fit,
                volume: vol,
                weighted_volume: wvol,
            });
        }
    }

    // Lloyd-style face rebalance: try moving boundary faces between
    // primitives to escape the greedy local minimum. Run BEFORE the
    // redundant-cull pass so the cull operates on the rebalanced state.
    let mut rebalance_moves = 0usize;
    if let Some(max_passes) = opts.rebalance {
        if let Some(bvh_ref) = &bvh {
            let t = std::time::Instant::now();
            rebalance_moves = rebalance_faces(
                &mut prims,
                mesh,
                adj,
                &mut dsu,
                &mut face_next,
                bvh_ref,
                opts.enabled,
                max_passes,
            );
            eprintln!(
                "rebalance: {} total face moves in {:.1} ms",
                rebalance_moves,
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }
    }

    let mut redundant_culled = 0usize;
    if opts.cull_redundant {
        redundant_culled = cull_redundant(&mut prims, &mesh.verts);
    }

    DecompResult {
        primitives: prims,
        merges_done,
        merges_skipped_stale,
        merges_rejected_empty,
        all_pairs_used,
        redundant_culled,
        rebalance_moves,
    }
}

/// Compact per-primitive cache used by the Lloyd rebalance pass. Storing
/// the fitted primitive + Hausdorff score avoids re-running fit_best on
/// the same group when only its score is needed.
struct RebalanceState {
    prim: Prim,
    q: Matrix3<f32>,
    volume: f32,
    weighted_volume: f32,
    /// Sampled local Hausdorff to the input mesh (cheap proxy for fit
    /// quality, same metric the cull and refit-quality paths use).
    hausdorff: f32,
    /// Sorted-unique vertex indices subsumed by the primitive's faces.
    vertex_indices: Vec<u32>,
}

fn refit_from_faces(
    faces: &[u32],
    mesh: &Mesh,
    enabled: PrimMask,
    bvh: &Bvh,
) -> RebalanceState {
    // Sum per-face quadrics, gather subsumed vertices.
    let mut q = Matrix3::zeros();
    let mut vidx_set: HashSet<u32> = HashSet::new();
    for &fi in faces {
        let t = mesh.tris[fi as usize];
        let p0 = mesh.verts[t[0] as usize];
        let p1 = mesh.verts[t[1] as usize];
        let p2 = mesh.verts[t[2] as usize];
        q += face_quadric(p0, p1, p2);
        vidx_set.insert(t[0]);
        vidx_set.insert(t[1]);
        vidx_set.insert(t[2]);
    }
    let mut vidx: Vec<u32> = vidx_set.into_iter().collect();
    vidx.sort();
    let pts: Vec<Point3<f32>> = vidx.iter().map(|&i| mesh.verts[i as usize]).collect();
    let axes = axes_from_q(q);
    let prim_fit = prim::fit_best(axes, &pts, enabled);
    let h = local_hausdorff(&prim_fit, bvh, mesh);
    RebalanceState {
        volume: prim_fit.volume(),
        weighted_volume: prim_fit.weighted_volume(),
        prim: prim_fit,
        q,
        hausdorff: h,
        vertex_indices: vidx,
    }
}

/// Lloyd-style face migration. Starts from the greedy merge result and
/// iteratively moves boundary faces between adjacent primitives; accepts
/// any move that reduces the summed local Hausdorff of the two primitives
/// it touches. Keeps the primitive count fixed (a face can only move if
/// its source primitive has more than one face). Pure local rebalancing —
/// breaks greedy local minima without changing N.
fn rebalance_faces(
    prims: &mut Vec<Primitive>,
    mesh: &Mesh,
    adj: &Adjacency,
    dsu: &mut Dsu,
    face_next: &mut Vec<u32>,
    bvh: &Bvh,
    enabled: PrimMask,
    max_passes: usize,
) -> usize {
    // Combined cost so we don't accept moves that improve Hausdorff at
    // catastrophic volume cost. Same shape as --quality.
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let beta = 5.0f32;
    let score_state = |s: &RebalanceState| -> f32 {
        s.weighted_volume * (1.0 + beta * s.hausdorff / mesh_diag)
    };
    let nf = mesh.tris.len();

    // Compact existing primitives to 0..N. dsu.find(face_idx) gives the
    // primitive's representative root face_idx; map those to dense ids.
    let mut root_to_id: HashMap<u32, u32> = HashMap::new();
    let mut face_assignment: Vec<u32> = Vec::with_capacity(nf);
    for f in 0..nf {
        let root = dsu.find(f as u32);
        let id = match root_to_id.get(&root) {
            Some(&id) => id,
            None => {
                let id = root_to_id.len() as u32;
                root_to_id.insert(root, id);
                id
            }
        };
        face_assignment.push(id);
    }
    let n_prims = root_to_id.len();
    if n_prims < 2 {
        return 0;
    }
    let mut prim_faces: Vec<Vec<u32>> = vec![Vec::new(); n_prims];
    for (fi, &pid) in face_assignment.iter().enumerate() {
        prim_faces[pid as usize].push(fi as u32);
    }

    // Refit from face groups so the rebalance state matches what's about
    // to be tracked (the greedy pass had its own per-primitive state, but
    // for migration we want a single source of truth).
    let mut state: Vec<RebalanceState> = (0..n_prims)
        .map(|pid| refit_from_faces(&prim_faces[pid], mesh, enabled, bvh))
        .collect();

    let mut total_moves = 0usize;
    for pass in 0..max_passes {
        let mut moves = 0usize;
        for f in 0..nf {
            let current_p = face_assignment[f] as usize;
            // Don't empty a primitive — that would drop the count.
            if prim_faces[current_p].len() <= 1 {
                continue;
            }
            // Candidate primitives: those held by topologically-adjacent
            // faces, minus our current primitive.
            let mut candidates: Vec<u32> = adj.neighbors[f]
                .iter()
                .map(|&nf_idx| face_assignment[nf_idx as usize])
                .filter(|&p| p as usize != current_p)
                .collect();
            candidates.sort_unstable();
            candidates.dedup();
            if candidates.is_empty() {
                continue;
            }

            let old_score = score_state(&state[current_p]);
            let mut best_pid: Option<u32> = None;
            let mut best_new_a: Option<RebalanceState> = None;
            let mut best_new_b: Option<RebalanceState> = None;
            let mut best_delta = 0.0f32;

            for cand in candidates {
                let cand_p = cand as usize;
                let cand_score = score_state(&state[cand_p]);
                let old_combined = old_score + cand_score;

                let new_a_faces: Vec<u32> = prim_faces[current_p]
                    .iter()
                    .filter(|&&x| x != f as u32)
                    .copied()
                    .collect();
                let mut new_b_faces = prim_faces[cand_p].clone();
                new_b_faces.push(f as u32);

                let new_a = refit_from_faces(&new_a_faces, mesh, enabled, bvh);
                let new_b = refit_from_faces(&new_b_faces, mesh, enabled, bvh);
                let new_combined = score_state(&new_a) + score_state(&new_b);
                let delta = new_combined - old_combined;
                if delta < best_delta {
                    best_delta = delta;
                    best_pid = Some(cand);
                    best_new_a = Some(new_a);
                    best_new_b = Some(new_b);
                }
            }

            if let Some(target) = best_pid {
                let target_p = target as usize;
                face_assignment[f] = target;
                prim_faces[current_p].retain(|&x| x != f as u32);
                prim_faces[target_p].push(f as u32);
                state[current_p] = best_new_a.unwrap();
                state[target_p] = best_new_b.unwrap();
                moves += 1;
            }
        }
        eprintln!(
            "rebalance pass {}: {} face moves",
            pass + 1,
            moves
        );
        total_moves += moves;
        if moves == 0 {
            break;
        }
    }

    // Rebuild dsu + face_next + prims from the final face assignment.
    *dsu = Dsu::new(nf);
    for pid in 0..n_prims {
        let faces = &prim_faces[pid];
        if faces.is_empty() {
            continue;
        }
        // Splice all faces in this primitive into one cyclic linked list,
        // and union them in the DSU under the first face as the root.
        let root = faces[0];
        for &f in &faces[1..] {
            dsu.link(root, f);
        }
        // Build the cyclic list: face_next[f_i] = f_{i+1}, last → root
        for i in 0..faces.len() {
            let next = if i + 1 < faces.len() {
                faces[i + 1]
            } else {
                faces[0]
            };
            face_next[faces[i] as usize] = next;
        }
        // Mark the data slot at `root` as the live primitive.
        let s = &state[pid];
        prims[root as usize] = Primitive {
            alive: true,
            version: prims[root as usize].version + 1,
            q: s.q,
            prim: s.prim.clone(),
            volume: s.volume,
            weighted_volume: s.weighted_volume,
            face_count: faces.len() as u32,
            vertex_indices: s.vertex_indices.clone(),
            // Neighbors will be re-derived: find unique adjacent primitive
            // roots from face adjacency.
            neighbors: HashSet::new(),
        };
    }
    // Mark everything else dead.
    for fi in 0..nf {
        if dsu.find(fi as u32) != fi as u32 {
            prims[fi].alive = false;
            prims[fi].vertex_indices = Vec::new();
            prims[fi].face_count = 0;
            prims[fi].neighbors.clear();
        }
    }
    // Rebuild neighbour sets at primitive roots.
    for f in 0..nf {
        let p = dsu.find(f as u32);
        for &nf_idx in &adj.neighbors[f] {
            let q = dsu.find(nf_idx);
            if p != q {
                prims[p as usize].neighbors.insert(q);
            }
        }
    }

    total_moves
}

/// Drop primitive A if every vertex it subsumes is also enclosed by some
/// other primitive B that A *shares at least one mesh vertex with* (paper
/// §3.4 "Removing Redundant Primitives", with our shared-vertex constraint
/// to avoid pathological global culling after the all-pairs phase wraps
/// disjoint components in giant primitives).
fn cull_redundant(prims: &mut [Primitive], mesh_verts: &[Point3<f32>]) -> usize {
    let live: Vec<usize> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive)
        .map(|(i, _)| i)
        .collect();

    let mut diag = 0.0f32;
    for &i in &live {
        for &vi in &prims[i].vertex_indices {
            let v = mesh_verts[vi as usize].coords;
            let l = v.norm();
            if l > diag {
                diag = l;
            }
        }
    }
    let tol = (diag.max(1.0)) * 1e-4;

    let mut to_drop: Vec<usize> = Vec::new();
    for &a in &live {
        if to_drop.contains(&a) {
            continue;
        }
        for &b in &live {
            if a == b || to_drop.contains(&b) {
                continue;
            }
            // require shared vertex — this means A and B were either merged
            // from adjacent topology at some point, or share a seam.
            // Without this, an all-pairs "monster" primitive (made of
            // multiple disjoint components) can engulf every other live
            // primitive, deleting the tight per-component fits.
            if !shared_vertex(&prims[a].vertex_indices, &prims[b].vertex_indices) {
                continue;
            }
            let bp = &prims[b].prim;
            let all_in = prims[a]
                .vertex_indices
                .iter()
                .all(|&vi| bp.contains(mesh_verts[vi as usize], tol));
            if all_in && prims[a].volume <= prims[b].volume {
                to_drop.push(a);
                break;
            }
        }
    }
    for i in &to_drop {
        prims[*i].alive = false;
    }
    to_drop.len()
}

fn shared_vertex(a: &[u32], b: &[u32]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => return true,
        }
    }
    false
}
