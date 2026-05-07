use crate::bvh::Bvh;
use crate::dsu::Dsu;
use crate::mesh::{Adjacency, Mesh, SharpEdges};
use crate::prim::{self, Prim, PrimMask};
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

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

fn push_all_pairs(
    prims: &[Primitive],
    mesh_verts: &[Point3<f32>],
    pq: &mut BinaryHeap<PqEntry>,
    volume_threshold: f32,
    enabled: PrimMask,
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
            let cost = vol - (pa.volume + pb.volume);
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
    let mut prims: Vec<Primitive> = Vec::with_capacity(nf);
    for (fi, tri) in mesh.tris.iter().enumerate() {
        let p0 = mesh.verts[tri[0] as usize];
        let p1 = mesh.verts[tri[1] as usize];
        let p2 = mesh.verts[tri[2] as usize];
        let q = face_quadric(p0, p1, p2);
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

    let bvh: Option<Bvh> = if opts.empty_space.is_some() {
        Some(Bvh::build(&mesh.verts, &mesh.tris))
    } else {
        None
    };

    // Pre-compute sharp edges if orientation refinement is on. ~30° dihedral
    // threshold catches creases on architecture/CAD-style meshes without
    // false-flagging slightly-curved smooth surfaces.
    let sharp_edges: Option<SharpEdges> = if opts.refine_orient {
        Some(crate::mesh::build_sharp_edges(mesh, std::f32::consts::FRAC_PI_6))
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
            let cost = vol - (pa.volume + pb.volume);
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
                if opts.empty_space.is_some() {
                    eprintln!(
                        "topology PQ drained at {} primitives; skipping all-pairs (--empty-space active)",
                        alive_count
                    );
                    break;
                }
                let pushed =
                    push_all_pairs(&prims, &mesh.verts, &mut pq, opts.volume_threshold, opts.enabled);
                all_pairs_used = true;
                eprintln!(
                    "topology PQ drained at {} primitives; pushed {} all-pairs candidates",
                    alive_count, pushed
                );
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
            let pts: Vec<Point3<f32>> = new_vidx
                .iter()
                .map(|&i| mesh.verts[i as usize])
                .collect();

            let mut best_prim = entry.prim;
            let mut best_wvol = entry.weighted_volume;
            let mut try_axes = |axes: [Vector3<f32>; 3], best: &mut Prim, best_wvol: &mut f32| {
                let cand = prim::fit_best(axes, &pts, opts.enabled);
                if cand.weighted_volume() < *best_wvol {
                    *best_wvol = cand.weighted_volume();
                    *best = cand;
                }
            };

            // Candidate 1: vertex PCA (geometric extent).
            try_axes(pca_axes(&pts), &mut best_prim, &mut best_wvol);
            // Candidate 2: tangent-plane PCA (auto-tangent-weight fix for
            // near-coplanar regions).
            try_axes(
                tangent_plane_pca_axes(new_q, &pts),
                &mut best_prim,
                &mut best_wvol,
            );
            // Candidate 3: sharp-edge directions (feature-aligned). Walk the
            // face linked lists of both primitives BEFORE the splice below.
            if let Some(sharp_ref) = &sharp_edges {
                let face_iter = walk_faces(a as u32, prims[a].face_count, &face_next)
                    .chain(walk_faces(b as u32, prims[b].face_count, &face_next));
                if let Some(axes) = sharp_edge_axes(sharp_ref, face_iter) {
                    try_axes(axes, &mut best_prim, &mut best_wvol);
                }
            }

            let v = best_prim.volume();
            (best_prim, v, best_wvol)
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
            prims
                .iter()
                .enumerate()
                .filter(|(i, p)| p.alive && *i != a)
                .map(|(i, _)| i as u32)
                .collect()
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
            let cost = vol - (pa.volume + pn.volume);
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
    }
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
