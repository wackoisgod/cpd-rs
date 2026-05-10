use crate::bvh::Bvh;
use crate::dsu::Dsu;
use crate::mesh::{Adjacency, Mesh, SharpEdges};
use crate::prim::{self, Prim, PrimMask};
use nalgebra::{Matrix3, Point3, SymmetricEigen, Vector3};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

pub fn face_quadric(
    p0: Point3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
    tangent_eps: f32,
) -> Matrix3<f32> {
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

    // The tangent term is the paper's per-mesh-decided knob (§3.4). At
    // ε=0 only the normal contributes — Q is strictly rank-1 for a
    // single face. The rank-1 case hands all in-plane axis decisions to
    // our PCA / tangent-plane / sharp-edge fallbacks. At ε>0 we
    // pre-bias the in-plane axes to whatever Gram-Schmidt of (n, t)
    // produces, which on big flat regions can rotate the OBB by an
    // arbitrary amount and produce a rotated slab that drifts past the
    // mesh corners. Caller decides via DecompOpts.tangent_eps.
    if tangent_eps <= 0.0 {
        return area * (n * n.transpose());
    }

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

    area * (n * n.transpose() + tangent_eps * t * t.transpose())
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
    let v0 = dec.eigenvectors.column(idx[0]).into_owned();
    let v1 = dec.eigenvectors.column(idx[1]).into_owned();
    // axes[0] from eigen, defended against NaN/zero
    let a0 = if v0.norm_squared() > 1e-20 {
        v0.normalize()
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    // Gram-Schmidt for axes[1] in-plane. If eigen gave a vector parallel
    // to a0 (happens for repeated eigenvalues / near-zero Q), the
    // subtraction goes to zero and normalize would return NaN; fall back
    // to a known-perpendicular helper. Same guard pca_axes uses.
    let proj = v1 - a0 * a0.dot(&v1);
    let a1 = if proj.norm_squared() > 1e-20 {
        proj.normalize()
    } else {
        let helper = if a0.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        (helper - a0 * a0.dot(&helper)).normalize()
    };
    let a2 = a0.cross(&a1).normalize();
    [a0, a1, a2]
}

/// Build an orthonormal basis whose first axis is `axis0`, with the other
/// two derived from the principal direction of `axis_seed` projected into
/// the plane perpendicular to `axis0`. Returns axes ordered as
/// `[axis0, in_plane_primary, in_plane_secondary]`.
fn orthonormal_basis_from_seed(axis0: Vector3<f32>, axis_seed: Vector3<f32>) -> [Vector3<f32>; 3] {
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
/// Denser variant of local_hausdorff for the split path. Samples every
/// vertex + every edge midpoint of the tessellation, then 256
/// area-weighted random barycentric points across the primitive's
/// surface. Designed to match the metrics module's sampling closely
/// enough that an "improvement" under this metric also shows up as an
/// improvement under the 10k-sample evaluation: the deterministic 24-
/// sample local_hausdorff was missing face-interior drift on thin OBBs
/// (rectangular bounding box sitting on a non-rectangular planar
/// region — worst point can be away from any vertex), causing splits
/// to be accepted that made the global Hausdorff *worse*.
#[derive(Clone, Copy)]
struct HausdorffWitness {
    dist: f32,
    sample: Point3<f32>,
    nearest: Point3<f32>,
    normal: Vector3<f32>,
}

#[derive(Clone, Copy)]
enum DistanceMode {
    Absolute,
    Outside,
}

fn signed_distance_score(signed: f32, mode: DistanceMode) -> f32 {
    match mode {
        DistanceMode::Absolute => signed.abs(),
        DistanceMode::Outside => signed.max(0.0),
    }
}

fn local_hausdorff_dense(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
    local_hausdorff_dense_witness(p, bvh, mesh)
        .map(|w| w.dist)
        .unwrap_or(0.0)
}

fn local_outside_dense(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
    local_distance_dense_witness(p, bvh, mesh, DistanceMode::Outside)
        .map(|w| w.dist)
        .unwrap_or(0.0)
}

fn local_hausdorff_dense_witness(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> Option<HausdorffWitness> {
    local_distance_dense_witness(p, bvh, mesh, DistanceMode::Absolute)
}

fn local_distance_dense_witness(
    p: &Prim,
    bvh: &Bvh,
    mesh: &Mesh,
    mode: DistanceMode,
) -> Option<HausdorffWitness> {
    const N_RANDOM: usize = 256;
    let (verts, tris) = prim::tessellate(p);
    if verts.is_empty() || tris.is_empty() {
        return None;
    }
    let mut best: Option<HausdorffWitness> = None;
    let mut probe = |q: Point3<f32>| {
        let (pt, n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, q);
        let d = signed_distance_score(signed, mode);
        if d.is_finite() && best.map(|w| d > w.dist).unwrap_or(true) {
            best = Some(HausdorffWitness {
                dist: d,
                sample: q,
                nearest: pt,
                normal: n,
            });
        }
    };
    // Every vertex of the tessellation.
    for v in &verts {
        probe(Point3::new(v[0], v[1], v[2]));
    }
    // Every edge midpoint.
    for t in &tris {
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        for &(p0, p1) in &[(a, b), (b, c), (c, a)] {
            let q = Point3::new(
                0.5 * (p0[0] + p1[0]),
                0.5 * (p0[1] + p1[1]),
                0.5 * (p0[2] + p1[2]),
            );
            probe(q);
        }
    }
    // Area-weighted random barycentric sampling — same scheme the
    // metrics module uses, lighter sample count.
    let mut tri_cum: Vec<f32> = Vec::with_capacity(tris.len());
    let mut acc = 0.0f32;
    for t in tris.iter() {
        let a = Vector3::new(
            verts[t[0] as usize][0],
            verts[t[0] as usize][1],
            verts[t[0] as usize][2],
        );
        let b = Vector3::new(
            verts[t[1] as usize][0],
            verts[t[1] as usize][1],
            verts[t[1] as usize][2],
        );
        let c = Vector3::new(
            verts[t[2] as usize][0],
            verts[t[2] as usize][1],
            verts[t[2] as usize][2],
        );
        acc += 0.5 * (b - a).cross(&(c - a)).norm();
        tri_cum.push(acc);
    }
    if acc <= 0.0 {
        return best;
    }
    // Tiny LCG seeded by the primitive's first vertex so results are
    // deterministic per-primitive while different primitives use
    // different sample sets.
    let mut state: u64 = 0xCAFEF00D
        ^ (verts[0][0].to_bits() as u64).wrapping_mul(2654435761)
        ^ (verts[0][1].to_bits() as u64).wrapping_mul(40503)
        ^ (verts[0][2].to_bits() as u64).wrapping_mul(67043);
    let mut step = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 32) as u32) as f32 * (1.0 / 4294967296.0)
    };
    for _ in 0..N_RANDOM {
        let r = step() * acc;
        // Lower bound search.
        let mut lo = 0usize;
        let mut hi = tri_cum.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if tri_cum[mid] >= r {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let ti = lo.min(tri_cum.len() - 1);
        let mut u = step();
        let mut v = step();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        let t = &tris[ti];
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        let q = Point3::new(
            a[0] * w + b[0] * u + c[0] * v,
            a[1] * w + b[1] * u + c[1] * v,
            a[2] * w + b[2] * u + c[2] * v,
        );
        probe(q);
    }
    best
}

fn local_hausdorff_repair(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
    local_hausdorff_repair_witness(p, bvh, mesh)
        .map(|w| w.dist)
        .unwrap_or(0.0)
}

fn local_outside_repair(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
    local_distance_repair_witness(p, bvh, mesh, DistanceMode::Outside)
        .map(|w| w.dist)
        .unwrap_or(0.0)
}

fn local_hausdorff_repair_witness(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> Option<HausdorffWitness> {
    local_distance_repair_witness(p, bvh, mesh, DistanceMode::Absolute)
}

fn local_distance_repair_witness(
    p: &Prim,
    bvh: &Bvh,
    mesh: &Mesh,
    mode: DistanceMode,
) -> Option<HausdorffWitness> {
    const GRID: usize = 12;
    let (verts, tris) = prim::tessellate(p);
    if verts.is_empty() || tris.is_empty() {
        return None;
    }

    let mut best: Option<HausdorffWitness> = None;
    let mut probe = |q: Point3<f32>| {
        let (pt, n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, q);
        let d = signed_distance_score(signed, mode);
        if d.is_finite() && best.map(|w| d > w.dist).unwrap_or(true) {
            best = Some(HausdorffWitness {
                dist: d,
                sample: q,
                nearest: pt,
                normal: n,
            });
        }
    };

    for t in &tris {
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        for iu in 0..=GRID {
            for iv in 0..=(GRID - iu) {
                let u = iu as f32 / GRID as f32;
                let v = iv as f32 / GRID as f32;
                let w = 1.0 - u - v;
                let q = Point3::new(
                    a[0] * w + b[0] * u + c[0] * v,
                    a[1] * w + b[1] * u + c[1] * v,
                    a[2] * w + b[2] * u + c[2] * v,
                );
                probe(q);
            }
        }
    }

    best
}

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

fn local_outside(p: &Prim, bvh: &Bvh, mesh: &Mesh) -> f32 {
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
            let d = signed_distance_score(signed, DistanceMode::Outside);
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
            let d = signed_distance_score(signed, DistanceMode::Outside);
            if d > max_d {
                max_d = d;
            }
            count += 1;
            ti += stride;
        }
    }
    max_d
}

pub fn single_triangle_dense_hausdorff(
    p0: Point3<f32>,
    p1: Point3<f32>,
    p2: Point3<f32>,
    bvh: &Bvh,
    reference_mesh: &Mesh,
    enabled: PrimMask,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
) -> f32 {
    let q = face_quadric(p0, p1, p2, tangent_eps);
    let axes = match axis_override {
        Some(a) => *a,
        None => axes_from_q(q),
    };
    let pts = [p0, p1, p2];
    let prim_fit = prim::fit_best(axes, &pts, enabled);
    local_hausdorff_dense(&prim_fit, bvh, reference_mesh)
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
    /// Subset of vertices used for collision fitting when decorative/detail
    /// faces are intentionally allowed to stop expanding the collider. Falls
    /// back to `vertex_indices` if too small.
    pub fit_vertex_indices: Vec<u32>,
    /// Synthetic support points used by collision support-plane fitting.
    /// Detail faces near a broad wall/roof plane can contribute projected
    /// points here instead of pulling the primitive out to their real depth.
    pub fit_proxy_points: Vec<Point3<f32>>,
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

const MIN_FIT_VERTS: usize = 3;

fn mesh_face_area(mesh: &Mesh, fi: usize) -> f32 {
    let t = mesh.tris[fi];
    let p0 = mesh.verts[t[0] as usize];
    let p1 = mesh.verts[t[1] as usize];
    let p2 = mesh.verts[t[2] as usize];
    0.5 * (p1 - p0).cross(&(p2 - p0)).norm()
}

fn gather_fit_points(
    verts: &[Point3<f32>],
    fit: &[u32],
    all: &[u32],
    proxy_points: &[Point3<f32>],
) -> Vec<Point3<f32>> {
    let base_idx = if fit.len() >= MIN_FIT_VERTS || !proxy_points.is_empty() {
        fit
    } else {
        all
    };
    let mut pts = Vec::with_capacity(base_idx.len() + proxy_points.len());
    pts.extend(base_idx.iter().map(|&i| verts[i as usize]));
    pts.extend_from_slice(proxy_points);
    if pts.len() >= MIN_FIT_VERTS {
        pts
    } else {
        gather(verts, all)
    }
}

fn merge_fit_proxy_points(a: &[Point3<f32>], b: &[Point3<f32>]) -> Vec<Point3<f32>> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

#[derive(Clone, Copy)]
struct FacePlaneInfo {
    normal: Vector3<f32>,
    centroid: Point3<f32>,
    area: f32,
    offset: f32,
}

struct SupportPlaneCluster {
    normal: Vector3<f32>,
    offset: f32,
    area: f32,
    faces: usize,
}

fn project_point_to_plane(p: Point3<f32>, normal: Vector3<f32>, offset: f32) -> Point3<f32> {
    Point3::from(p.coords - normal * (normal.dot(&p.coords) - offset))
}

fn canonical_normal(n: Vector3<f32>) -> Vector3<f32> {
    let mut out = n;
    let flip = if out.x.abs() > 1e-4 {
        out.x < 0.0
    } else if out.y.abs() > 1e-4 {
        out.y < 0.0
    } else {
        out.z < 0.0
    };
    if flip {
        out = -out;
    }
    out
}

fn face_plane_info(mesh: &Mesh, fi: usize) -> Option<FacePlaneInfo> {
    let t = mesh.tris[fi];
    let p0 = mesh.verts[t[0] as usize];
    let p1 = mesh.verts[t[1] as usize];
    let p2 = mesh.verts[t[2] as usize];
    let cross = (p1 - p0).cross(&(p2 - p0));
    let area2 = cross.norm();
    if area2 <= 1e-12 {
        return None;
    }
    let normal = canonical_normal(cross / area2);
    let centroid = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
    let offset = normal.dot(&centroid.coords);
    Some(FacePlaneInfo {
        normal,
        centroid,
        area: 0.5 * area2,
        offset,
    })
}

fn combine_fit_mask(base: &mut Option<Vec<bool>>, next: Vec<bool>) {
    match base {
        Some(mask) => {
            for (a, b) in mask.iter_mut().zip(next) {
                *a = *a && b;
            }
        }
        None => *base = Some(next),
    }
}

fn collision_support_plane_fit_mask(
    mesh: &Mesh,
    mesh_diag: f32,
    frac: f32,
) -> (Vec<bool>, Vec<Vec<Point3<f32>>>) {
    let nf = mesh.tris.len();
    let plane_tol = frac * mesh_diag;
    let near_tol = plane_tol * 2.0;
    let detail_size = plane_tol * 1.5;
    let cluster_cos = 15.0f32.to_radians().cos();
    let detail_parallel_cos = 35.0f32.to_radians().cos();

    let infos: Vec<Option<FacePlaneInfo>> = (0..nf).map(|fi| face_plane_info(mesh, fi)).collect();
    let total_area: f32 = infos.iter().filter_map(|info| info.map(|i| i.area)).sum();
    if total_area <= 1e-12 {
        return (vec![true; nf], vec![Vec::new(); nf]);
    }

    let mut order: Vec<usize> = (0..nf).collect();
    order.sort_by(|&a, &b| {
        let aa = infos[a].map(|i| i.area).unwrap_or(0.0);
        let bb = infos[b].map(|i| i.area).unwrap_or(0.0);
        bb.partial_cmp(&aa).unwrap_or(Ordering::Equal)
    });

    let mut clusters: Vec<SupportPlaneCluster> = Vec::new();
    let mut face_cluster: Vec<Option<usize>> = vec![None; nf];
    for fi in order {
        let Some(info) = infos[fi] else {
            continue;
        };
        let mut best: Option<usize> = None;
        let mut best_dot = cluster_cos;
        for (ci, c) in clusters.iter().enumerate() {
            let dot = info.normal.dot(&c.normal);
            if dot >= best_dot && (info.offset - c.offset).abs() <= plane_tol {
                best_dot = dot;
                best = Some(ci);
            }
        }

        let ci = match best {
            Some(ci) => {
                let c = &mut clusters[ci];
                let new_area = c.area + info.area;
                let blended = c.normal * c.area + info.normal * info.area;
                if blended.norm_squared() > 1e-12 {
                    c.normal = blended.normalize();
                }
                c.offset = (c.offset * c.area + info.offset * info.area) / new_area;
                c.area = new_area;
                c.faces += 1;
                ci
            }
            None => {
                clusters.push(SupportPlaneCluster {
                    normal: info.normal,
                    offset: info.offset,
                    area: info.area,
                    faces: 1,
                });
                clusters.len() - 1
            }
        };
        face_cluster[fi] = Some(ci);
    }

    let support_area_min = (total_area * 0.01).max((mesh_diag * 0.03).powi(2));
    let mut support_clusters: Vec<usize> = clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| c.area >= support_area_min && c.faces >= 2)
        .map(|(i, _)| i)
        .collect();
    support_clusters.sort_by(|&a, &b| {
        clusters[b]
            .area
            .partial_cmp(&clusters[a].area)
            .unwrap_or(Ordering::Equal)
    });

    if support_clusters.is_empty() {
        eprintln!(
            "collision-support-planes: no support planes found (tol {:.4}, min area {:.3})",
            plane_tol, support_area_min
        );
        return (vec![true; nf], vec![Vec::new(); nf]);
    }

    let support_set: HashSet<usize> = support_clusters.iter().copied().collect();
    let mut mask = vec![true; nf];
    let mut proxy_points = vec![Vec::new(); nf];
    let mut ignored = 0usize;
    let mut near_parallel = 0usize;
    let mut near_small = 0usize;
    for fi in 0..nf {
        let Some(info) = infos[fi] else {
            continue;
        };
        if face_cluster[fi]
            .map(|ci| support_set.contains(&ci))
            .unwrap_or(false)
        {
            continue;
        }

        let mut ignore = false;
        let mut proxy_cluster: Option<usize> = None;
        for &ci in &support_clusters {
            let c = &clusters[ci];
            let dist = (info.centroid.coords.dot(&c.normal) - c.offset).abs();
            if dist > near_tol {
                continue;
            }
            let parallel = info.normal.dot(&c.normal).abs() >= detail_parallel_cos;
            let small = info.area.sqrt() <= detail_size;
            if parallel || small {
                ignore = true;
                proxy_cluster = Some(ci);
                if parallel {
                    near_parallel += 1;
                } else {
                    near_small += 1;
                }
                break;
            }
        }
        if ignore {
            mask[fi] = false;
            if let Some(ci) = proxy_cluster {
                let c = &clusters[ci];
                let tri = mesh.tris[fi];
                proxy_points[fi] = tri
                    .iter()
                    .map(|&vi| project_point_to_plane(mesh.verts[vi as usize], c.normal, c.offset))
                    .collect();
            }
            ignored += 1;
        }
    }

    let support_area: f32 = support_clusters.iter().map(|&ci| clusters[ci].area).sum();
    eprintln!(
        "collision-support-planes: {} support planes ({:.1}% area), ignored {} detail faces (parallel {}, small {}), tol {:.4} ({:.3}% diag)",
        support_clusters.len(),
        100.0 * support_area / total_area,
        ignored,
        near_parallel,
        near_small,
        plane_tol,
        frac * 100.0
    );
    (mask, proxy_points)
}

fn combined_surface_error_score(
    hausdorff: f32,
    outside: f32,
    hausdorff_threshold: f32,
    outside_threshold: Option<f32>,
) -> f32 {
    let mut score = hausdorff / hausdorff_threshold.max(1e-6);
    if let Some(threshold) = outside_threshold {
        score = score.max(outside / threshold.max(1e-6));
    }
    score
}

fn collision_shape_weight(kind: prim::PrimKind) -> f32 {
    match kind {
        prim::PrimKind::Cylinder => 0.72,
        prim::PrimKind::Capsule => 0.82,
        prim::PrimKind::Frustum => 0.88,
        prim::PrimKind::Sphere => 0.95,
        prim::PrimKind::Obb => 1.0,
        prim::PrimKind::Prism => 1.18,
    }
}

fn fit_best_mode(
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
    enabled: PrimMask,
    collision_simplify: bool,
) -> Prim {
    const FANCY_FIT_MIN_VERTS: usize = 8;
    if !collision_simplify || points.len() < FANCY_FIT_MIN_VERTS {
        return prim::fit_best(axes, points, enabled);
    }

    let mut best: Option<(f32, Prim)> = None;
    for cand in prim::fit_all(axes, points, enabled) {
        let score = cand.volume() * collision_shape_weight(cand.kind());
        match &best {
            None => best = Some((score, cand)),
            Some((best_score, _)) if score < *best_score => best = Some((score, cand)),
            _ => {}
        }
    }
    best.map(|(_, p)| p)
        .unwrap_or_else(|| prim::fit_best(axes, points, enabled))
}

fn merge_pair(
    a: &Primitive,
    b: &Primitive,
    mesh_verts: &[Point3<f32>],
    enabled: PrimMask,
    axis_override: Option<&[Vector3<f32>; 3]>,
    collision_simplify: bool,
) -> (Matrix3<f32>, Prim, f32, f32, Vec<u32>, Vec<u32>) {
    let q = a.q + b.q;
    let axes = match axis_override {
        Some(a) => *a,
        None => axes_from_q(q),
    };
    let vidx = merge_sorted_unique(&a.vertex_indices, &b.vertex_indices);
    let fit_vidx = merge_sorted_unique(&a.fit_vertex_indices, &b.fit_vertex_indices);
    let fit_proxy_points = merge_fit_proxy_points(&a.fit_proxy_points, &b.fit_proxy_points);
    let pts = gather_fit_points(mesh_verts, &fit_vidx, &vidx, &fit_proxy_points);
    let prim_fit = fit_best_mode(axes, &pts, enabled, collision_simplify);
    let vol = prim_fit.volume();
    let wvol = prim_fit.weighted_volume();
    (q, prim_fit, vol, wvol, vidx, fit_vidx)
}

fn collision_detail_like_prim(p: &Primitive, tol: f32) -> bool {
    let ext = prim_aabb_extents(&p.prim);
    let mut dims = [ext.x.abs(), ext.y.abs(), ext.z.abs()];
    dims.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));
    dims[0] <= tol || dims[1] <= tol * 2.0
}

fn collision_merge_cost_bias(
    cost: f32,
    a: &Primitive,
    b: &Primitive,
    merged: &Prim,
    mesh_diag: f32,
    threshold_frac: Option<f32>,
) -> f32 {
    let Some(frac) = threshold_frac else {
        return cost;
    };
    if frac <= 0.0 || !frac.is_finite() {
        return cost;
    }
    let tol = frac * mesh_diag.max(1e-6);
    let a_detail = collision_detail_like_prim(a, tol);
    let b_detail = collision_detail_like_prim(b, tol);
    if !a_detail && !b_detail {
        return cost;
    }

    let merged_vol = merged.volume();
    let smaller = a.volume.min(b.volume);
    let larger = a.volume.max(b.volume);
    if a_detail && b_detail {
        return cost * 0.25;
    }
    if larger > 0.0 && smaller / larger < 0.35 && merged_vol <= larger * 1.65 + smaller * 2.0 {
        return cost * 0.05 - smaller * 0.01;
    }
    cost * 0.5
}

/// Mesh-dominant orientation: eigendecomposition of the area-weighted
/// outer product of face normals summed over the entire mesh, with no
/// tangent term (we want the orientation determined by where the surface
/// faces, not by ad-hoc tangent stabilisation). For architectural meshes
/// this typically returns world-aligned axes. Used as a global
/// orientation override when `--axis-align` is on.
fn compute_dominant_axes(mesh: &Mesh) -> [Vector3<f32>; 3] {
    let mut q = Matrix3::zeros();
    for t in &mesh.tris {
        let p0 = mesh.verts[t[0] as usize];
        let p1 = mesh.verts[t[1] as usize];
        let p2 = mesh.verts[t[2] as usize];
        q += face_quadric(p0, p1, p2, 0.0);
    }
    axes_from_q(q)
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
        aabbs.push((
            Point3::new(lo[0], lo[1], lo[2]),
            Point3::new(hi[0], hi[1], hi[2]),
        ));
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

fn aabb_to_aabb_dist(a: &(Point3<f32>, Point3<f32>), b: &(Point3<f32>, Point3<f32>)) -> f32 {
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
    mesh_diag: f32,
    max_dist: f32,
    k: usize,
    max_angle_rad: f32,
    weighted_cost: bool,
    reject_pancakes: bool,
    axis_override: Option<[Vector3<f32>; 3]>,
    collision_simplify: Option<f32>,
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
            let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
                pa,
                pb,
                mesh_verts,
                enabled,
                axis_override.as_ref(),
                collision_simplify.is_some(),
            );
            let mut cost = if weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            cost =
                collision_merge_cost_bias(cost, pa, pb, &prim_fit, mesh_diag, collision_simplify);
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
    mesh_diag: f32,
    weighted_cost: bool,
    reject_pancakes: bool,
    axis_override: Option<[Vector3<f32>; 3]>,
    collision_simplify: Option<f32>,
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
            let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
                pa,
                pb,
                mesh_verts,
                enabled,
                axis_override.as_ref(),
                collision_simplify.is_some(),
            );
            let mut cost = if weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            cost =
                collision_merge_cost_bias(cost, pa, pb, &prim_fit, mesh_diag, collision_simplify);
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
    pub merges_rejected_feasibility: usize,
    pub merges_rejected_outside: usize,
    pub all_pairs_used: bool,
    pub redundant_culled: usize,
    pub overlap_culled: usize,
    pub collision_simplified: usize,
    pub thin_stripped: usize,
    pub rebalance_moves: usize,
    pub splits_done: usize,
    pub slab_repairs_done: usize,
    pub post_budget_merges: usize,
    pub high_error_shrunk: usize,
    pub refine_iters: usize,
    pub split_debug: Vec<SplitDebugRow>,
}

#[derive(Clone, Copy, Debug)]
pub struct SplitDebugOpts {
    /// Restrict split diagnostics to a specific live primitive root/id.
    /// This is the same id the viewer and metrics report.
    pub primitive: Option<u32>,
}

#[derive(Clone, Debug)]
pub struct SplitDebugRow {
    pub pass: &'static str,
    pub split_attempt: usize,
    pub primitive_root: u32,
    pub compact_pid: usize,
    pub candidate_index: usize,
    pub source: String,
    pub original_kind: prim::PrimKind,
    pub original_face_count: usize,
    pub original_hausdorff: f32,
    pub threshold: f32,
    pub faces_a: usize,
    pub faces_b: usize,
    pub kind_a: prim::PrimKind,
    pub kind_b: prim::PrimKind,
    pub hausdorff_a: f32,
    pub hausdorff_b: f32,
    pub hausdorff_max: f32,
    pub delta_hausdorff_max: f32,
    pub volume_a: f32,
    pub volume_b: f32,
    pub volume_sum: f32,
    pub delta_volume_sum: f32,
    pub would_improve: bool,
    pub accepted: bool,
    pub witness_dist: Option<f32>,
    pub witness_sample: Option<Point3<f32>>,
    pub witness_nearest: Option<Point3<f32>>,
    pub witness_normal: Option<Vector3<f32>>,
}

impl SplitDebugRow {
    pub fn json_line(&self) -> String {
        fn kind(k: prim::PrimKind) -> &'static str {
            match k {
                prim::PrimKind::Obb => "obb",
                prim::PrimKind::Sphere => "sphere",
                prim::PrimKind::Cylinder => "cylinder",
                prim::PrimKind::Capsule => "capsule",
                prim::PrimKind::Frustum => "frustum",
                prim::PrimKind::Prism => "prism",
            }
        }
        fn esc(s: &str) -> String {
            s.replace('\\', "\\\\").replace('"', "\\\"")
        }
        fn opt_f32(v: Option<f32>) -> String {
            match v {
                Some(x) if x.is_finite() => x.to_string(),
                _ => "null".to_string(),
            }
        }
        fn opt_point(v: Option<Point3<f32>>) -> String {
            match v {
                Some(p) if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() => {
                    format!("[{},{},{}]", p.x, p.y, p.z)
                }
                _ => "null".to_string(),
            }
        }
        fn opt_vec(v: Option<Vector3<f32>>) -> String {
            match v {
                Some(p) if p.x.is_finite() && p.y.is_finite() && p.z.is_finite() => {
                    format!("[{},{},{}]", p.x, p.y, p.z)
                }
                _ => "null".to_string(),
            }
        }

        format!(
            concat!(
                "{{",
                "\"pass\":\"{}\",",
                "\"split_attempt\":{},",
                "\"primitive_root\":{},",
                "\"compact_pid\":{},",
                "\"candidate_index\":{},",
                "\"source\":\"{}\",",
                "\"original_kind\":\"{}\",",
                "\"original_face_count\":{},",
                "\"original_hausdorff\":{},",
                "\"threshold\":{},",
                "\"faces_a\":{},",
                "\"faces_b\":{},",
                "\"kind_a\":\"{}\",",
                "\"kind_b\":\"{}\",",
                "\"hausdorff_a\":{},",
                "\"hausdorff_b\":{},",
                "\"hausdorff_max\":{},",
                "\"delta_hausdorff_max\":{},",
                "\"volume_a\":{},",
                "\"volume_b\":{},",
                "\"volume_sum\":{},",
                "\"delta_volume_sum\":{},",
                "\"would_improve\":{},",
                "\"accepted\":{},",
                "\"witness_dist\":{},",
                "\"witness_sample\":{},",
                "\"witness_nearest\":{},",
                "\"witness_normal\":{}",
                "}}"
            ),
            self.pass,
            self.split_attempt,
            self.primitive_root,
            self.compact_pid,
            self.candidate_index,
            esc(&self.source),
            kind(self.original_kind),
            self.original_face_count,
            self.original_hausdorff,
            self.threshold,
            self.faces_a,
            self.faces_b,
            kind(self.kind_a),
            kind(self.kind_b),
            self.hausdorff_a,
            self.hausdorff_b,
            self.hausdorff_max,
            self.delta_hausdorff_max,
            self.volume_a,
            self.volume_b,
            self.volume_sum,
            self.delta_volume_sum,
            self.would_improve,
            self.accepted,
            opt_f32(self.witness_dist),
            opt_point(self.witness_sample),
            opt_point(self.witness_nearest),
            opt_vec(self.witness_normal),
        )
    }
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
    /// Postprocess thin-OBB removal (paper appendix Fig 22, the Bistro
    /// scene). After the merge + redundant cull complete, delete any OBB
    /// whose smallest half-extent is ≤ `threshold_frac × mesh_diag`. The
    /// paper uses 1e-4 of the mesh diagonal. Targets the "many walls are
    /// entirely planar but may not be rectangular, leading to regions
    /// jutting out" failure mode: the merge produces a slab whose surface
    /// drifts metres past the actual outline; deleting that slab leaves
    /// the underlying smaller primitives in place. Different from
    /// `reject_pancakes`, which penalises the merge during the PQ — this
    /// runs after the merge has fully converged. None disables.
    pub strip_thin_obbs: Option<f32>,
    /// Merge-time feasibility check. None disables; Some(frac) sets a
    /// threshold so that any popped-and-realized merge whose resulting
    /// primitive has local Hausdorff > `frac × mesh_diag` is rejected
    /// before being committed. The two source primitives stay alive and
    /// the algorithm picks the next-best PQ candidate.
    ///
    /// Targets the failure mode where the cost function
    /// (V(merge) − V(p0) − V(p1)) is ≈ 0 for two flat coplanar slabs
    /// merging — the cost can't see surface drift past the input. A
    /// single such merge produces the "1mm × Nm slab" that dominates
    /// forward Hausdorff on architecture meshes regardless of N. The
    /// feasibility check rejects it directly using the metric we care
    /// about. Cost: BVH nearest-face on ~24 surface samples per
    /// realized merge.
    pub feasibility: Option<f32>,
    /// Outward-space preservation. None disables; Some(frac) rejects merges
    /// whose primitive surface samples are more than `frac × mesh_diag` on the
    /// positive signed-distance side of the nearest input face. This assumes
    /// reasonably outward triangle winding and is intended for collision runs
    /// where protruding outside the source envelope is worse than leaving small
    /// interior gaps.
    pub outside_space: Option<f32>,
    /// Outside-aware merge-time refit. When > 0 and `outside_space` is set,
    /// candidate primitive orientations/types are scored as
    /// `base_score * (1 + beta * outside / outside_threshold)`, where
    /// `outside` is the positive signed primitive-surface distance to the
    /// input. This nudges realized merges toward slightly larger but less
    /// protruding fits before the hard outside-space rejection runs.
    pub outside_fit_beta: f32,
    /// Post-merge split-worst pass. None disables; Some((threshold_frac,
    /// max_splits)) enables. After greedy merge converges (and rebalance
    /// + cull run, if enabled), repeatedly find the live primitive with
    /// the highest local Hausdorff. If h > `threshold_frac × mesh_diag`,
    /// split its face set along the longest PCA axis (median split into
    /// two halves) and refit each half. Accept only when both new halves
    /// have strictly lower Hausdorff than the original.
    ///
    /// Each accepted split increases primitive count by 1. `max_splits`
    /// caps growth. Targets the OBB-on-non-rectangular-planar-region
    /// failure mode (e.g. L-shaped rooftops where the bounding rectangle
    /// has corners protruding past the input outline) — a single OBB
    /// can't fit those tightly, but two can.
    pub split_worst: Option<(f32, usize)>,
    /// Final collision-oriented bad-slab repair. None disables; Some((frac,
    /// max_repairs)) runs after split/refine and before collision cleanup.
    /// It only considers medium-sized, slab-like OBBs whose sampled surface
    /// protrusion exceeds `frac * mesh_diag`. Multi-face slabs get
    /// corner/footprint split candidates; one-face triangular slabs can be
    /// replaced with OBB-only synthetic sub-triangle pieces. Repairs are
    /// accepted only if sampled error improves.
    pub repair_bad_slabs: Option<(f32, usize)>,
    /// Error-region split mode for `split_worst`. When true, the split pass
    /// also samples the worst primitive's surface, finds the point farthest
    /// from the input mesh, and tries split candidates that isolate faces
    /// near / beyond that error region. Default false: preserves the
    /// original PCA/world-axis median split behavior.
    pub split_error_region: bool,
    /// Optional JSONL diagnostics for split-worst candidate scoring.
    /// The path is owned by the CLI; this option only controls row capture.
    pub debug_splits: Option<SplitDebugOpts>,
    /// Optional final compression pass. None disables; Some((target, frac))
    /// greedily merges live primitives after repair as long as the merged
    /// primitive's sampled local surface error stays below `frac * diag`.
    pub post_merge_budget: Option<(usize, f32)>,
    /// Final collision-oriented OBB shrink pass. None disables; Some(frac)
    /// tries small half-extent reductions on high-error OBBs after budget
    /// compression/refine, accepting only dense primitive→input improvement.
    pub shrink_high_error: Option<f32>,
    /// Local-search refine pass (Park & Sung 2024-inspired). None
    /// disables; Some((threshold_frac, max_iters)) enables. After merge
    /// + cull + split, for each primitive whose local Hausdorff exceeds
    /// `threshold_frac × mesh_diag`, hill-climb its OBB orientation:
    /// try ±step rotations around each principal axis, refit half-extents
    /// to subsumed verts, accept the orientation that minimises local
    /// Hausdorff. Step adapts (start 15°, halve on no-improvement, stop
    /// at <1°). Caps per-primitive iters at `max_iters`. The full set of
    /// subsumed verts stays enclosed (refit is a tight-AABB in the new
    /// frame), so the paper's containment guarantee is preserved.
    ///
    /// Cheaper analogue of Park & Sung's MCTS-over-MDP refine — no
    /// translation/scale actions, only rotation. Rotation is the only
    /// real DoF for an OBB constrained to enclose its subsumed verts;
    /// center and half-extents are fully determined by the axis choice.
    pub refine_search: Option<(f32, usize)>,
    /// Partial-overlap cull. None disables; Some(frac) drops any live
    /// primitive A whose ≥frac fraction of tessellated surface samples
    /// lie inside another live primitive B (shared-vertex constraint
    /// applies, same as `cull_redundant`). Catches the visible-overlap
    /// stacking on hollow architectural meshes that strict
    /// `cull_redundant` (full vertex containment) misses. Try 0.85–0.95.
    pub cull_overlap: Option<f32>,
    /// Collision-oriented detail simplification. None disables. Some(frac)
    /// drops small/thin primitives when most of their surface lies within
    /// `frac * mesh_diag` of a larger live primitive. This intentionally
    /// relaxes visual-surface fidelity in favor of cleaner coarse colliders:
    /// trim, bands, bevels, and small protrusions should not force separate
    /// collision pieces when a nearby primitive already provides support.
    pub collision_simplify: Option<f32>,
    /// Optional collision-only merge target multiplier. When collision
    /// simplification is enabled, values below 1 intentionally merge past the
    /// requested visual target before postprocess cleanup, trading detail
    /// fidelity for simpler collision shapes.
    pub collision_target_scale: Option<f32>,
    /// Collision fitting detail suppression. None disables; Some(frac)
    /// marks faces whose sqrt(area) is below `frac * mesh_diag` as detail.
    /// Detail faces still participate in adjacency and can exist alone, but
    /// once they merge into a primitive with larger support faces their
    /// vertices are omitted from the fit. This intentionally relaxes the
    /// visual containment guarantee for game collision.
    pub collision_ignore_detail: Option<f32>,
    /// Collision support-plane simplification. None disables; Some(frac)
    /// detects broad coplanar-ish support clusters, then omits vertices from
    /// nearby non-support detail faces when fitting merged primitives. The
    /// fraction is the support-plane distance tolerance relative to mesh diag.
    pub collision_support_planes: Option<f32>,
    /// Lock all primitive orientations to the mesh's dominant
    /// orientation (eigendecomposition of the area-weighted face-normal
    /// outer-product summed over the entire mesh). Targets the
    /// rotated-slab failure mode on architectural meshes: the building's
    /// rooftop OBB picks an arbitrary in-plane orientation from
    /// near-rank-1 Q, which can be 30°-60° off from world axes and
    /// introduces drift past the silhouette. Locking to the global
    /// dominant axes (typically world-axis-aligned for game-art
    /// architecture) eliminates that pathology — at the cost of
    /// preventing per-primitive orientation refinement that helps on
    /// rotated organic regions. Default off.
    pub axis_align: bool,
    /// World-axis lock: like axis_align but forces (1,0,0)/(0,1,0)/(0,0,1)
    /// regardless of mesh orientation. For game architecture authored in
    /// world space (the common case) this is what users actually want —
    /// avoids picking up small ambient rotations from the source mesh.
    pub world_axis_align: bool,
    /// Per-face quadric's tangent-term coefficient (paper §3.4 "Coplanar
    /// Vertices", value `ε`). Default 0.01 stabilises the eigendecomp on
    /// coplanar regions. Setting to 0 makes Q rank-1, leaving in-plane
    /// orientation entirely to the refit's PCA/tangent-plane/sharp-edge
    /// candidates. The paper notes this is decided per-mesh; on
    /// architecture meshes with large flat surfaces, ε=0 can avoid the
    /// rotated-slab failure mode.
    pub tangent_eps: f32,
}

/// Fraction of stratified-grid samples inside `prim` that are deeper than
/// `signed_dist_threshold` *outside* the input mesh. Used to reject merges
/// that bridge open regions (stairwells, holes, vents, slots).
///
/// "Outside" here is determined by the sign of the dot product of the face
/// normal with (sample − closest_point). For non-watertight meshes this is
/// more robust than generalized winding number, which can flicker near
/// boundaries.
fn empty_space_fraction(p: &Prim, mesh: &Mesh, bvh: &Bvh, signed_dist_threshold: f32) -> f32 {
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
    // rebalance / feasibility / split-worst / refine-search).
    let bvh: Option<Bvh> = if opts.empty_space.is_some()
        || opts.quality_beta > 0.0
        || opts.shell_aware
        || opts.rebalance.is_some()
        || opts.feasibility.is_some()
        || opts.outside_space.is_some()
        || opts.outside_fit_beta > 0.0
        || opts.split_worst.is_some()
        || opts.refine_search.is_some()
        || opts.post_merge_budget.is_some()
        || opts.shrink_high_error.is_some()
    {
        Some(Bvh::build(&mesh.verts, &mesh.tris))
    } else {
        None
    };
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let collision_mode = opts.collision_simplify.is_some();
    let target_n = if collision_mode {
        opts.collision_target_scale
            .map(|scale| ((opts.target_n as f32) * scale).round().max(1.0) as usize)
            .unwrap_or(opts.target_n)
    } else {
        opts.target_n
    };
    if target_n != opts.target_n {
        eprintln!(
            "collision-target-scale: merge target {} -> {}",
            opts.target_n, target_n
        );
    }
    if opts.outside_fit_beta > 0.0 {
        eprintln!(
            "outside-fit: beta {:.3} using outside-space threshold {:.3}% of diag",
            opts.outside_fit_beta,
            opts.outside_space.unwrap_or(1.0) * 100.0
        );
    }

    let dominant_axes: Option<[Vector3<f32>; 3]> = if opts.world_axis_align {
        let axes = [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        eprintln!("world-axis-align: locked to (1,0,0)/(0,1,0)/(0,0,1)");
        Some(axes)
    } else if opts.axis_align {
        let axes = compute_dominant_axes(mesh);
        eprintln!(
            "axis-align: locked to mesh-dominant axes [{:.3},{:.3},{:.3}], [{:.3},{:.3},{:.3}], [{:.3},{:.3},{:.3}]",
            axes[0].x,
            axes[0].y,
            axes[0].z,
            axes[1].x,
            axes[1].y,
            axes[1].z,
            axes[2].x,
            axes[2].y,
            axes[2].z,
        );
        Some(axes)
    } else {
        None
    };

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

    let mut face_fit_mask: Option<Vec<bool>> = None;
    let mut face_fit_proxy_points: Option<Vec<Vec<Point3<f32>>>> = None;
    if let Some(frac) = opts.collision_ignore_detail {
        let threshold = frac * mesh_diag;
        let mut mask = Vec::with_capacity(nf);
        let mut detail = 0usize;
        for fi in 0..nf {
            let support = mesh_face_area(mesh, fi).sqrt() >= threshold;
            if !support {
                detail += 1;
            }
            mask.push(support);
        }
        eprintln!(
            "collision-ignore-detail: {} of {} faces are detail (sqrt(area) < {:.4}, {:.3}% of diag)",
            detail,
            nf,
            threshold,
            frac * 100.0
        );
        combine_fit_mask(&mut face_fit_mask, mask);
    }
    if let Some(frac) = opts.collision_support_planes {
        let (mask, proxy_points) = collision_support_plane_fit_mask(mesh, mesh_diag, frac);
        if face_fit_proxy_points.is_none() {
            face_fit_proxy_points = Some(vec![Vec::new(); nf]);
        }
        if let Some(slots) = &mut face_fit_proxy_points {
            for (fi, pts) in proxy_points.into_iter().enumerate() {
                if !pts.is_empty() {
                    slots[fi] = pts;
                }
            }
        }
        combine_fit_mask(&mut face_fit_mask, mask);
    }

    let mut prims: Vec<Primitive> = Vec::with_capacity(nf);
    for (fi, tri) in mesh.tris.iter().enumerate() {
        let p0 = mesh.verts[tri[0] as usize];
        let p1 = mesh.verts[tri[1] as usize];
        let p2 = mesh.verts[tri[2] as usize];
        // Q is linear in face area, so multiplying the per-face quadric
        // by exposure simply down-weights interior faces in any later
        // Q_a + Q_b sum during merging.
        let q_unit = face_quadric(p0, p1, p2, opts.tangent_eps);
        let mut q_weight = face_exposure.as_ref().map(|exp| exp[fi]).unwrap_or(1.0);
        if matches!(&face_fit_mask, Some(mask) if !mask[fi]) {
            q_weight *= 0.05;
        }
        let q = q_unit * q_weight;
        let axes = match &dominant_axes {
            Some(a) => *a,
            None => axes_from_q(q),
        };
        let mut vidx = [tri[0], tri[1], tri[2]];
        vidx.sort();
        let fit_vertex_indices = if matches!(&face_fit_mask, Some(mask) if !mask[fi]) {
            Vec::new()
        } else {
            vidx.to_vec()
        };
        let vertex_indices = vidx.to_vec();
        let fit_proxy_points = face_fit_proxy_points
            .as_ref()
            .map(|proxies| proxies[fi].clone())
            .unwrap_or_default();
        let pts = gather_fit_points(
            &mesh.verts,
            &fit_vertex_indices,
            &vertex_indices,
            &fit_proxy_points,
        );
        let prim_fit = fit_best_mode(axes, &pts, opts.enabled, collision_mode);
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
            vertex_indices,
            fit_vertex_indices,
            fit_proxy_points,
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
        let face_shell_mask: Option<Vec<bool>> = face_exposure
            .as_ref()
            .map(|exp| exp.iter().map(|&e| e > 0.05).collect());
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
            let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
                pa,
                pb,
                &mesh.verts,
                opts.enabled,
                dominant_axes.as_ref(),
                collision_mode,
            );
            let mut cost = if opts.weighted_cost {
                wvol - (pa.weighted_volume + pb.weighted_volume)
            } else {
                vol - (pa.volume + pb.volume)
            };
            cost = collision_merge_cost_bias(
                cost,
                pa,
                pb,
                &prim_fit,
                mesh_diag,
                opts.collision_simplify,
            );
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
    let mut merges_rejected_feasibility = 0usize;
    let mut merges_rejected_outside = 0usize;
    let mut all_pairs_used = false;
    // Memoize rejected pairs so we don't re-evaluate the BVH for the same
    // pair every time a stale/cheaper entry of theirs gets popped. Key
    // includes versions so a fresh post-merge primitive re-checks.
    let mut rejected_pairs: HashSet<(u32, u64, u32, u64)> = HashSet::new();

    while alive_count > target_n {
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
                        mesh_diag,
                        max_dist,
                        k,
                        angle_rad,
                        opts.weighted_cost,
                        opts.reject_pancakes,
                        dominant_axes,
                        opts.collision_simplify,
                    );
                    eprintln!(
                        "topology PQ drained at {} primitives; pushed {} proximity candidates (k={}, r={:.3}, angle<={:.0}°)",
                        alive_count,
                        p,
                        k,
                        max_dist,
                        angle_rad.to_degrees(),
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
                        mesh_diag,
                        opts.weighted_cost,
                        opts.reject_pancakes,
                        dominant_axes,
                        opts.collision_simplify,
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
        let new_fit_vidx =
            merge_sorted_unique(&prims[a].fit_vertex_indices, &prims[b].fit_vertex_indices);
        let new_fit_proxy_points =
            merge_fit_proxy_points(&prims[a].fit_proxy_points, &prims[b].fit_proxy_points);

        // Optional post-merge orientation refinement. The Q-eigenbasis
        // orientation is what the cached prim was fit against, so the refit
        // can only equal-or-improve the cached primitive — no need to
        // re-push the candidate to the priority queue.
        let (new_prim, new_vol, new_wvol) = if opts.refine_orient {
            // Containment fits use support vertices. In collision detail
            // suppression mode this intentionally omits decorative vertices
            // once a primitive has larger support faces.
            let pts =
                gather_fit_points(&mesh.verts, &new_fit_vidx, &new_vidx, &new_fit_proxy_points);
            // Orientation fits prefer shell vertices when shell-awareness
            // is on. Fall back to the full set if too few shell verts.
            let pts_orient: Vec<Point3<f32>> = match &shell_vertex_mask {
                Some(mask) => {
                    let mut v: Vec<Point3<f32>> = new_fit_vidx
                        .iter()
                        .filter(|&&i| mask[i as usize])
                        .map(|&i| mesh.verts[i as usize])
                        .collect();
                    v.extend_from_slice(&new_fit_proxy_points);
                    if v.len() < 4 { pts.clone() } else { v }
                }
                None => pts.clone(),
            };

            // Selection criterion: when quality_beta / outside_fit are 0,
            // just minimise weighted volume (paper-faithful). Quality mode
            // adds a cheap sampled Hausdorff penalty; outside-fit adds a
            // positive signed-distance penalty using the outside-space
            // threshold so merge-time refit can prefer slightly larger but
            // less protruding candidates.
            let use_quality = opts.quality_beta > 0.0 && bvh.is_some();
            let outside_threshold = opts.outside_space.map(|frac| frac * mesh_diag);
            let use_outside_fit = opts.outside_fit_beta > 0.0 && bvh.is_some();
            let bvh_ref = bvh.as_ref();

            let base_score = |p: &Prim| -> f32 {
                if collision_mode {
                    p.volume() * collision_shape_weight(p.kind())
                } else {
                    p.weighted_volume()
                }
            };
            let score = |p: &Prim| -> f32 {
                let mut s = base_score(p);
                if use_quality {
                    let h = local_hausdorff(p, bvh_ref.unwrap(), mesh);
                    s *= 1.0 + opts.quality_beta * h / mesh_diag;
                }
                if use_outside_fit {
                    let outside = local_outside(p, bvh_ref.unwrap(), mesh);
                    let denom = outside_threshold.unwrap_or(mesh_diag).max(1e-6);
                    s *= 1.0 + opts.outside_fit_beta * outside / denom;
                }
                s
            };

            let mut best_prim = entry.prim;
            let mut best_score = score(&best_prim);

            let try_axes = |axes: [Vector3<f32>; 3], best: &mut Prim, best_score: &mut f32| {
                if use_quality || use_outside_fit {
                    // try every primitive type for this orientation, score
                    // each individually.
                    let mask = if opts.enabled.obb || opts.enabled.sphere {
                        let mut m = opts.enabled;
                        if use_quality {
                            m.sphere = true; // unconditional sphere in quality mode
                        }
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
                    let cand = fit_best_mode(axes, &pts, opts.enabled, collision_mode);
                    let s = score(&cand);
                    if s < *best_score {
                        *best_score = s;
                        *best = cand;
                    }
                }
            };

            if let Some(da) = &dominant_axes {
                // Axis-align mode: only the locked dominant axes are
                // considered for refit. PCA / tangent-plane / sharp-edge
                // candidates would re-introduce per-primitive rotated
                // orientations the user explicitly asked us to avoid.
                try_axes(*da, &mut best_prim, &mut best_score);
            } else {
                // Candidate 1: vertex PCA (shell-only when shell-aware is on).
                try_axes(pca_axes(&pts_orient), &mut best_prim, &mut best_score);
                // Candidate 2: tangent-plane PCA (shell-only when on).
                try_axes(
                    tangent_plane_pca_axes(new_q, &pts_orient),
                    &mut best_prim,
                    &mut best_score,
                );
                // Candidate 3: sharp-edge directions.
                if let Some(sharp_ref) = &sharp_edges {
                    let face_iter = walk_faces(a as u32, prims[a].face_count, &face_next)
                        .chain(walk_faces(b as u32, prims[b].face_count, &face_next));
                    if let Some(axes) = sharp_edge_axes(sharp_ref, face_iter) {
                        try_axes(axes, &mut best_prim, &mut best_score);
                    }
                }
            }

            let v = best_prim.volume();
            let w = best_prim.weighted_volume();
            (best_prim, v, w)
        } else {
            (entry.prim, entry.volume, entry.weighted_volume)
        };

        // Merge-time feasibility check. The PQ cost is V(merge) − V(p0) −
        // V(p1); for two flat coplanar slabs merging, the cost is ≈ 0 even
        // though the merged primitive's surface drifts metres past the
        // input. Sample the merged primitive against the input mesh BVH
        // and reject if the local Hausdorff exceeds the configured
        // fraction of mesh diag. The two source primitives stay alive,
        // their other PQ candidates are still in the queue, and the loop
        // continues with the next-best candidate.
        if let (Some(frac), Some(bvh_ref)) = (opts.feasibility, &bvh) {
            let h = local_hausdorff(&new_prim, bvh_ref, mesh);
            if h > frac * mesh_diag {
                merges_rejected_feasibility += 1;
                continue;
            }
        }

        if let (Some(frac), Some(bvh_ref)) = (opts.outside_space, &bvh) {
            let h = local_outside(&new_prim, bvh_ref, mesh);
            if h > frac * mesh_diag {
                merges_rejected_outside += 1;
                continue;
            }
        }

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
        prims[a].fit_vertex_indices = new_fit_vidx;
        prims[a].fit_proxy_points = new_fit_proxy_points;
        prims[a].neighbors = new_neighbors;
        prims[a].version += 1;

        prims[b].alive = false;
        prims[b].neighbors.clear();
        // free the now-stale vertex/face data on the loser slot to reclaim mem
        prims[b].vertex_indices = Vec::new();
        prims[b].fit_vertex_indices = Vec::new();
        prims[b].fit_proxy_points = Vec::new();
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
            let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
                pa,
                pn,
                &mesh.verts,
                opts.enabled,
                dominant_axes.as_ref(),
                collision_mode,
            );
            let mut cost = if opts.weighted_cost {
                wvol - (pa.weighted_volume + pn.weighted_volume)
            } else {
                vol - (pa.volume + pn.volume)
            };
            cost = collision_merge_cost_bias(
                cost,
                pa,
                pn,
                &prim_fit,
                mesh_diag,
                opts.collision_simplify,
            );
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
                opts.tangent_eps,
                dominant_axes.as_ref(),
                face_fit_mask.as_deref(),
                face_fit_proxy_points.as_deref(),
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

    let overlap_culled = match opts.cull_overlap {
        Some(frac) => cull_overlapping(&mut prims, frac),
        None => 0,
    };

    let mut split_debug_rows: Vec<SplitDebugRow> = Vec::new();
    let mut splits_done = 0usize;
    if let (Some((threshold_frac, max_splits)), Some(bvh_ref)) = (opts.split_worst, &bvh) {
        let t = std::time::Instant::now();
        splits_done = split_worst_primitives(
            &mut prims,
            mesh,
            adj,
            &mut dsu,
            &mut face_next,
            bvh_ref,
            opts.enabled,
            threshold_frac,
            max_splits,
            opts.tangent_eps,
            dominant_axes.as_ref(),
            face_fit_mask.as_deref(),
            face_fit_proxy_points.as_deref(),
            opts.split_error_region,
            opts.outside_space,
            opts.debug_splits,
            &mut split_debug_rows,
            "pre_refine",
            SplitPassMode::General,
        );
        eprintln!(
            "split-worst: {} primitives split in {:.1} ms",
            splits_done,
            t.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let mut refine_iters = 0usize;
    if let (Some((threshold_frac, max_iters)), Some(bvh_ref)) = (opts.refine_search, &bvh) {
        let t = std::time::Instant::now();
        refine_iters = refine_search_pass(
            &mut prims,
            mesh,
            bvh_ref,
            threshold_frac,
            max_iters,
            opts.outside_space,
        );
        eprintln!(
            "refine-search: {} hill-climb iters in {:.1} ms",
            refine_iters,
            t.elapsed().as_secs_f64() * 1000.0,
        );
    }

    if opts.split_error_region {
        if let (Some((threshold_frac, max_splits)), Some(bvh_ref)) = (opts.split_worst, &bvh) {
            let remaining = max_splits.saturating_sub(splits_done);
            if remaining > 0 {
                let t = std::time::Instant::now();
                let post_splits = split_worst_primitives(
                    &mut prims,
                    mesh,
                    adj,
                    &mut dsu,
                    &mut face_next,
                    bvh_ref,
                    opts.enabled,
                    threshold_frac,
                    remaining,
                    opts.tangent_eps,
                    dominant_axes.as_ref(),
                    face_fit_mask.as_deref(),
                    face_fit_proxy_points.as_deref(),
                    true,
                    opts.outside_space,
                    opts.debug_splits,
                    &mut split_debug_rows,
                    "post_refine",
                    SplitPassMode::General,
                );
                splits_done += post_splits;
                eprintln!(
                    "split-error-region: {} post-refine primitives split in {:.1} ms",
                    post_splits,
                    t.elapsed().as_secs_f64() * 1000.0,
                );

                if post_splits > 0 {
                    if let Some((threshold_frac, max_iters)) = opts.refine_search {
                        let t = std::time::Instant::now();
                        let post_refine_iters = refine_search_pass(
                            &mut prims,
                            mesh,
                            bvh_ref,
                            threshold_frac,
                            max_iters,
                            opts.outside_space,
                        );
                        refine_iters += post_refine_iters;
                        eprintln!(
                            "refine-search post-split: {} hill-climb iters in {:.1} ms",
                            post_refine_iters,
                            t.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
            }
        }
    }

    let mut slab_repairs_done = 0usize;
    if let (Some((threshold_frac, max_repairs)), Some(bvh_ref)) = (opts.repair_bad_slabs, &bvh) {
        let t = std::time::Instant::now();
        slab_repairs_done = split_worst_primitives(
            &mut prims,
            mesh,
            adj,
            &mut dsu,
            &mut face_next,
            bvh_ref,
            opts.enabled,
            threshold_frac,
            max_repairs,
            opts.tangent_eps,
            dominant_axes.as_ref(),
            face_fit_mask.as_deref(),
            face_fit_proxy_points.as_deref(),
            true,
            opts.outside_space,
            opts.debug_splits,
            &mut split_debug_rows,
            "bad_slab_repair",
            SplitPassMode::SlabCornerRepair,
        );
        eprintln!(
            "repair-bad-slabs: {} slab primitives split in {:.1} ms",
            slab_repairs_done,
            t.elapsed().as_secs_f64() * 1000.0,
        );

        let remaining_repairs = max_repairs.saturating_sub(slab_repairs_done);
        if remaining_repairs > 0 {
            let t = std::time::Instant::now();
            let subdivided = repair_few_face_slab_obbs(
                &mut prims,
                mesh,
                &face_next,
                bvh_ref,
                threshold_frac,
                remaining_repairs,
                opts.tangent_eps,
                dominant_axes.as_ref(),
                opts.outside_space,
            );
            slab_repairs_done += subdivided;
            eprintln!(
                "repair-bad-slabs: {} few-face slab primitives subdivided in {:.1} ms",
                subdivided,
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        let remaining_repairs = max_repairs.saturating_sub(slab_repairs_done);
        if remaining_repairs > 0 {
            let t = std::time::Instant::now();
            let trimmed = repair_single_face_slab_obbs(
                &mut prims,
                mesh,
                &face_next,
                bvh_ref,
                threshold_frac,
                remaining_repairs,
                opts.tangent_eps,
                dominant_axes.as_ref(),
                opts.outside_space,
            );
            slab_repairs_done += trimmed;
            eprintln!(
                "repair-bad-slabs: {} one-face slab primitives trimmed in {:.1} ms",
                trimmed,
                t.elapsed().as_secs_f64() * 1000.0,
            );
        }

        if slab_repairs_done > 0 {
            if let Some((threshold_frac, max_iters)) = opts.refine_search {
                let t = std::time::Instant::now();
                let repair_refine_iters = refine_search_pass(
                    &mut prims,
                    mesh,
                    bvh_ref,
                    threshold_frac,
                    max_iters,
                    opts.outside_space,
                );
                refine_iters += repair_refine_iters;
                eprintln!(
                    "refine-search post-slab-repair: {} hill-climb iters in {:.1} ms",
                    repair_refine_iters,
                    t.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }

    if let (Some(debug), Some((threshold_frac, _)), Some(bvh_ref)) =
        (opts.debug_splits, opts.split_worst, &bvh)
    {
        if let Some(target) = debug.primitive {
            let already_logged = split_debug_rows
                .iter()
                .any(|row| row.primitive_root == target);
            if !already_logged {
                debug_live_primitive_splits(
                    &prims,
                    mesh,
                    &face_next,
                    bvh_ref,
                    opts.enabled,
                    threshold_frac,
                    opts.tangent_eps,
                    dominant_axes.as_ref(),
                    face_fit_mask.as_deref(),
                    face_fit_proxy_points.as_deref(),
                    opts.split_error_region,
                    target,
                    &mut split_debug_rows,
                );
            }
        }
    }

    let mut post_budget_merges = 0usize;
    if let (Some((target, threshold_frac)), Some(bvh_ref)) = (opts.post_merge_budget, &bvh) {
        let t = std::time::Instant::now();
        post_budget_merges = post_merge_to_budget(
            &mut prims,
            mesh,
            &face_next,
            bvh_ref,
            opts.enabled,
            target,
            threshold_frac,
            opts.outside_space,
            dominant_axes.as_ref(),
            opts.collision_simplify.is_some(),
        );
        eprintln!(
            "post-merge-budget: {} accepted merges in {:.1} ms",
            post_budget_merges,
            t.elapsed().as_secs_f64() * 1000.0,
        );
        if post_budget_merges > 0 {
            if let Some((threshold_frac, max_iters)) = opts.refine_search {
                let t = std::time::Instant::now();
                let iters = refine_search_pass(
                    &mut prims,
                    mesh,
                    bvh_ref,
                    threshold_frac,
                    max_iters,
                    opts.outside_space,
                );
                refine_iters += iters;
                eprintln!(
                    "refine-search post-budget: {} hill-climb iters in {:.1} ms",
                    iters,
                    t.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
    }

    let mut high_error_shrunk = 0usize;
    if let (Some(threshold_frac), Some(bvh_ref)) = (opts.shrink_high_error, &bvh) {
        let t = std::time::Instant::now();
        high_error_shrunk = shrink_high_error_obbs(&mut prims, mesh, bvh_ref, threshold_frac);
        eprintln!(
            "shrink-high-error: adjusted {} OBBs in {:.1} ms",
            high_error_shrunk,
            t.elapsed().as_secs_f64() * 1000.0,
        );
    }

    let collision_simplified = match opts.collision_simplify {
        Some(frac) => {
            let t = std::time::Instant::now();
            let removed = collision_simplify_primitives(&mut prims, mesh_diag, frac);
            eprintln!(
                "collision-simplify: removed {} detail primitives in {:.1} ms",
                removed,
                t.elapsed().as_secs_f64() * 1000.0,
            );
            removed
        }
        None => 0,
    };

    let thin_stripped = match opts.strip_thin_obbs {
        Some(frac) => strip_thin_obbs(&mut prims, &mesh.verts, frac),
        None => 0,
    };

    DecompResult {
        primitives: prims,
        merges_done,
        merges_skipped_stale,
        merges_rejected_empty,
        merges_rejected_feasibility,
        merges_rejected_outside,
        all_pairs_used,
        redundant_culled,
        overlap_culled,
        collision_simplified,
        thin_stripped,
        rebalance_moves,
        splits_done,
        slab_repairs_done,
        post_budget_merges,
        high_error_shrunk,
        refine_iters,
        split_debug: split_debug_rows,
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
    /// Support vertex subset used for collision detail-suppressed fitting.
    fit_vertex_indices: Vec<u32>,
    fit_proxy_points: Vec<Point3<f32>>,
}

fn refit_from_faces(
    faces: &[u32],
    mesh: &Mesh,
    enabled: PrimMask,
    bvh: &Bvh,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    face_fit_mask: Option<&[bool]>,
    face_fit_proxy_points: Option<&[Vec<Point3<f32>>]>,
) -> RebalanceState {
    // Sum per-face quadrics, gather subsumed vertices.
    let mut q = Matrix3::zeros();
    let mut vidx_set: HashSet<u32> = HashSet::new();
    let mut fit_vidx_set: HashSet<u32> = HashSet::new();
    let mut fit_proxy_points: Vec<Point3<f32>> = Vec::new();
    for &fi in faces {
        let t = mesh.tris[fi as usize];
        let p0 = mesh.verts[t[0] as usize];
        let p1 = mesh.verts[t[1] as usize];
        let p2 = mesh.verts[t[2] as usize];
        let mut q_weight = 1.0;
        if matches!(face_fit_mask, Some(mask) if !mask[fi as usize]) {
            q_weight = 0.05;
        }
        q += face_quadric(p0, p1, p2, tangent_eps) * q_weight;
        vidx_set.insert(t[0]);
        vidx_set.insert(t[1]);
        vidx_set.insert(t[2]);
        if matches!(face_fit_mask, Some(mask) if !mask[fi as usize]) {
            if let Some(proxies) = face_fit_proxy_points {
                fit_proxy_points.extend_from_slice(&proxies[fi as usize]);
            }
        } else {
            fit_vidx_set.insert(t[0]);
            fit_vidx_set.insert(t[1]);
            fit_vidx_set.insert(t[2]);
        }
    }
    let mut vidx: Vec<u32> = vidx_set.into_iter().collect();
    vidx.sort();
    let mut fit_vidx: Vec<u32> = fit_vidx_set.into_iter().collect();
    fit_vidx.sort();
    let pts = gather_fit_points(&mesh.verts, &fit_vidx, &vidx, &fit_proxy_points);
    let axes = match axis_override {
        Some(a) => *a,
        None => axes_from_q(q),
    };
    let prim_fit = prim::fit_best(axes, &pts, enabled);
    let h = local_hausdorff(&prim_fit, bvh, mesh);
    RebalanceState {
        volume: prim_fit.volume(),
        weighted_volume: prim_fit.weighted_volume(),
        prim: prim_fit,
        q,
        hausdorff: h,
        vertex_indices: vidx,
        fit_vertex_indices: fit_vidx,
        fit_proxy_points,
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
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    face_fit_mask: Option<&[bool]>,
    face_fit_proxy_points: Option<&[Vec<Point3<f32>>]>,
) -> usize {
    // Combined cost so we don't accept moves that improve Hausdorff at
    // catastrophic volume cost. Same shape as --quality.
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let beta = 5.0f32;
    let score_state =
        |s: &RebalanceState| -> f32 { s.weighted_volume * (1.0 + beta * s.hausdorff / mesh_diag) };
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
        .map(|pid| {
            refit_from_faces(
                &prim_faces[pid],
                mesh,
                enabled,
                bvh,
                tangent_eps,
                axis_override,
                face_fit_mask,
                face_fit_proxy_points,
            )
        })
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

                let new_a = refit_from_faces(
                    &new_a_faces,
                    mesh,
                    enabled,
                    bvh,
                    tangent_eps,
                    axis_override,
                    face_fit_mask,
                    face_fit_proxy_points,
                );
                let new_b = refit_from_faces(
                    &new_b_faces,
                    mesh,
                    enabled,
                    bvh,
                    tangent_eps,
                    axis_override,
                    face_fit_mask,
                    face_fit_proxy_points,
                );
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
        eprintln!("rebalance pass {}: {} face moves", pass + 1, moves);
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
            fit_vertex_indices: s.fit_vertex_indices.clone(),
            fit_proxy_points: s.fit_proxy_points.clone(),
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
            prims[fi].fit_vertex_indices = Vec::new();
            prims[fi].fit_proxy_points = Vec::new();
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

fn primitive_center(prim: &Prim) -> Point3<f32> {
    match prim {
        Prim::Obb { center, .. }
        | Prim::Sphere { center, .. }
        | Prim::Cylinder { center, .. }
        | Prim::Capsule { center, .. }
        | Prim::Frustum { center, .. }
        | Prim::Prism { center, .. } => *center,
    }
}

fn face_centroid(mesh: &Mesh, fi: u32) -> Point3<f32> {
    let t = mesh.tris[fi as usize];
    Point3::from(
        (mesh.verts[t[0] as usize].coords
            + mesh.verts[t[1] as usize].coords
            + mesh.verts[t[2] as usize].coords)
            / 3.0,
    )
}

fn normalized_axis(v: Vector3<f32>) -> Option<Vector3<f32>> {
    if v.norm_squared() > 1e-12 && v.x.is_finite() && v.y.is_finite() && v.z.is_finite() {
        Some(v.normalize())
    } else {
        None
    }
}

#[derive(Clone)]
struct SplitCandidate {
    source: String,
    faces_a: Vec<u32>,
    faces_b: Vec<u32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SplitPassMode {
    General,
    SlabCornerRepair,
}

fn push_split_candidate(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: String,
    mut a: Vec<u32>,
    mut b: Vec<u32>,
) {
    if a.is_empty() || b.is_empty() {
        return;
    }
    a.sort_unstable();
    b.sort_unstable();
    let key = if a <= b { a.clone() } else { b.clone() };
    if seen.insert(key) {
        out.push(SplitCandidate {
            source,
            faces_a: a,
            faces_b: b,
        });
    }
}

fn push_projection_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: &str,
    faces: &[u32],
    mesh: &Mesh,
    axis: Vector3<f32>,
    fractions: &[f32],
) {
    let Some(axis) = normalized_axis(axis) else {
        return;
    };
    if faces.len() < 2 {
        return;
    }
    let mut face_proj: Vec<(u32, f32)> = faces
        .iter()
        .map(|&fi| (fi, face_centroid(mesh, fi).coords.dot(&axis)))
        .collect();
    face_proj.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    for &frac in fractions {
        if !(0.0..=1.0).contains(&frac) {
            continue;
        }
        let cut = ((face_proj.len() as f32) * frac).round() as usize;
        let cut = cut.clamp(1, face_proj.len() - 1);
        let faces_a: Vec<u32> = face_proj[..cut].iter().map(|&(fi, _)| fi).collect();
        let faces_b: Vec<u32> = face_proj[cut..].iter().map(|&(fi, _)| fi).collect();
        push_split_candidate(
            out,
            seen,
            format!("{}:q{:.2}", source, frac),
            faces_a,
            faces_b,
        );
    }
}

fn push_threshold_split_candidate(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: &str,
    faces: &[u32],
    mesh: &Mesh,
    axis: Vector3<f32>,
    threshold: f32,
) {
    let Some(axis) = normalized_axis(axis) else {
        return;
    };
    let mut faces_a = Vec::new();
    let mut faces_b = Vec::new();
    for &fi in faces {
        if face_centroid(mesh, fi).coords.dot(&axis) <= threshold {
            faces_a.push(fi);
        } else {
            faces_b.push(fi);
        }
    }
    push_split_candidate(out, seen, source.to_string(), faces_a, faces_b);
}

fn push_nearest_point_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: &str,
    faces: &[u32],
    mesh: &Mesh,
    point: Point3<f32>,
    fractions: &[f32],
) {
    if faces.len() < 2 {
        return;
    }
    let mut face_dist: Vec<(u32, f32)> = faces
        .iter()
        .map(|&fi| (fi, (face_centroid(mesh, fi) - point).norm_squared()))
        .collect();
    face_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    for &frac in fractions {
        if !(0.0..=1.0).contains(&frac) {
            continue;
        }
        let cut = ((face_dist.len() as f32) * frac).round() as usize;
        let cut = cut.clamp(1, face_dist.len() - 1);
        let faces_a: Vec<u32> = face_dist[..cut].iter().map(|&(fi, _)| fi).collect();
        let faces_b: Vec<u32> = face_dist[cut..].iter().map(|&(fi, _)| fi).collect();
        push_split_candidate(
            out,
            seen,
            format!("{}:nearest{:.2}", source, frac),
            faces_a,
            faces_b,
        );
    }
}

fn primitive_footprint_axes(prim: &Prim) -> Option<(Vector3<f32>, Vector3<f32>)> {
    let mut axes_with_extent: [(Vector3<f32>, f32); 3] = match *prim {
        Prim::Obb {
            axes, half_extents, ..
        } => [
            (axes[0], half_extents[0]),
            (axes[1], half_extents[1]),
            (axes[2], half_extents[2]),
        ],
        Prim::Prism {
            axes,
            hx,
            hy,
            hzt,
            hzb,
            ..
        } => [(axes[0], hx), (axes[1], hy), (axes[2], hzt.max(hzb))],
        _ => return None,
    };
    axes_with_extent.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let a0 = normalized_axis(axes_with_extent[0].0)?;
    let a1 = normalized_axis(axes_with_extent[1].0 - a0 * a0.dot(&axes_with_extent[1].0))?;
    Some((a0, a1))
}

fn push_projected_kmeans_split_candidate(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: &str,
    faces: &[u32],
    mesh: &Mesh,
    axis0: Vector3<f32>,
    axis1: Vector3<f32>,
    seed_a: (f32, f32),
    seed_b: (f32, f32),
) {
    if faces.len() < 2 {
        return;
    }
    let Some(axis0) = normalized_axis(axis0) else {
        return;
    };
    let Some(axis1) = normalized_axis(axis1 - axis0 * axis0.dot(&axis1)) else {
        return;
    };
    let samples: Vec<(u32, f32, f32)> = faces
        .iter()
        .map(|&fi| {
            let c = face_centroid(mesh, fi);
            (fi, c.coords.dot(&axis0), c.coords.dot(&axis1))
        })
        .collect();

    let mut sa = seed_a;
    let mut sb = seed_b;
    let mut assign_a = vec![false; samples.len()];
    for _ in 0..5 {
        let mut count_a = 0usize;
        let mut count_b = 0usize;
        let mut sum_a = (0.0f32, 0.0f32);
        let mut sum_b = (0.0f32, 0.0f32);
        for (i, &(_, x, y)) in samples.iter().enumerate() {
            let da = (x - sa.0).powi(2) + (y - sa.1).powi(2);
            let db = (x - sb.0).powi(2) + (y - sb.1).powi(2);
            assign_a[i] = da <= db;
            if assign_a[i] {
                count_a += 1;
                sum_a.0 += x;
                sum_a.1 += y;
            } else {
                count_b += 1;
                sum_b.0 += x;
                sum_b.1 += y;
            }
        }
        if count_a == 0 || count_b == 0 {
            return;
        }
        sa = (sum_a.0 / count_a as f32, sum_a.1 / count_a as f32);
        sb = (sum_b.0 / count_b as f32, sum_b.1 / count_b as f32);
    }

    let mut faces_a = Vec::new();
    let mut faces_b = Vec::new();
    for (i, &(fi, _, _)) in samples.iter().enumerate() {
        if assign_a[i] {
            faces_a.push(fi);
        } else {
            faces_b.push(fi);
        }
    }
    push_split_candidate(out, seen, source.to_string(), faces_a, faces_b);
}

fn push_footprint_gap_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    faces: &[u32],
    mesh: &Mesh,
    prim: &Prim,
    witness: HausdorffWitness,
) {
    let Some((axis0, axis1)) = primitive_footprint_axes(prim) else {
        return;
    };
    if faces.len() < 2 {
        return;
    }

    let mut lo0 = f32::INFINITY;
    let mut hi0 = f32::NEG_INFINITY;
    let mut lo1 = f32::INFINITY;
    let mut hi1 = f32::NEG_INFINITY;
    for &fi in faces {
        let c = face_centroid(mesh, fi);
        let x = c.coords.dot(&axis0);
        let y = c.coords.dot(&axis1);
        lo0 = lo0.min(x);
        hi0 = hi0.max(x);
        lo1 = lo1.min(y);
        hi1 = hi1.max(y);
    }
    if !lo0.is_finite() || !lo1.is_finite() || hi0 <= lo0 || hi1 <= lo1 {
        return;
    }

    let delta = witness.sample - witness.nearest;
    let far0 = if delta.dot(&axis0) >= 0.0 { hi0 } else { lo0 };
    let near0 = if delta.dot(&axis0) >= 0.0 { lo0 } else { hi0 };
    let far1 = if delta.dot(&axis1) >= 0.0 { hi1 } else { lo1 };
    let near1 = if delta.dot(&axis1) >= 0.0 { lo1 } else { hi1 };
    let mid0 = 0.5 * (lo0 + hi0);
    let mid1 = 0.5 * (lo1 + hi1);
    let witness_seed = (
        witness.sample.coords.dot(&axis0).clamp(lo0, hi0),
        witness.sample.coords.dot(&axis1).clamp(lo1, hi1),
    );
    let support_seed = (
        witness.nearest.coords.dot(&axis0).clamp(lo0, hi0),
        witness.nearest.coords.dot(&axis1).clamp(lo1, hi1),
    );

    // L-shaped slabs usually fail at an unsupported rectangle corner.
    // These two seeds target the arms adjacent to that corner, while the
    // diagonal and support/witness seeds cover simpler one-sided protrusions.
    let seed_pairs = [
        ("footprint:arms", (far0, near1), (near0, far1)),
        ("footprint:diagonal", (far0, far1), (near0, near1)),
        ("footprint:axis0", (far0, mid1), (near0, mid1)),
        ("footprint:axis1", (mid0, far1), (mid0, near1)),
        ("footprint:witness_support", witness_seed, support_seed),
    ];
    for (source, seed_a, seed_b) in seed_pairs {
        push_projected_kmeans_split_candidate(
            out, seen, source, faces, mesh, axis0, axis1, seed_a, seed_b,
        );
    }
}

fn is_medium_slab_prim(prim: &Prim, volume: f32, mesh_diag: f32) -> bool {
    let half_extents = match prim {
        Prim::Obb { half_extents, .. } => *half_extents,
        _ => return false,
    };
    let mut dims = [
        half_extents[0].abs() * 2.0,
        half_extents[1].abs() * 2.0,
        half_extents[2].abs() * 2.0,
    ];
    dims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let min_volume = (mesh_diag * 0.025).powi(3);
    volume >= min_volume
        && dims[0] <= mesh_diag * 0.08
        && dims[1] >= mesh_diag * 0.035
        && dims[2] >= mesh_diag * 0.08
}

fn is_medium_slab_obb(state: &RebalanceState, mesh_diag: f32) -> bool {
    is_medium_slab_prim(&state.prim, state.volume, mesh_diag)
}

fn is_repairable_slab_prim(prim: &Prim, volume: f32, mesh_diag: f32) -> bool {
    if is_medium_slab_prim(prim, volume, mesh_diag) {
        return true;
    }
    let half_extents = match prim {
        Prim::Obb { half_extents, .. } => *half_extents,
        _ => return false,
    };
    let mut dims = [
        half_extents[0].abs() * 2.0,
        half_extents[1].abs() * 2.0,
        half_extents[2].abs() * 2.0,
    ];
    dims.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    dims[0] <= mesh_diag * 0.01 && dims[1] >= mesh_diag * 0.03 && dims[2] >= mesh_diag * 0.06
}

fn push_ranked_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    source: &str,
    mut ranked: Vec<(u32, f32)>,
    fractions: &[f32],
) {
    if ranked.len() < 4 {
        return;
    }
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    for &frac in fractions {
        if !(0.0..=1.0).contains(&frac) {
            continue;
        }
        let cut = ((ranked.len() as f32) * frac).round() as usize;
        let cut = cut.clamp(2, ranked.len().saturating_sub(2));
        let faces_a: Vec<u32> = ranked[..cut].iter().map(|&(fi, _)| fi).collect();
        let faces_b: Vec<u32> = ranked[cut..].iter().map(|&(fi, _)| fi).collect();
        push_split_candidate(
            out,
            seen,
            format!("{}:q{:.2}", source, frac),
            faces_a,
            faces_b,
        );
    }
}

fn push_slab_corner_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    faces: &[u32],
    mesh: &Mesh,
    prim: &Prim,
    witness: HausdorffWitness,
) {
    const EDGE_FRACS: &[f32] = &[0.12, 0.18, 0.25, 0.35, 0.65, 0.75, 0.82, 0.88];
    const CORNER_FRACS: &[f32] = &[0.65, 0.75, 0.82, 0.88];

    let (center, axes, half_extents) = match prim {
        Prim::Obb {
            center,
            axes,
            half_extents,
        } => (*center, *axes, *half_extents),
        _ => return,
    };
    if faces.len() < 4 {
        return;
    }

    let mut axes_by_extent = [
        (0usize, half_extents[0]),
        (1, half_extents[1]),
        (2, half_extents[2]),
    ];
    axes_by_extent.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    let footprint = [axes_by_extent[0].0, axes_by_extent[1].0];

    let rel = witness.sample - center;
    for &axis_idx in &footprint {
        let sign = if rel.dot(&axes[axis_idx]) >= 0.0 {
            1.0
        } else {
            -1.0
        };
        let axis = axes[axis_idx] * sign;
        push_projection_split_candidates(
            out,
            seen,
            &format!("slab:edge_axis{}", axis_idx),
            faces,
            mesh,
            axis,
            EDGE_FRACS,
        );
        push_threshold_split_candidate(
            out,
            seen,
            &format!("slab:nearest_axis{}", axis_idx),
            faces,
            mesh,
            axis,
            witness.nearest.coords.dot(&axis),
        );
    }

    let axis0_idx = footprint[0];
    let axis1_idx = footprint[1];
    let sign0 = if rel.dot(&axes[axis0_idx]) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let sign1 = if rel.dot(&axes[axis1_idx]) >= 0.0 {
        1.0
    } else {
        -1.0
    };
    let h0 = half_extents[axis0_idx].abs().max(1e-6);
    let h1 = half_extents[axis1_idx].abs().max(1e-6);
    let ranked: Vec<(u32, f32)> = faces
        .iter()
        .map(|&fi| {
            let c = face_centroid(mesh, fi);
            let d = c - center;
            let u = sign0 * d.dot(&axes[axis0_idx]) / h0;
            let v = sign1 * d.dot(&axes[axis1_idx]) / h1;
            (fi, u + v)
        })
        .collect();
    push_ranked_split_candidates(out, seen, "slab:corner_score", ranked, CORNER_FRACS);
}

fn push_error_region_split_candidates(
    out: &mut Vec<SplitCandidate>,
    seen: &mut HashSet<Vec<u32>>,
    faces: &[u32],
    mesh: &Mesh,
    prim: &Prim,
    witness: HausdorffWitness,
) {
    const QUANTILES: &[f32] = &[0.15, 0.25, 0.35, 0.65, 0.75, 0.85];
    const NEAR_FRACS: &[f32] = &[0.10, 0.15, 0.25, 0.35, 0.50];

    let center = primitive_center(prim);
    if let Some(axis) = normalized_axis(witness.sample - center) {
        push_projection_split_candidates(
            out,
            seen,
            "error:sample_center",
            faces,
            mesh,
            axis,
            QUANTILES,
        );
    }
    if let Some(axis) = normalized_axis(witness.sample - witness.nearest) {
        push_projection_split_candidates(
            out,
            seen,
            "error:sample_nearest",
            faces,
            mesh,
            axis,
            QUANTILES,
        );
        push_threshold_split_candidate(
            out,
            seen,
            "error:sample_nearest_threshold",
            faces,
            mesh,
            axis,
            witness.nearest.coords.dot(&axis),
        );
    }
    if let Some(axis) = normalized_axis(witness.normal) {
        push_projection_split_candidates(out, seen, "error:normal", faces, mesh, axis, QUANTILES);
        push_threshold_split_candidate(
            out,
            seen,
            "error:normal_threshold",
            faces,
            mesh,
            axis,
            witness.nearest.coords.dot(&axis),
        );
    }
    if let Prim::Obb { axes, .. } | Prim::Prism { axes, .. } = prim {
        for (i, axis) in axes.iter().enumerate() {
            push_projection_split_candidates(
                out,
                seen,
                &format!("error:prim_axis{}", i),
                faces,
                mesh,
                *axis,
                QUANTILES,
            );
        }
    }

    // Localized candidates detach the face cluster nearest the protruding
    // primitive-surface sample or nearest supporting input point. These are
    // useful when the unsupported area is an L-shaped footprint that no
    // median projection cut separates cleanly.
    push_nearest_point_split_candidates(
        out,
        seen,
        "error:sample",
        faces,
        mesh,
        witness.sample,
        NEAR_FRACS,
    );
    push_nearest_point_split_candidates(
        out,
        seen,
        "error:nearest",
        faces,
        mesh,
        witness.nearest,
        NEAR_FRACS,
    );
    push_footprint_gap_split_candidates(out, seen, faces, mesh, prim, witness);
}

fn debug_live_primitive_splits(
    prims: &[Primitive],
    mesh: &Mesh,
    face_next: &[u32],
    bvh: &Bvh,
    enabled: PrimMask,
    threshold_frac: f32,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    face_fit_mask: Option<&[bool]>,
    face_fit_proxy_points: Option<&[Vec<Point3<f32>>]>,
    split_error_region: bool,
    target: u32,
    split_debug_rows: &mut Vec<SplitDebugRow>,
) {
    fn unavailable_row(
        target: u32,
        reason: &str,
        kind: prim::PrimKind,
        face_count: usize,
        original_hausdorff: f32,
        threshold: f32,
        volume: f32,
    ) -> SplitDebugRow {
        SplitDebugRow {
            pass: "debug_live",
            split_attempt: 0,
            primitive_root: target,
            compact_pid: target as usize,
            candidate_index: 0,
            source: format!("debug:unavailable:{}", reason),
            original_kind: kind,
            original_face_count: face_count,
            original_hausdorff,
            threshold,
            faces_a: 0,
            faces_b: 0,
            kind_a: kind,
            kind_b: kind,
            hausdorff_a: 0.0,
            hausdorff_b: 0.0,
            hausdorff_max: 0.0,
            delta_hausdorff_max: -original_hausdorff,
            volume_a: 0.0,
            volume_b: 0.0,
            volume_sum: 0.0,
            delta_volume_sum: -volume,
            would_improve: false,
            accepted: false,
            witness_dist: None,
            witness_sample: None,
            witness_nearest: None,
            witness_normal: None,
        }
    }

    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let target_usize = target as usize;
    if target_usize >= prims.len() {
        split_debug_rows.push(unavailable_row(
            target,
            "out_of_range",
            prim::PrimKind::Obb,
            0,
            0.0,
            threshold,
            0.0,
        ));
        return;
    }
    let p = &prims[target_usize];
    let original_hausdorff = if p.alive {
        local_hausdorff_dense(&p.prim, bvh, mesh)
    } else {
        0.0
    };
    if !p.alive {
        split_debug_rows.push(unavailable_row(
            target,
            "not_alive",
            p.prim.kind(),
            p.face_count as usize,
            original_hausdorff,
            threshold,
            p.volume,
        ));
        return;
    }
    if p.face_count < 2 || p.vertex_indices.len() < 4 {
        split_debug_rows.push(unavailable_row(
            target,
            "insufficient_faces_or_vertices",
            p.prim.kind(),
            p.face_count as usize,
            original_hausdorff,
            threshold,
            p.volume,
        ));
        return;
    }

    let faces: Vec<u32> = walk_faces(target, p.face_count, face_next).collect();
    if faces.len() < 2 {
        split_debug_rows.push(unavailable_row(
            target,
            "face_walk_too_short",
            p.prim.kind(),
            p.face_count as usize,
            original_hausdorff,
            threshold,
            p.volume,
        ));
        return;
    }
    let witness = local_hausdorff_dense_witness(&p.prim, bvh, mesh);
    let pts = gather_fit_points(
        &mesh.verts,
        &p.fit_vertex_indices,
        &p.vertex_indices,
        &p.fit_proxy_points,
    );
    let pca = pca_axes(&pts);
    let candidate_axes: [Vector3<f32>; 6] = [
        pca[0],
        pca[1],
        pca[2],
        Vector3::new(1.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0),
    ];
    let mut split_candidates: Vec<SplitCandidate> = Vec::new();
    let mut seen_candidates: HashSet<Vec<u32>> = HashSet::new();
    for (i, axis) in candidate_axes.iter().enumerate() {
        push_projection_split_candidates(
            &mut split_candidates,
            &mut seen_candidates,
            match i {
                0 => "baseline:pca0",
                1 => "baseline:pca1",
                2 => "baseline:pca2",
                3 => "baseline:world_x",
                4 => "baseline:world_y",
                _ => "baseline:world_z",
            },
            &faces,
            mesh,
            *axis,
            &[0.5],
        );
    }
    if split_error_region {
        if let Some(witness) = witness {
            push_error_region_split_candidates(
                &mut split_candidates,
                &mut seen_candidates,
                &faces,
                mesh,
                &p.prim,
                witness,
            );
        }
    }

    for (candidate_index, candidate) in split_candidates.into_iter().enumerate() {
        let faces_a = candidate.faces_a;
        let faces_b = candidate.faces_b;
        let mut new_a = refit_from_faces(
            &faces_a,
            mesh,
            enabled,
            bvh,
            tangent_eps,
            axis_override,
            face_fit_mask,
            face_fit_proxy_points,
        );
        let mut new_b = refit_from_faces(
            &faces_b,
            mesh,
            enabled,
            bvh,
            tangent_eps,
            axis_override,
            face_fit_mask,
            face_fit_proxy_points,
        );
        new_a.hausdorff = local_hausdorff_dense(&new_a.prim, bvh, mesh);
        new_b.hausdorff = local_hausdorff_dense(&new_b.prim, bvh, mesh);
        let new_max_h = new_a.hausdorff.max(new_b.hausdorff);
        split_debug_rows.push(SplitDebugRow {
            pass: "debug_live",
            split_attempt: 0,
            primitive_root: target,
            compact_pid: target_usize,
            candidate_index,
            source: candidate.source,
            original_kind: p.prim.kind(),
            original_face_count: faces.len(),
            original_hausdorff,
            threshold,
            faces_a: faces_a.len(),
            faces_b: faces_b.len(),
            kind_a: new_a.prim.kind(),
            kind_b: new_b.prim.kind(),
            hausdorff_a: new_a.hausdorff,
            hausdorff_b: new_b.hausdorff,
            hausdorff_max: new_max_h,
            delta_hausdorff_max: new_max_h - original_hausdorff,
            volume_a: new_a.volume,
            volume_b: new_b.volume,
            volume_sum: new_a.volume + new_b.volume,
            delta_volume_sum: (new_a.volume + new_b.volume) - p.volume,
            would_improve: new_max_h < original_hausdorff,
            accepted: false,
            witness_dist: witness.map(|w| w.dist),
            witness_sample: witness.map(|w| w.sample),
            witness_nearest: witness.map(|w| w.nearest),
            witness_normal: witness.map(|w| w.normal),
        });
    }
}

fn fit_synthetic_slab_part(
    points: [Point3<f32>; 3],
    base_axes: [Vector3<f32>; 3],
    axis_override: Option<&[Vector3<f32>; 3]>,
    mesh: &Mesh,
    bvh: &Bvh,
    threshold: f32,
    outside_threshold: Option<f32>,
    tangent_eps: f32,
    version: u64,
) -> (Primitive, f32) {
    let q = face_quadric(points[0], points[1], points[2], tangent_eps);
    let mut axes_candidates = Vec::with_capacity(4);
    axes_candidates.push(base_axes);
    axes_candidates.push(pca_axes(&points));
    axes_candidates.push(axes_from_q(q));
    if let Some(axes) = axis_override {
        axes_candidates.push(*axes);
    }

    let mut best_prim: Option<Prim> = None;
    let mut best_score = f32::INFINITY;
    for axes in axes_candidates {
        let prim_fit = prim::fit_obb(axes, &points);
        let h = local_hausdorff_repair(&prim_fit, bvh, mesh);
        let outside = if outside_threshold.is_some() {
            local_outside_repair(&prim_fit, bvh, mesh)
        } else {
            0.0
        };
        let score = combined_surface_error_score(h, outside, threshold, outside_threshold);
        if score < best_score {
            best_score = score;
            best_prim = Some(prim_fit);
        }
    }

    let prim_fit = best_prim.unwrap_or_else(|| prim::fit_obb(base_axes, &points));
    let volume = prim_fit.volume();
    (
        Primitive {
            alive: true,
            version,
            q,
            prim: prim_fit.clone(),
            volume,
            weighted_volume: prim_fit.weighted_volume(),
            face_count: 0,
            vertex_indices: Vec::new(),
            fit_vertex_indices: Vec::new(),
            fit_proxy_points: points.to_vec(),
            neighbors: HashSet::new(),
        },
        best_score,
    )
}

fn repair_single_face_slab_obbs(
    prims: &mut Vec<Primitive>,
    mesh: &Mesh,
    face_next: &[u32],
    bvh: &Bvh,
    threshold_frac: f32,
    max_repairs: usize,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    outside_threshold_frac: Option<f32>,
) -> usize {
    if max_repairs == 0 {
        return 0;
    }
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let outside_threshold = outside_threshold_frac.map(|frac| frac * mesh_diag);

    let mut ranked: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in prims.iter().enumerate().take(face_next.len()) {
        if !p.alive || p.face_count != 1 || !is_repairable_slab_prim(&p.prim, p.volume, mesh_diag) {
            continue;
        }
        let h = local_hausdorff_repair(&p.prim, bvh, mesh);
        let outside = if outside_threshold.is_some() {
            local_outside_repair(&p.prim, bvh, mesh)
        } else {
            0.0
        };
        let score = combined_surface_error_score(h, outside, threshold, outside_threshold);
        if score > 1.0 {
            ranked.push((idx, score));
        }
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut repairs = 0usize;
    for (idx, _) in ranked {
        if repairs >= max_repairs || idx >= prims.len() || idx >= face_next.len() {
            break;
        }
        if !prims[idx].alive
            || prims[idx].face_count != 1
            || !is_repairable_slab_prim(&prims[idx].prim, prims[idx].volume, mesh_diag)
        {
            continue;
        }
        let face = match walk_faces(idx as u32, prims[idx].face_count, face_next).next() {
            Some(face) => face as usize,
            None => continue,
        };
        if face >= mesh.tris.len() {
            continue;
        }
        let base_axes = match &prims[idx].prim {
            Prim::Obb { axes, .. } => *axes,
            _ => continue,
        };

        let original_h = local_hausdorff_repair(&prims[idx].prim, bvh, mesh);
        let original_outside = if outside_threshold.is_some() {
            local_outside_repair(&prims[idx].prim, bvh, mesh)
        } else {
            0.0
        };
        let original_score = combined_surface_error_score(
            original_h,
            original_outside,
            threshold,
            outside_threshold,
        );

        let tri = mesh.tris[face];
        let p0 = mesh.verts[tri[0] as usize];
        let p1 = mesh.verts[tri[1] as usize];
        let p2 = mesh.verts[tri[2] as usize];
        let m01 = Point3::from((p0.coords + p1.coords) * 0.5);
        let m12 = Point3::from((p1.coords + p2.coords) * 0.5);
        let m20 = Point3::from((p2.coords + p0.coords) * 0.5);
        let centroid = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);

        let split_options: [Vec<[Point3<f32>; 3]>; 5] = [
            vec![[p2, p0, m01], [p2, m01, p1]],
            vec![[p0, p1, m12], [p0, m12, p2]],
            vec![[p1, p2, m20], [p1, m20, p0]],
            vec![[p0, p1, centroid], [p1, p2, centroid], [p2, p0, centroid]],
            vec![
                [p0, m01, m20],
                [m01, p1, m12],
                [m20, m12, p2],
                [m01, m12, m20],
            ],
        ];

        let mut best_pieces: Option<(Vec<Primitive>, f32, f32)> = None;
        for parts in split_options {
            let mut pieces = Vec::with_capacity(parts.len());
            let mut max_score = 0.0f32;
            let mut volume_sum = 0.0f32;
            for part in parts {
                let (piece, score) = fit_synthetic_slab_part(
                    part,
                    base_axes,
                    axis_override,
                    mesh,
                    bvh,
                    threshold,
                    outside_threshold,
                    tangent_eps,
                    prims[idx].version + 1,
                );
                max_score = max_score.max(score);
                volume_sum += piece.volume;
                pieces.push(piece);
            }
            if volume_sum > prims[idx].volume * 1.35 {
                continue;
            }
            if max_score > original_score * 0.98 {
                continue;
            }
            if best_pieces
                .as_ref()
                .map(|(_, best_score, _)| max_score < *best_score)
                .unwrap_or(true)
            {
                best_pieces = Some((pieces, max_score, volume_sum));
            }
        }

        let Some((mut pieces, _, _)) = best_pieces else {
            continue;
        };
        let old_neighbors = prims[idx].neighbors.clone();
        let mut first = pieces.remove(0);
        first.neighbors = old_neighbors;
        prims[idx] = first;
        for mut piece in pieces {
            piece.version = 0;
            prims.push(piece);
        }
        repairs += 1;
    }

    repairs
}

fn repair_few_face_slab_obbs(
    prims: &mut Vec<Primitive>,
    mesh: &Mesh,
    face_next: &[u32],
    bvh: &Bvh,
    threshold_frac: f32,
    max_repairs: usize,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    outside_threshold_frac: Option<f32>,
) -> usize {
    if max_repairs == 0 {
        return 0;
    }
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let outside_threshold = outside_threshold_frac.map(|frac| frac * mesh_diag);

    let mut ranked: Vec<(usize, f32)> = Vec::new();
    for (idx, p) in prims.iter().enumerate().take(face_next.len()) {
        if !p.alive
            || p.face_count < 2
            || p.face_count > 3
            || !is_repairable_slab_prim(&p.prim, p.volume, mesh_diag)
        {
            continue;
        }
        let h = local_hausdorff_repair(&p.prim, bvh, mesh);
        let outside = if outside_threshold.is_some() {
            local_outside_repair(&p.prim, bvh, mesh)
        } else {
            0.0
        };
        let score = combined_surface_error_score(h, outside, threshold, outside_threshold);
        if score > 1.0 {
            ranked.push((idx, score));
        }
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut repairs = 0usize;
    for (idx, _) in ranked {
        if repairs >= max_repairs || idx >= prims.len() || idx >= face_next.len() {
            break;
        }
        if !prims[idx].alive
            || prims[idx].face_count < 2
            || prims[idx].face_count > 3
            || !is_repairable_slab_prim(&prims[idx].prim, prims[idx].volume, mesh_diag)
        {
            continue;
        }
        let faces: Vec<usize> = walk_faces(idx as u32, prims[idx].face_count, face_next)
            .map(|face| face as usize)
            .filter(|&face| face < mesh.tris.len())
            .collect();
        if faces.len() != prims[idx].face_count as usize {
            continue;
        }

        let base_axes = match &prims[idx].prim {
            Prim::Obb { axes, .. } => *axes,
            _ => continue,
        };
        let original_h = local_hausdorff_repair(&prims[idx].prim, bvh, mesh);
        let original_outside = if outside_threshold.is_some() {
            local_outside_repair(&prims[idx].prim, bvh, mesh)
        } else {
            0.0
        };
        let original_score = combined_surface_error_score(
            original_h,
            original_outside,
            threshold,
            outside_threshold,
        );

        let mut pieces = Vec::new();
        let mut max_score = 0.0f32;
        let mut volume_sum = 0.0f32;
        for face in faces {
            let tri = mesh.tris[face];
            let p0 = mesh.verts[tri[0] as usize];
            let p1 = mesh.verts[tri[1] as usize];
            let p2 = mesh.verts[tri[2] as usize];
            let m01 = Point3::from((p0.coords + p1.coords) * 0.5);
            let m12 = Point3::from((p1.coords + p2.coords) * 0.5);
            let m20 = Point3::from((p2.coords + p0.coords) * 0.5);
            let parts = [
                [p0, m01, m20],
                [m01, p1, m12],
                [m20, m12, p2],
                [m01, m12, m20],
            ];
            for part in parts {
                let (piece, score) = fit_synthetic_slab_part(
                    part,
                    base_axes,
                    axis_override,
                    mesh,
                    bvh,
                    threshold,
                    outside_threshold,
                    tangent_eps,
                    prims[idx].version + 1,
                );
                max_score = max_score.max(score);
                volume_sum += piece.volume;
                pieces.push(piece);
            }
        }
        if pieces.is_empty() || volume_sum > prims[idx].volume * 1.5 {
            continue;
        }
        if max_score > original_score * 0.98 {
            continue;
        }

        let old_neighbors = prims[idx].neighbors.clone();
        let mut first = pieces.remove(0);
        first.neighbors = old_neighbors;
        prims[idx] = first;
        for mut piece in pieces {
            piece.version = 0;
            prims.push(piece);
        }
        repairs += 1;
    }

    repairs
}

/// Post-merge split-worst pass. After greedy merge converges, repeatedly
/// find the live primitive with highest local Hausdorff > threshold ×
/// diag and split its face set along the longest PCA axis (median split).
/// Each accepted split increases primitive count by 1.
///
/// Targets the OBB-on-non-rectangular-planar-region failure mode that
/// the merge cost can't see (V(merge) ≈ V(p0) + V(p1) for two flat
/// slabs, even when the merged OBB's corners protrude metres past the
/// input outline). After feasibility rejection has done what it can,
/// the remaining drift comes from primitives whose vertex cloud is
/// bimodal — e.g. an L-shaped rooftop. Splitting along the longest
/// axis separates the modes and each half gets its own tight-fit OBB.
///
/// Acceptance: both halves must have strictly lower Hausdorff than the
/// original. Otherwise the split would produce two equally-bad pieces.
fn split_worst_primitives(
    prims: &mut Vec<Primitive>,
    mesh: &Mesh,
    adj: &Adjacency,
    dsu: &mut Dsu,
    face_next: &mut [u32],
    bvh: &Bvh,
    enabled: PrimMask,
    threshold_frac: f32,
    max_splits: usize,
    tangent_eps: f32,
    axis_override: Option<&[Vector3<f32>; 3]>,
    face_fit_mask: Option<&[bool]>,
    face_fit_proxy_points: Option<&[Vec<Point3<f32>>]>,
    split_error_region: bool,
    outside_threshold_frac: Option<f32>,
    split_debug: Option<SplitDebugOpts>,
    split_debug_rows: &mut Vec<SplitDebugRow>,
    pass: &'static str,
    mode: SplitPassMode,
) -> usize {
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let outside_threshold = outside_threshold_frac.map(|frac| frac * mesh_diag);
    let nf = mesh.tris.len();

    // Compact face → primitive id, mirroring rebalance_faces.
    let mut root_to_id: HashMap<u32, u32> = HashMap::new();
    let mut id_to_root: Vec<u32> = Vec::new();
    let mut face_assignment: Vec<u32> = Vec::with_capacity(nf);
    for f in 0..nf {
        let root = dsu.find(f as u32);
        let id = match root_to_id.get(&root) {
            Some(&id) => id,
            None => {
                let id = root_to_id.len() as u32;
                root_to_id.insert(root, id);
                id_to_root.push(root);
                id
            }
        };
        face_assignment.push(id);
    }
    let mut n_prims = root_to_id.len();
    let mut prim_faces: Vec<Vec<u32>> = vec![Vec::new(); n_prims];
    for (fi, &pid) in face_assignment.iter().enumerate() {
        prim_faces[pid as usize].push(fi as u32);
    }

    // Refit each primitive so we have a current Hausdorff to rank by.
    // We replace the cached `hausdorff` with the dense variant — the
    // 24-sample one misses drift on rectangular OBBs sitting on
    // non-rectangular planar regions (worst point is often a face- or
    // edge-interior, not a vertex), and we'd rather pay 10× sampling
    // cost on the few worst primitives than split the wrong ones.
    let mut state: Vec<RebalanceState> = (0..n_prims)
        .map(|pid| {
            let mut s = refit_from_faces(
                &prim_faces[pid],
                mesh,
                enabled,
                bvh,
                tangent_eps,
                axis_override,
                face_fit_mask,
                face_fit_proxy_points,
            );
            if mode == SplitPassMode::SlabCornerRepair {
                let p = &prims[id_to_root[pid] as usize];
                if p.alive {
                    s.prim = p.prim.clone();
                    s.q = p.q;
                    s.volume = p.volume;
                    s.weighted_volume = p.weighted_volume;
                    s.vertex_indices = p.vertex_indices.clone();
                    s.fit_vertex_indices = p.fit_vertex_indices.clone();
                    s.fit_proxy_points = p.fit_proxy_points.clone();
                }
            }
            s.hausdorff = if mode == SplitPassMode::SlabCornerRepair {
                local_hausdorff_repair(&s.prim, bvh, mesh)
            } else {
                local_hausdorff_dense(&s.prim, bvh, mesh)
            };
            s
        })
        .collect();
    let mut outside_error: Vec<f32> = if outside_threshold.is_some() {
        state
            .iter()
            .map(|s| {
                if mode == SplitPassMode::SlabCornerRepair {
                    local_outside_repair(&s.prim, bvh, mesh)
                } else {
                    local_outside_dense(&s.prim, bvh, mesh)
                }
            })
            .collect()
    } else {
        vec![0.0; n_prims]
    };
    // `tried` flags primitives we've already attempted to split (and
    // rejected) so the worst-search loop doesn't re-pick them every
    // iteration. Reset on a successful split since a primitive's faces
    // may have been replaced wholesale via an earlier split's
    // re-fit (extremely unlikely but cheap to be safe).
    let mut tried: Vec<bool> = vec![false; n_prims];

    let mut splits_done = 0usize;
    while splits_done < max_splits {
        let mut worst_pid: Option<usize> = None;
        let mut worst_score = 1.0f32;
        for (pid, s) in state.iter().enumerate() {
            if tried[pid] || prim_faces[pid].len() < 2 {
                continue;
            }
            if mode == SplitPassMode::SlabCornerRepair && !is_medium_slab_obb(s, mesh_diag) {
                continue;
            }
            let score = combined_surface_error_score(
                s.hausdorff,
                outside_error[pid],
                threshold,
                outside_threshold,
            );
            if score > worst_score {
                worst_score = score;
                worst_pid = Some(pid);
            }
        }
        let mut debug_only = false;
        let pid = match worst_pid {
            Some(p) => p,
            None => {
                let forced = split_debug
                    .and_then(|debug| debug.primitive)
                    .and_then(|target| {
                        id_to_root.iter().enumerate().find_map(|(pid, &root)| {
                            let owns_target_face = (target as usize) < face_assignment.len()
                                && face_assignment[target as usize] == pid as u32;
                            if (root == target || owns_target_face)
                                && !tried[pid]
                                && prim_faces[pid].len() >= 2
                            {
                                Some(pid)
                            } else {
                                None
                            }
                        })
                    });
                match forced {
                    Some(p) => {
                        debug_only = true;
                        p
                    }
                    None => break,
                }
            }
        };

        // Run PCA on the primitive's vertex cloud. PCA's longest axis is
        // typically the right split direction, but on L-shaped planar
        // regions (the dominant failure case) the longest axis is
        // sometimes the L's diagonal, which median-split cuts at 45°
        // and does not separate the L's arms. Try splits along all 3
        // PCA axes plus the 3 world axes; pick the one that produces
        // the lowest max-Hausdorff between the two halves.
        let pts = gather_fit_points(
            &mesh.verts,
            &state[pid].fit_vertex_indices,
            &state[pid].vertex_indices,
            &state[pid].fit_proxy_points,
        );
        if pts.len() < 4 {
            tried[pid] = true;
            continue;
        }
        let pca = pca_axes(&pts);
        let candidate_axes: [Vector3<f32>; 6] = [
            pca[0],
            pca[1],
            pca[2],
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ];
        let primitive_root = id_to_root[pid];
        let debug_this = split_debug
            .map(|debug| {
                debug
                    .primitive
                    .map(|id| {
                        primitive_root == id
                            || ((id as usize) < face_assignment.len()
                                && face_assignment[id as usize] == pid as u32)
                    })
                    .unwrap_or(true)
            })
            .unwrap_or(false);
        let use_error_region = split_error_region || mode == SplitPassMode::SlabCornerRepair;
        let witness = if use_error_region || debug_this {
            if mode == SplitPassMode::SlabCornerRepair {
                let h_score = state[pid].hausdorff / threshold.max(1e-6);
                let outside_score = outside_threshold
                    .map(|threshold| outside_error[pid] / threshold.max(1e-6))
                    .unwrap_or(0.0);
                if outside_score >= h_score {
                    local_distance_repair_witness(
                        &state[pid].prim,
                        bvh,
                        mesh,
                        DistanceMode::Outside,
                    )
                    .or_else(|| local_hausdorff_repair_witness(&state[pid].prim, bvh, mesh))
                } else {
                    local_hausdorff_repair_witness(&state[pid].prim, bvh, mesh)
                }
            } else {
                local_hausdorff_dense_witness(&state[pid].prim, bvh, mesh)
            }
        } else {
            None
        };

        let mut split_candidates: Vec<SplitCandidate> = Vec::new();
        let mut seen_candidates: HashSet<Vec<u32>> = HashSet::new();
        for (i, axis) in candidate_axes.iter().enumerate() {
            push_projection_split_candidates(
                &mut split_candidates,
                &mut seen_candidates,
                match i {
                    0 => "baseline:pca0",
                    1 => "baseline:pca1",
                    2 => "baseline:pca2",
                    3 => "baseline:world_x",
                    4 => "baseline:world_y",
                    _ => "baseline:world_z",
                },
                &prim_faces[pid],
                mesh,
                *axis,
                &[0.5],
            );
        }
        if use_error_region {
            if let Some(witness) = witness {
                push_error_region_split_candidates(
                    &mut split_candidates,
                    &mut seen_candidates,
                    &prim_faces[pid],
                    mesh,
                    &state[pid].prim,
                    witness,
                );
                if mode == SplitPassMode::SlabCornerRepair {
                    push_slab_corner_split_candidates(
                        &mut split_candidates,
                        &mut seen_candidates,
                        &prim_faces[pid],
                        mesh,
                        &state[pid].prim,
                        witness,
                    );
                }
            }
        }

        let mut best_split: Option<(
            RebalanceState,
            RebalanceState,
            Vec<u32>,
            Vec<u32>,
            f32,
            f32,
            Option<usize>,
        )> = None;
        let mut best_score = combined_surface_error_score(
            state[pid].hausdorff,
            outside_error[pid],
            threshold,
            outside_threshold,
        );

        for (candidate_index, candidate) in split_candidates.into_iter().enumerate() {
            let faces_a = candidate.faces_a;
            let faces_b = candidate.faces_b;
            if mode == SplitPassMode::SlabCornerRepair && (faces_a.len() < 2 || faces_b.len() < 2) {
                continue;
            }
            let mut new_a = refit_from_faces(
                &faces_a,
                mesh,
                enabled,
                bvh,
                tangent_eps,
                axis_override,
                face_fit_mask,
                face_fit_proxy_points,
            );
            let mut new_b = refit_from_faces(
                &faces_b,
                mesh,
                enabled,
                bvh,
                tangent_eps,
                axis_override,
                face_fit_mask,
                face_fit_proxy_points,
            );
            new_a.hausdorff = if mode == SplitPassMode::SlabCornerRepair {
                local_hausdorff_repair(&new_a.prim, bvh, mesh)
            } else {
                local_hausdorff_dense(&new_a.prim, bvh, mesh)
            };
            new_b.hausdorff = if mode == SplitPassMode::SlabCornerRepair {
                local_hausdorff_repair(&new_b.prim, bvh, mesh)
            } else {
                local_hausdorff_dense(&new_b.prim, bvh, mesh)
            };
            let new_a_outside = if outside_threshold.is_some() {
                if mode == SplitPassMode::SlabCornerRepair {
                    local_outside_repair(&new_a.prim, bvh, mesh)
                } else {
                    local_outside_dense(&new_a.prim, bvh, mesh)
                }
            } else {
                0.0
            };
            let new_b_outside = if outside_threshold.is_some() {
                if mode == SplitPassMode::SlabCornerRepair {
                    local_outside_repair(&new_b.prim, bvh, mesh)
                } else {
                    local_outside_dense(&new_b.prim, bvh, mesh)
                }
            } else {
                0.0
            };
            let new_max_h = new_a.hausdorff.max(new_b.hausdorff);
            let new_score = combined_surface_error_score(
                new_a.hausdorff,
                new_a_outside,
                threshold,
                outside_threshold,
            )
            .max(combined_surface_error_score(
                new_b.hausdorff,
                new_b_outside,
                threshold,
                outside_threshold,
            ));
            let debug_row_idx = if debug_this {
                let row = SplitDebugRow {
                    pass,
                    split_attempt: splits_done,
                    primitive_root,
                    compact_pid: pid,
                    candidate_index,
                    source: candidate.source,
                    original_kind: state[pid].prim.kind(),
                    original_face_count: prim_faces[pid].len(),
                    original_hausdorff: state[pid].hausdorff,
                    threshold,
                    faces_a: faces_a.len(),
                    faces_b: faces_b.len(),
                    kind_a: new_a.prim.kind(),
                    kind_b: new_b.prim.kind(),
                    hausdorff_a: new_a.hausdorff,
                    hausdorff_b: new_b.hausdorff,
                    hausdorff_max: new_max_h,
                    delta_hausdorff_max: new_max_h - state[pid].hausdorff,
                    volume_a: new_a.volume,
                    volume_b: new_b.volume,
                    volume_sum: new_a.volume + new_b.volume,
                    delta_volume_sum: (new_a.volume + new_b.volume) - state[pid].volume,
                    would_improve: new_score < best_score,
                    accepted: false,
                    witness_dist: witness.map(|w| w.dist),
                    witness_sample: witness.map(|w| w.sample),
                    witness_nearest: witness.map(|w| w.nearest),
                    witness_normal: witness.map(|w| w.normal),
                };
                split_debug_rows.push(row);
                Some(split_debug_rows.len() - 1)
            } else {
                None
            };
            if new_score + 1e-6 < best_score {
                if mode == SplitPassMode::SlabCornerRepair {
                    let volume_sum = new_a.volume + new_b.volume;
                    if volume_sum > state[pid].volume * 1.35 {
                        continue;
                    }
                    if new_score > best_score * 0.98 {
                        continue;
                    }
                }
                best_score = new_score;
                best_split = Some((
                    new_a,
                    new_b,
                    faces_a,
                    faces_b,
                    new_a_outside,
                    new_b_outside,
                    debug_row_idx,
                ));
            }
        }

        if debug_only {
            tried[pid] = true;
            continue;
        }

        let (new_a, new_b, faces_a, faces_b, new_a_outside, new_b_outside, debug_row_idx) =
            match best_split {
                Some(s) => s,
                None => {
                    tried[pid] = true;
                    continue;
                }
            };
        if let Some(row_idx) = debug_row_idx {
            if let Some(row) = split_debug_rows.get_mut(row_idx) {
                row.accepted = true;
            }
        }

        // Accept: keep current pid for half A, allocate a new pid for B.
        let root_a = faces_a[0];
        let root_b = faces_b[0];
        let new_pid = n_prims as u32;
        for &fi in &faces_b {
            face_assignment[fi as usize] = new_pid;
        }
        prim_faces[pid] = faces_a;
        prim_faces.push(faces_b);
        id_to_root[pid] = root_a;
        id_to_root.push(root_b);
        state[pid] = new_a;
        state.push(new_b);
        outside_error[pid] = new_a_outside;
        outside_error.push(new_b_outside);
        tried.push(false);
        n_prims += 1;
        splits_done += 1;
    }

    if splits_done == 0 {
        return 0;
    }

    // Rebuild dsu + face_next + prims from the final face assignment.
    // Same pattern as rebalance_faces.
    *dsu = Dsu::new(nf);
    for pid in 0..n_prims {
        let faces = &prim_faces[pid];
        if faces.is_empty() {
            continue;
        }
        let root = faces[0];
        for &f in &faces[1..] {
            dsu.link(root, f);
        }
        for i in 0..faces.len() {
            let next = if i + 1 < faces.len() {
                faces[i + 1]
            } else {
                faces[0]
            };
            face_next[faces[i] as usize] = next;
        }
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
            fit_vertex_indices: s.fit_vertex_indices.clone(),
            fit_proxy_points: s.fit_proxy_points.clone(),
            neighbors: HashSet::new(),
        };
    }
    for fi in 0..nf {
        if dsu.find(fi as u32) != fi as u32 {
            prims[fi].alive = false;
            prims[fi].vertex_indices = Vec::new();
            prims[fi].fit_vertex_indices = Vec::new();
            prims[fi].fit_proxy_points = Vec::new();
            prims[fi].face_count = 0;
            prims[fi].neighbors.clear();
        }
    }
    for f in 0..nf {
        let p = dsu.find(f as u32);
        for &nf_idx in &adj.neighbors[f] {
            let q = dsu.find(nf_idx);
            if p != q {
                prims[p as usize].neighbors.insert(q);
            }
        }
    }

    splits_done
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

fn prim_aabb_extents(prim: &Prim) -> Vector3<f32> {
    let (lo, hi) = prim::world_aabb(prim);
    Vector3::new(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2])
}

fn prim_aabb_distance(a: &Prim, b: &Prim) -> f32 {
    let (alo, ahi) = prim::world_aabb(a);
    let (blo, bhi) = prim::world_aabb(b);
    let mut d2 = 0.0f32;
    for i in 0..3 {
        let gap = if ahi[i] < blo[i] {
            blo[i] - ahi[i]
        } else if bhi[i] < alo[i] {
            alo[i] - bhi[i]
        } else {
            0.0
        };
        d2 += gap * gap;
    }
    d2.sqrt()
}

fn prim_debug_samples(prim: &Prim) -> Vec<Point3<f32>> {
    let (verts, tris) = prim::tessellate(prim);
    let mut samples: Vec<Point3<f32>> = Vec::with_capacity(verts.len() + tris.len().min(48));
    for v in &verts {
        samples.push(Point3::new(v[0], v[1], v[2]));
    }
    let stride = (tris.len() / 48).max(1);
    for (i, t) in tris.iter().enumerate() {
        if i % stride != 0 {
            continue;
        }
        let a = verts[t[0] as usize];
        let b = verts[t[1] as usize];
        let c = verts[t[2] as usize];
        samples.push(Point3::new(
            (a[0] + b[0] + c[0]) / 3.0,
            (a[1] + b[1] + c[1]) / 3.0,
            (a[2] + b[2] + c[2]) / 3.0,
        ));
        if samples.len() >= verts.len() + 48 {
            break;
        }
    }
    samples
}

fn collision_simplify_primitives(
    prims: &mut [Primitive],
    mesh_diag: f32,
    threshold_frac: f32,
) -> usize {
    if threshold_frac <= 0.0 || !threshold_frac.is_finite() {
        return 0;
    }
    let tol = threshold_frac * mesh_diag.max(1e-6);
    let mut live: Vec<usize> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive)
        .map(|(i, _)| i)
        .collect();
    live.sort_by(|&a, &b| {
        prims[a]
            .volume
            .partial_cmp(&prims[b].volume)
            .unwrap_or(Ordering::Equal)
    });

    let mut to_drop = Vec::new();
    for &a in &live {
        if !prims[a].alive || to_drop.contains(&a) {
            continue;
        }
        let ext = prim_aabb_extents(&prims[a].prim);
        let mut dims = [ext.x.abs(), ext.y.abs(), ext.z.abs()];
        dims.sort_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal));

        // Detail primitives are either physically thin, or small enough in
        // their second dimension that they read as trim/bands/bevels rather
        // than structural collision. Broad walls/floors generally fail the
        // support test below because there is no larger nearby primitive
        // containing most of their surface within tolerance.
        let is_detail_like = dims[0] <= tol || dims[1] <= tol * 2.0;
        if !is_detail_like {
            continue;
        }

        let samples = prim_debug_samples(&prims[a].prim);
        if samples.is_empty() {
            continue;
        }
        for &b in live.iter().rev() {
            if a == b || !prims[b].alive || to_drop.contains(&b) {
                continue;
            }
            if prims[b].volume < prims[a].volume * 2.0 {
                continue;
            }
            if prim_aabb_distance(&prims[a].prim, &prims[b].prim) > tol {
                continue;
            }
            let supported = samples
                .iter()
                .filter(|&&q| prims[b].prim.contains(q, tol))
                .count();
            let frac = supported as f32 / samples.len() as f32;
            if frac >= 0.65 {
                to_drop.push(a);
                break;
            }
        }
    }

    for &i in &to_drop {
        prims[i].alive = false;
        prims[i].vertex_indices.clear();
        prims[i].fit_vertex_indices.clear();
        prims[i].fit_proxy_points.clear();
        prims[i].face_count = 0;
        prims[i].neighbors.clear();
    }
    to_drop.len()
}

fn push_post_budget_pairs_for(
    prims: &[Primitive],
    mesh_verts: &[Point3<f32>],
    pq: &mut BinaryHeap<PqEntry>,
    idx: usize,
    enabled: PrimMask,
    axis_override: Option<&[Vector3<f32>; 3]>,
    collision_simplify: bool,
) {
    if idx >= prims.len() || !prims[idx].alive {
        return;
    }
    for (j, p) in prims.iter().enumerate() {
        if j == idx || !p.alive {
            continue;
        }
        let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
            &prims[idx],
            p,
            mesh_verts,
            enabled,
            axis_override,
            collision_simplify,
        );
        pq.push(PqEntry {
            cost: wvol - (prims[idx].weighted_volume + p.weighted_volume),
            a: idx as u32,
            b: j as u32,
            va: prims[idx].version,
            vb: p.version,
            prim: prim_fit,
            volume: vol,
            weighted_volume: wvol,
        });
    }
}

fn build_face_reverse_samples(mesh: &Mesh) -> Vec<[Point3<f32>; 4]> {
    mesh.tris
        .iter()
        .map(|tri| {
            let p0 = mesh.verts[tri[0] as usize];
            let p1 = mesh.verts[tri[1] as usize];
            let p2 = mesh.verts[tri[2] as usize];
            [
                Point3::from((p0.coords + p1.coords + p2.coords) / 3.0),
                Point3::from((p0.coords + p1.coords) * 0.5),
                Point3::from((p1.coords + p2.coords) * 0.5),
                Point3::from((p2.coords + p0.coords) * 0.5),
            ]
        })
        .collect()
}

fn analytic_prim_surface_distance(prim: &Prim, p: Point3<f32>) -> Option<f32> {
    match *prim {
        Prim::Obb {
            center,
            axes,
            half_extents,
        } => {
            let d = p - center;
            let mut outside_sq = 0.0f32;
            let mut inside_margin = f32::INFINITY;
            for i in 0..3 {
                let a = axes[i].dot(&d).abs();
                let over = a - half_extents[i];
                if over > 0.0 {
                    outside_sq += over * over;
                } else {
                    inside_margin = inside_margin.min(-over);
                }
            }
            if outside_sq > 0.0 {
                Some(outside_sq.sqrt())
            } else {
                Some(inside_margin)
            }
        }
        Prim::Sphere { center, r } => Some(((p - center).norm() - r).abs()),
        Prim::Cylinder { center, axis, h, r } => {
            let d = p - center;
            let ax = axis.dot(&d);
            let radial = (d - axis * ax).norm();
            let qx = radial - r;
            let qy = ax.abs() - h * 0.5;
            let ox = qx.max(0.0);
            let oy = qy.max(0.0);
            let outside = (ox * ox + oy * oy).sqrt();
            let inside = qx.max(qy).min(0.0);
            Some((outside + inside).abs())
        }
        Prim::Capsule { center, axis, h, r } => {
            let d = p - center;
            let ax = axis.dot(&d).clamp(-h * 0.5, h * 0.5);
            Some(((d - axis * ax).norm() - r).abs())
        }
        Prim::Frustum { .. } | Prim::Prism { .. } => None,
    }
}

fn tessellated_prim_surface_distance(prim: &Prim, points: &[Point3<f32>], stop_above: f32) -> f32 {
    let (verts_raw, tris) = prim::tessellate(prim);
    if verts_raw.is_empty() || tris.is_empty() {
        return 0.0;
    }
    if verts_raw
        .iter()
        .any(|v| !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite())
    {
        return 0.0;
    }
    let verts: Vec<Point3<f32>> = verts_raw
        .into_iter()
        .map(|v| Point3::new(v[0], v[1], v[2]))
        .collect();
    let bvh = Bvh::build(&verts, &tris);
    let mut max_d = 0.0f32;
    for &p in points {
        let (_pt, _n, signed) = bvh.nearest_face(&verts, &tris, p);
        max_d = max_d.max(signed.abs());
        if max_d > stop_above {
            break;
        }
    }
    max_d
}

fn local_reverse_mesh_error(
    prim: &Prim,
    faces_a: &[u32],
    faces_b: &[u32],
    proxy_a: &[Point3<f32>],
    proxy_b: &[Point3<f32>],
    face_samples: &[[Point3<f32>; 4]],
    stop_above: f32,
) -> f32 {
    let mut points =
        Vec::with_capacity((faces_a.len() + faces_b.len()) * 4 + proxy_a.len() + proxy_b.len());
    for &fi in faces_a.iter().chain(faces_b.iter()) {
        if let Some(samples) = face_samples.get(fi as usize) {
            points.extend_from_slice(samples);
        }
    }
    points.extend_from_slice(proxy_a);
    points.extend_from_slice(proxy_b);
    if points.is_empty() {
        return 0.0;
    }
    if analytic_prim_surface_distance(prim, points[0]).is_none() {
        return tessellated_prim_surface_distance(prim, &points, stop_above);
    }
    let mut max_d = 0.0f32;
    for p in points {
        if let Some(d) = analytic_prim_surface_distance(prim, p) {
            max_d = max_d.max(d);
            if max_d > stop_above {
                break;
            }
        }
    }
    max_d
}

fn post_merge_to_budget(
    prims: &mut Vec<Primitive>,
    mesh: &Mesh,
    face_next: &[u32],
    bvh: &Bvh,
    enabled: PrimMask,
    target: usize,
    threshold_frac: f32,
    outside_threshold_frac: Option<f32>,
    axis_override: Option<&[Vector3<f32>; 3]>,
    collision_simplify: bool,
) -> usize {
    let mut live_count = prims.iter().filter(|p| p.alive).count();
    if target == 0 || live_count <= target {
        return 0;
    }
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let outside_threshold = outside_threshold_frac.map(|frac| frac * mesh_diag);
    let face_samples = build_face_reverse_samples(mesh);
    let mut prim_faces: Vec<Vec<u32>> = prims
        .iter()
        .enumerate()
        .map(|(idx, p)| {
            if p.alive && p.face_count > 0 && idx < face_next.len() {
                walk_faces(idx as u32, p.face_count, face_next).collect()
            } else {
                Vec::new()
            }
        })
        .collect();
    let mut pq = BinaryHeap::new();
    let live: Vec<usize> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive)
        .map(|(i, _)| i)
        .collect();
    for (pos, &i) in live.iter().enumerate() {
        for &j in &live[(pos + 1)..] {
            let (_q, prim_fit, vol, wvol, _vidx, _fit_vidx) = merge_pair(
                &prims[i],
                &prims[j],
                &mesh.verts,
                enabled,
                axis_override,
                collision_simplify,
            );
            pq.push(PqEntry {
                cost: wvol - (prims[i].weighted_volume + prims[j].weighted_volume),
                a: i as u32,
                b: j as u32,
                va: prims[i].version,
                vb: prims[j].version,
                prim: prim_fit,
                volume: vol,
                weighted_volume: wvol,
            });
        }
    }

    let mut accepted = 0usize;
    while live_count > target {
        let Some(entry) = pq.pop() else {
            break;
        };
        let a = entry.a as usize;
        let b = entry.b as usize;
        if a >= prims.len()
            || b >= prims.len()
            || !prims[a].alive
            || !prims[b].alive
            || prims[a].version != entry.va
            || prims[b].version != entry.vb
        {
            continue;
        }

        let h = local_hausdorff_dense(&entry.prim, bvh, mesh);
        let outside = if outside_threshold.is_some() {
            local_outside_dense(&entry.prim, bvh, mesh)
        } else {
            0.0
        };
        let score = combined_surface_error_score(h, outside, threshold, outside_threshold);
        if score > 1.0 {
            continue;
        }
        let reverse = local_reverse_mesh_error(
            &entry.prim,
            &prim_faces[a],
            &prim_faces[b],
            &prims[a].fit_proxy_points,
            &prims[b].fit_proxy_points,
            &face_samples,
            threshold,
        );
        if reverse > threshold {
            continue;
        }

        let new_q = prims[a].q + prims[b].q;
        let new_vidx = merge_sorted_unique(&prims[a].vertex_indices, &prims[b].vertex_indices);
        let new_fit_vidx =
            merge_sorted_unique(&prims[a].fit_vertex_indices, &prims[b].fit_vertex_indices);
        let new_fit_proxy_points =
            merge_fit_proxy_points(&prims[a].fit_proxy_points, &prims[b].fit_proxy_points);
        let mut new_neighbors: HashSet<u32> = prims[a]
            .neighbors
            .iter()
            .chain(prims[b].neighbors.iter())
            .copied()
            .collect();
        new_neighbors.remove(&(a as u32));
        new_neighbors.remove(&(b as u32));

        prims[a].q = new_q;
        prims[a].prim = entry.prim;
        prims[a].volume = entry.volume;
        prims[a].weighted_volume = entry.weighted_volume;
        prims[a].face_count += prims[b].face_count;
        prims[a].vertex_indices = new_vidx;
        prims[a].fit_vertex_indices = new_fit_vidx;
        prims[a].fit_proxy_points = new_fit_proxy_points;
        prims[a].neighbors = new_neighbors;
        prims[a].version += 1;
        let mut b_faces = std::mem::take(&mut prim_faces[b]);
        prim_faces[a].append(&mut b_faces);

        prims[b].alive = false;
        prims[b].neighbors.clear();
        prims[b].vertex_indices.clear();
        prims[b].fit_vertex_indices.clear();
        prims[b].fit_proxy_points.clear();
        prims[b].face_count = 0;

        live_count -= 1;
        accepted += 1;
        push_post_budget_pairs_for(
            prims,
            &mesh.verts,
            &mut pq,
            a,
            enabled,
            axis_override,
            collision_simplify,
        );
    }

    accepted
}

/// Paper appendix Fig 22 postprocess: drop OBBs whose smallest half-extent
/// is ≤ `threshold_frac × mesh_diag`. The paper's Bistro recipe is
/// `threshold_frac = 1e-4`. Distinct from `is_pancake` (which is a
/// merge-time penalty keyed off the MIN_HALF_EXTENT clamp): this runs
/// after the merge fully converges and uses an absolute fraction of the
/// mesh diag, catching slabs that didn't quite hit the clamp but still
/// have surface drifting many metres from the input. OBBs only — Prisms
/// can have legitimately tiny `hzt` (gable roof tip), capsules with tiny
/// radius are valid cables, etc.
fn strip_thin_obbs(
    prims: &mut [Primitive],
    mesh_verts: &[Point3<f32>],
    threshold_frac: f32,
) -> usize {
    if mesh_verts.is_empty() || threshold_frac <= 0.0 {
        return 0;
    }
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for v in mesh_verts {
        for i in 0..3 {
            if v[i] < lo[i] {
                lo[i] = v[i];
            }
            if v[i] > hi[i] {
                hi[i] = v[i];
            }
        }
    }
    let dx = hi[0] - lo[0];
    let dy = hi[1] - lo[1];
    let dz = hi[2] - lo[2];
    let diag = (dx * dx + dy * dy + dz * dz).sqrt();
    let tol = (diag.max(1.0)) * threshold_frac;

    let mut dropped = 0usize;
    for p in prims.iter_mut() {
        if !p.alive {
            continue;
        }
        if let Prim::Obb { half_extents, .. } = &p.prim {
            let min_h = half_extents[0].min(half_extents[1]).min(half_extents[2]);
            if min_h <= tol {
                p.alive = false;
                dropped += 1;
            }
        }
    }
    dropped
}

fn shrink_high_error_obbs(
    prims: &mut [Primitive],
    mesh: &Mesh,
    bvh: &Bvh,
    threshold_frac: f32,
) -> usize {
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let mut adjusted = 0usize;
    for p in prims.iter_mut() {
        if !p.alive {
            continue;
        }
        let Prim::Obb {
            center,
            axes,
            half_extents,
        } = p.prim.clone()
        else {
            continue;
        };
        let mut best_h = local_hausdorff_dense(&p.prim, bvh, mesh)
            .max(local_hausdorff_repair(&p.prim, bvh, mesh));
        if best_h <= threshold {
            continue;
        }

        let mut best_extents = half_extents;
        let mut improved = false;
        for _ in 0..5 {
            let mut iter_best_h = best_h;
            let mut iter_best_extents = best_extents;
            for axis in 0..3 {
                for factor in [0.998f32, 0.995, 0.99] {
                    let mut trial_extents = best_extents;
                    trial_extents[axis] = (trial_extents[axis] * factor).max(1e-4);
                    let trial = Prim::Obb {
                        center,
                        axes,
                        half_extents: trial_extents,
                    };
                    let h = local_hausdorff_dense(&trial, bvh, mesh)
                        .max(local_hausdorff_repair(&trial, bvh, mesh));
                    if h + 1e-5 < iter_best_h {
                        iter_best_h = h;
                        iter_best_extents = trial_extents;
                    }
                }
            }
            if iter_best_h + 1e-5 < best_h {
                best_h = iter_best_h;
                best_extents = iter_best_extents;
                improved = true;
            } else {
                break;
            }
        }

        if improved {
            p.prim = Prim::Obb {
                center,
                axes,
                half_extents: best_extents,
            };
            p.volume = p.prim.volume();
            p.weighted_volume = p.prim.weighted_volume();
            p.version += 1;
            adjusted += 1;
        }
    }
    adjusted
}

/// Park & Sung 2024-inspired refine. After merge converges, hill-climb
/// the orientation of each high-Hausdorff OBB primitive: try small
/// rotations around each principal axis, refit half-extents to subsumed
/// vertices in the new frame, accept the rotation that minimises local
/// Hausdorff. Step adapts (start 15°, halve on no-improvement, stop at
/// <1°). All subsumed verts stay enclosed by construction (refit is the
/// tight AABB in the new axis frame).
///
/// Only OBBs are refined — for cylinders/capsules/prisms the orientation
/// has additional structure (axial direction matters) and a generic
/// rotation perturbation can break the fit. OBB is the dominant primitive
/// type by count in our outputs anyway, so this covers most of the cost.
fn refine_search_pass(
    prims: &mut [Primitive],
    mesh: &Mesh,
    bvh: &Bvh,
    threshold_frac: f32,
    max_iters_per_prim: usize,
    outside_threshold_frac: Option<f32>,
) -> usize {
    let mesh_diag = crate::mesh::aabb_diag(&mesh.verts).max(1e-6);
    let threshold = threshold_frac * mesh_diag;
    let outside_threshold = outside_threshold_frac.map(|frac| frac * mesh_diag);

    // Worst-first ordering — fix the primitives that contribute most to
    // the global Hausdorff / outside-space violation first.
    let mut candidates: Vec<(usize, f32)> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive && matches!(p.prim, Prim::Obb { .. }))
        .map(|(i, p)| {
            let h = local_hausdorff_dense(&p.prim, bvh, mesh);
            let outside = if outside_threshold.is_some() {
                local_outside_dense(&p.prim, bvh, mesh)
            } else {
                0.0
            };
            (
                i,
                combined_surface_error_score(h, outside, threshold, outside_threshold),
            )
        })
        .filter(|&(_, score)| score > 1.0)
        .collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut total_iters = 0usize;
    for (idx, _) in candidates {
        let p = &prims[idx];
        let (mut axes, _, _) = match &p.prim {
            Prim::Obb {
                center,
                axes,
                half_extents,
            } => (*axes, *center, *half_extents),
            _ => continue,
        };
        let pts = gather_fit_points(
            &mesh.verts,
            &p.fit_vertex_indices,
            &p.vertex_indices,
            &p.fit_proxy_points,
        );
        if pts.len() < 2 {
            continue;
        }

        let refit =
            |axes: &[Vector3<f32>; 3], pts: &[Point3<f32>]| -> Prim { prim::fit_obb(*axes, pts) };
        let score_prim = |prim: &Prim| -> (f32, f32, f32) {
            let h = local_hausdorff_dense(prim, bvh, mesh);
            let outside = if outside_threshold.is_some() {
                local_outside_dense(prim, bvh, mesh)
            } else {
                0.0
            };
            (
                combined_surface_error_score(h, outside, threshold, outside_threshold),
                h,
                outside,
            )
        };

        let mut best_prim = refit(&axes, &pts);
        let (mut best_score, _, _) = score_prim(&best_prim);

        let mut step_deg = 15.0f32;
        let stop_deg = 1.0f32;
        let mut iters_used = 0usize;
        while step_deg >= stop_deg && iters_used < max_iters_per_prim {
            let step_rad = step_deg.to_radians();
            let mut improved = false;
            for axis_k in 0..3 {
                for sign in [-1.0f32, 1.0] {
                    let theta = sign * step_rad;
                    let new_axes = rotate_axes_about(axes, axis_k, theta);
                    let cand_prim = refit(&new_axes, &pts);
                    let (cand_score, _, _) = score_prim(&cand_prim);
                    if cand_score + 1e-6 < best_score {
                        best_score = cand_score;
                        best_prim = cand_prim;
                        axes = new_axes;
                        improved = true;
                    }
                }
            }
            iters_used += 1;
            total_iters += 1;
            if !improved {
                step_deg *= 0.5;
            }
        }

        // Commit if we improved.
        if let Prim::Obb { half_extents, .. } = &best_prim {
            let v = 8.0 * half_extents[0] * half_extents[1] * half_extents[2];
            prims[idx].prim = best_prim.clone();
            prims[idx].volume = v;
            prims[idx].weighted_volume = v * best_prim.weight();
            prims[idx].version += 1;
        }
    }
    total_iters
}

/// Rotate the 3-axis frame `axes` around axis index `k` by `theta`
/// radians: axis k is unchanged, the other two rotate within the plane
/// they span. Re-orthogonalises to fight float drift.
fn rotate_axes_about(axes: [Vector3<f32>; 3], k: usize, theta: f32) -> [Vector3<f32>; 3] {
    let (i, j) = match k {
        0 => (1, 2),
        1 => (2, 0),
        _ => (0, 1),
    };
    let c = theta.cos();
    let s = theta.sin();
    let new_i = (axes[i] * c + axes[j] * s).normalize();
    let new_j = (axes[j] * c - axes[i] * s).normalize();
    let mut out = axes;
    out[i] = new_i;
    out[j] = new_j;
    // Final orthogonalisation cross — guarantees right-handed frame.
    out[k] = out[i].cross(&out[j]).normalize();
    out
}

/// Loose cull pass — drops primitive A if ≥`frac` of A's tessellated
/// surface samples lie inside some other primitive B that A shares a
/// mesh vertex with. Same shared-vertex constraint as `cull_redundant`
/// (avoids the all-pairs phase wrapping disjoint components). Targets
/// the visible-overlap issue on hollow architectural meshes: adjacent
/// thin wall-OBBs partially contain each other at corners; the strict
/// `cull_redundant` (require full vertex containment) doesn't catch
/// this, leading to many redundant primitives stacked in the same
/// region. A loose 80–95% threshold strips most of the visible
/// overlap while preserving primitives that contribute unique coverage.
///
/// Sampling: surface tessellation vertices + a coarse barycentric grid
/// per triangle. Cheap relative to the redundant-cull cost since we
/// already need to check against each candidate B. We re-fit nothing —
/// dropped primitives are simply marked alive=false; the caller uses
/// the live set as before.
fn cull_overlapping(prims: &mut [Primitive], frac_threshold: f32) -> usize {
    let live: Vec<usize> = prims
        .iter()
        .enumerate()
        .filter(|(_, p)| p.alive)
        .map(|(i, _)| i)
        .collect();

    // Pre-tessellate every live primitive once.
    let mut prim_samples: Vec<Vec<Point3<f32>>> = Vec::with_capacity(live.len());
    for &i in &live {
        let (verts, tris) = prim::tessellate(&prims[i].prim);
        let mut samples: Vec<Point3<f32>> = Vec::with_capacity(verts.len() + tris.len() * 4);
        for v in &verts {
            samples.push(Point3::new(v[0], v[1], v[2]));
        }
        // 4-point barycentric grid per triangle for face-interior coverage.
        for t in &tris {
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let baries = [
                (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                (0.5, 0.25, 0.25),
                (0.25, 0.5, 0.25),
                (0.25, 0.25, 0.5),
            ];
            for (u, v, w) in baries {
                samples.push(Point3::new(
                    a[0] * w + b[0] * u + c[0] * v,
                    a[1] * w + b[1] * u + c[1] * v,
                    a[2] * w + b[2] * u + c[2] * v,
                ));
            }
        }
        prim_samples.push(samples);
    }

    let mut to_drop: Vec<usize> = Vec::new();
    for (a_idx, &a) in live.iter().enumerate() {
        if to_drop.contains(&a) {
            continue;
        }
        for (b_idx, &b) in live.iter().enumerate() {
            if a == b || to_drop.contains(&b) {
                continue;
            }
            // Cull only the smaller-volume primitive — keeps the more
            // representative shape, avoids reciprocal-cull oscillation.
            if prims[a].volume > prims[b].volume {
                continue;
            }
            // NOTE: We deliberately drop the shared-vertex constraint here
            // (cull_redundant uses it). Overlapping primitives on hollow
            // architecture often DON'T share a mesh vertex even though
            // they spatially overlap — adjacent thin-wall OBBs may have
            // partitioned faces such that no single mesh vertex is in
            // both. We rely instead on the volume gate (smaller-only) +
            // the high overlap threshold to avoid culling unrelated
            // primitives across components.
            let samples = &prim_samples[a_idx];
            if samples.is_empty() {
                continue;
            }
            let mut inside = 0usize;
            for q in samples {
                if prims[b].prim.contains(*q, 0.0) {
                    inside += 1;
                }
            }
            let ratio = inside as f32 / samples.len() as f32;
            if ratio >= frac_threshold {
                to_drop.push(a);
                break;
            }
            // Also probe whether B is inside A (smaller-of-two check
            // already covered by the volume gate, so this only fires
            // when volumes are equal — extremely rare).
            let _ = b_idx;
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
