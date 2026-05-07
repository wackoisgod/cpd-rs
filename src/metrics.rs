//! Quantitative metrics matching paper §4.4: one-way Hausdorff and
//! Chamfer distance from the collider (primitives) to the input mesh.
//! Plus a few extra signals (per-primitive distribution, volume ratio).
//!
//! All distances are reported both raw and normalized by the input AABB's
//! diagonal length so they can be compared across meshes.

use crate::bvh::Bvh;
use crate::decomp::Primitive;
use crate::mesh::{aabb, aabb_diag, Mesh};
use crate::prim;
use nalgebra::{Point3, Vector3};

#[derive(Clone, Debug)]
pub struct Metrics {
    pub samples: u32,
    /// Forward direction: max/mean distance from samples on the primitive
    /// collider's surface to the input mesh. Penalises primitives whose
    /// surface drifts past the input (the slab failure mode).
    pub hausdorff: f32,
    pub hausdorff_norm: f32,
    pub chamfer: f32,
    pub chamfer_norm: f32,
    /// Reverse direction: max/mean distance from samples on the *input
    /// mesh* surface to the *nearest primitive surface*. Penalises gaps in
    /// coverage — large reverse Hausdorff means the input has regions no
    /// primitive is near. Forward Hausdorff alone can be fooled by deleting
    /// primitives (fewer primitives → fewer slabs to drift, but coverage
    /// craters); reverse catches that.
    pub reverse_hausdorff: f32,
    pub reverse_hausdorff_norm: f32,
    pub reverse_chamfer: f32,
    pub reverse_chamfer_norm: f32,
    /// Symmetric (max of forward and reverse) — the standard 2-way
    /// Hausdorff. This is the honest single number for "how close are
    /// these two surfaces" regardless of direction.
    pub hausdorff_2way: f32,
    pub hausdorff_2way_norm: f32,
    pub chamfer_2way: f32,
    pub chamfer_2way_norm: f32,
    /// Fraction of input-surface samples that lie inside the volume of at
    /// least one live primitive. 1.0 = every sample covered, 0.0 = no
    /// coverage. A direct collision-detection signal that
    /// reverse-Hausdorff doesn't quite capture (a sample sitting just
    /// inside a paper-thin shell counts as covered, sample 1mm outside
    /// does not).
    pub coverage_fraction: f32,
    pub total_volume: f32,
    pub aabb_volume: f32,
    pub volume_ratio: f32,
    pub n_primitives: u32,
    pub by_kind: [u32; 6], // obb, sphere, cyl, cap, frustum, prism
    /// Per-primitive worst-case Hausdorff (max sample distance) and median
    /// distance, useful for finding outliers.
    pub worst_prim_idx: usize,
    pub worst_prim_kind: prim::PrimKind,
    pub worst_prim_max: f32,
    pub worst_prim_volume: f32,
}

impl Metrics {
    pub fn human(&self) -> String {
        let by = self.by_kind;
        format!(
            "metrics:\n  primitives:        {}  (obb={} sphere={} cyl={} cap={} frustum={} prism={})\n  total volume:      {:.4}\n  aabb volume:       {:.4}  ({:.1}% ratio)\n  hausdorff fwd:     {:.5}  ({:.4}% of diag)   [primitive→input, paper §4.4]\n  chamfer   fwd:     {:.5}  ({:.4}% of diag)\n  hausdorff rev:     {:.5}  ({:.4}% of diag)   [input→primitive surf, gap detector]\n  chamfer   rev:     {:.5}  ({:.4}% of diag)\n  hausdorff 2-way:   {:.5}  ({:.4}% of diag)   [max of fwd/rev — honest number]\n  chamfer   2-way:   {:.5}  ({:.4}% of diag)\n  coverage fraction: {:.4}                       [frac of input pts inside any primitive]\n  worst primitive:   #{} {:?}, max-sample-dist {:.4}, vol {:.4}\n  samples used:      {}",
            self.n_primitives,
            by[0], by[1], by[2], by[3], by[4], by[5],
            self.total_volume,
            self.aabb_volume,
            100.0 * self.volume_ratio,
            self.hausdorff,
            100.0 * self.hausdorff_norm,
            self.chamfer,
            100.0 * self.chamfer_norm,
            self.reverse_hausdorff,
            100.0 * self.reverse_hausdorff_norm,
            self.reverse_chamfer,
            100.0 * self.reverse_chamfer_norm,
            self.hausdorff_2way,
            100.0 * self.hausdorff_2way_norm,
            self.chamfer_2way,
            100.0 * self.chamfer_2way_norm,
            self.coverage_fraction,
            self.worst_prim_idx,
            self.worst_prim_kind,
            self.worst_prim_max,
            self.worst_prim_volume,
            self.samples,
        )
    }

    pub fn json(&self) -> String {
        let by = self.by_kind;
        format!(
            r#"{{"primitives":{},"by_kind":{{"obb":{},"sphere":{},"cylinder":{},"capsule":{},"frustum":{},"prism":{}}},"total_volume":{},"aabb_volume":{},"volume_ratio":{},"hausdorff":{},"hausdorff_norm":{},"chamfer":{},"chamfer_norm":{},"reverse_hausdorff":{},"reverse_hausdorff_norm":{},"reverse_chamfer":{},"reverse_chamfer_norm":{},"hausdorff_2way":{},"hausdorff_2way_norm":{},"chamfer_2way":{},"chamfer_2way_norm":{},"coverage_fraction":{},"samples":{}}}"#,
            self.n_primitives,
            by[0], by[1], by[2], by[3], by[4], by[5],
            self.total_volume,
            self.aabb_volume,
            self.volume_ratio,
            self.hausdorff,
            self.hausdorff_norm,
            self.chamfer,
            self.chamfer_norm,
            self.reverse_hausdorff,
            self.reverse_hausdorff_norm,
            self.reverse_chamfer,
            self.reverse_chamfer_norm,
            self.hausdorff_2way,
            self.hausdorff_2way_norm,
            self.chamfer_2way,
            self.chamfer_2way_norm,
            self.coverage_fraction,
            self.samples,
        )
    }
}

/// Tiny LCG so we don't add a `rand` dependency. Fine for area-weighted
/// surface sampling; not for cryptography.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self { Self(seed.wrapping_mul(6364136223846793005).wrapping_add(1)) }
    fn step(&mut self) -> u32 {
        self.0 = self.0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn unit(&mut self) -> f32 {
        // map u32 → [0, 1)
        (self.step() as f32) * (1.0 / 4294967296.0)
    }
}

pub fn compute(prims: &[Primitive], mesh: &Mesh, n_samples: u32) -> Metrics {
    let bvh = Bvh::build(&mesh.verts, &mesh.tris);
    let (lo, hi) = aabb(&mesh.verts);
    let aabb_volume =
        ((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z)).max(1e-12);
    let diag = aabb_diag(&mesh.verts).max(1e-12);

    // Tessellate each live primitive and gather (verts, tris, per-tri area)
    // along with cumulative per-primitive area for stratified sampling.
    let mut prim_verts: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut prim_tris: Vec<Vec<[u32; 3]>> = Vec::new();
    let mut prim_tri_cum: Vec<Vec<f32>> = Vec::new();
    let mut prim_total_area: Vec<f32> = Vec::new();
    let mut total_volume = 0.0f32;
    let mut by_kind = [0u32; 6];
    let mut n_alive = 0u32;
    for p in prims {
        if !p.alive {
            continue;
        }
        n_alive += 1;
        total_volume += p.volume;
        by_kind[match p.prim.kind() {
            prim::PrimKind::Obb => 0,
            prim::PrimKind::Sphere => 1,
            prim::PrimKind::Cylinder => 2,
            prim::PrimKind::Capsule => 3,
            prim::PrimKind::Frustum => 4,
            prim::PrimKind::Prism => 5,
        }] += 1;

        let (verts, tris) = prim::tessellate(&p.prim);
        // Defend against NaN in tessellation: a single non-finite vertex
        // turns its triangle's area into NaN, NaN propagates into cum,
        // global_total becomes NaN, every binary-search probe returns
        // out-of-range, and the entire forward sample loop produces 0
        // hits. Sanitize per-tri area to 0 if non-finite — primitives
        // with broken tessellations contribute nothing rather than
        // poisoning the metric.
        let any_nan_vert = verts.iter().any(|v| !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite());
        let (verts, tris) = if any_nan_vert {
            (Vec::new(), Vec::new())
        } else {
            (verts, tris)
        };
        let mut cum = Vec::with_capacity(tris.len());
        let mut acc = 0.0f32;
        for t in &tris {
            let a = Vector3::new(verts[t[0] as usize][0], verts[t[0] as usize][1], verts[t[0] as usize][2]);
            let b = Vector3::new(verts[t[1] as usize][0], verts[t[1] as usize][1], verts[t[1] as usize][2]);
            let c = Vector3::new(verts[t[2] as usize][0], verts[t[2] as usize][1], verts[t[2] as usize][2]);
            let mut area = 0.5 * (b - a).cross(&(c - a)).norm();
            if !area.is_finite() {
                area = 0.0;
            }
            acc += area;
            cum.push(acc);
        }
        prim_verts.push(verts);
        prim_tris.push(tris);
        prim_total_area.push(acc);
        prim_tri_cum.push(cum);
    }

    let global_total: f32 = prim_total_area.iter().sum();
    if global_total <= 0.0 || n_alive == 0 {
        return Metrics {
            samples: 0,
            hausdorff: 0.0,
            hausdorff_norm: 0.0,
            chamfer: 0.0,
            chamfer_norm: 0.0,
            reverse_hausdorff: 0.0,
            reverse_hausdorff_norm: 0.0,
            reverse_chamfer: 0.0,
            reverse_chamfer_norm: 0.0,
            hausdorff_2way: 0.0,
            hausdorff_2way_norm: 0.0,
            chamfer_2way: 0.0,
            chamfer_2way_norm: 0.0,
            coverage_fraction: 0.0,
            total_volume,
            aabb_volume,
            volume_ratio: total_volume / aabb_volume,
            n_primitives: n_alive,
            by_kind,
            worst_prim_idx: 0,
            worst_prim_kind: prim::PrimKind::Obb,
            worst_prim_max: 0.0,
            worst_prim_volume: 0.0,
        };
    }

    // Cumulative per-primitive area for outer sampling tier.
    let mut prim_cum = Vec::with_capacity(prim_total_area.len());
    let mut acc = 0.0f32;
    for &a in &prim_total_area {
        acc += a;
        prim_cum.push(acc);
    }

    let mut rng = Lcg::new(0x9E3779B97F4A7C15);
    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f64;
    let mut samples_taken = 0u32;
    // Track worst primitive — tells us which primitive index is causing
    // the high Hausdorff so we can debug.
    let mut per_prim_max: Vec<f32> = vec![0.0; prim_total_area.len()];
    for _ in 0..n_samples {
        // Pick a primitive proportional to its surface area.
        let r = rng.unit() * global_total;
        let pi = upper_bound(&prim_cum, r);
        if pi >= prim_cum.len() {
            continue;
        }
        // Pick a triangle within that primitive proportional to area.
        let cum = &prim_tri_cum[pi];
        if cum.is_empty() {
            continue;
        }
        let r2 = rng.unit() * prim_total_area[pi];
        let ti = upper_bound(cum, r2);
        if ti >= cum.len() {
            continue;
        }
        // Uniform barycentric sample within the triangle.
        let mut u = rng.unit();
        let mut v = rng.unit();
        if u + v > 1.0 {
            u = 1.0 - u;
            v = 1.0 - v;
        }
        let w = 1.0 - u - v;
        let t = &prim_tris[pi][ti];
        let a = prim_verts[pi][t[0] as usize];
        let b = prim_verts[pi][t[1] as usize];
        let c = prim_verts[pi][t[2] as usize];
        let p = Point3::new(
            a[0] * w + b[0] * u + c[0] * v,
            a[1] * w + b[1] * u + c[1] * v,
            a[2] * w + b[2] * u + c[2] * v,
        );
        let (_pt, _n, signed) = bvh.nearest_face(&mesh.verts, &mesh.tris, p);
        let d = signed.abs();
        if d > max_d {
            max_d = d;
        }
        if d > per_prim_max[pi] {
            per_prim_max[pi] = d;
        }
        sum_d += d as f64;
        samples_taken += 1;
    }

    // Find the alive primitive index whose worst sample dominates the
    // Hausdorff. We need to map back from local index `pi` to the global
    // prims[] index since we filtered out dead slots.
    let mut alive_to_global: Vec<usize> = Vec::with_capacity(per_prim_max.len());
    for (gi, p) in prims.iter().enumerate() {
        if p.alive {
            alive_to_global.push(gi);
        }
    }
    let mut worst_local = 0usize;
    let mut worst_val = 0.0f32;
    for (i, &v) in per_prim_max.iter().enumerate() {
        if v > worst_val {
            worst_val = v;
            worst_local = i;
        }
    }
    let worst_global = alive_to_global.get(worst_local).copied().unwrap_or(0);
    let worst_kind = prims[worst_global].prim.kind();
    let worst_volume = prims[worst_global].volume;

    let chamfer = (sum_d / samples_taken.max(1) as f64) as f32;

    // ── Reverse direction: input mesh surface → primitive surface. ──
    //
    // Build a BVH over the union of all live primitives' tessellated
    // surfaces, then sample input mesh points (area-weighted) and measure
    // distance to the nearest primitive face. Catches the failure mode
    // where deleting primitives drops forward-Hausdorff (no slabs to
    // drift) but leaves swathes of input uncovered.
    //
    // Coverage fraction: separately, count what fraction of input samples
    // are inside any primitive's volume. Closer to the collision-detection
    // truth than reverse-Hausdorff alone.
    let mut prim_bvh_verts: Vec<Point3<f32>> = Vec::new();
    let mut prim_bvh_tris: Vec<[u32; 3]> = Vec::new();
    for (verts, tris) in prim_verts.iter().zip(prim_tris.iter()) {
        let base = prim_bvh_verts.len() as u32;
        for v in verts {
            prim_bvh_verts.push(Point3::new(v[0], v[1], v[2]));
        }
        for t in tris {
            prim_bvh_tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
    }
    let live_prims: Vec<&Primitive> = prims.iter().filter(|p| p.alive).collect();

    let mut reverse_max = 0.0f32;
    let mut reverse_sum = 0.0f64;
    let mut reverse_taken = 0u32;
    let mut covered = 0u32;
    if !prim_bvh_tris.is_empty() {
        let prim_bvh = Bvh::build(&prim_bvh_verts, &prim_bvh_tris);

        // Per-input-triangle area for area-weighted sampling.
        let mut tri_cum: Vec<f32> = Vec::with_capacity(mesh.tris.len());
        let mut tri_acc = 0.0f32;
        for t in &mesh.tris {
            let a = mesh.verts[t[0] as usize];
            let b = mesh.verts[t[1] as usize];
            let c = mesh.verts[t[2] as usize];
            let area = 0.5 * (b - a).cross(&(c - a)).norm();
            tri_acc += area;
            tri_cum.push(tri_acc);
        }
        if tri_acc > 0.0 {
            for _ in 0..n_samples {
                let r = rng.unit() * tri_acc;
                let ti = upper_bound(&tri_cum, r);
                if ti >= mesh.tris.len() {
                    continue;
                }
                let t = &mesh.tris[ti];
                let a = mesh.verts[t[0] as usize];
                let b = mesh.verts[t[1] as usize];
                let c = mesh.verts[t[2] as usize];
                let mut u = rng.unit();
                let mut v = rng.unit();
                if u + v > 1.0 {
                    u = 1.0 - u;
                    v = 1.0 - v;
                }
                let w = 1.0 - u - v;
                let p = Point3::new(
                    a.x * w + b.x * u + c.x * v,
                    a.y * w + b.y * u + c.y * v,
                    a.z * w + b.z * u + c.z * v,
                );
                let (_pt, _n, signed) =
                    prim_bvh.nearest_face(&prim_bvh_verts, &prim_bvh_tris, p);
                let d = signed.abs();
                if d > reverse_max {
                    reverse_max = d;
                }
                reverse_sum += d as f64;
                reverse_taken += 1;

                // Coverage: is the sample inside ANY primitive volume?
                // Tolerance 0 — we want strict containment, not the cull's
                // "near surface counts" mode.
                for lp in &live_prims {
                    if lp.prim.contains(p, 0.0) {
                        covered += 1;
                        break;
                    }
                }
            }
        }
    }
    let reverse_chamfer = if reverse_taken > 0 {
        (reverse_sum / reverse_taken as f64) as f32
    } else {
        0.0
    };
    let coverage_fraction = if reverse_taken > 0 {
        covered as f32 / reverse_taken as f32
    } else {
        0.0
    };
    let hausdorff_2way = max_d.max(reverse_max);
    let chamfer_2way = 0.5 * (chamfer + reverse_chamfer);

    Metrics {
        samples: samples_taken,
        hausdorff: max_d,
        hausdorff_norm: max_d / diag,
        chamfer,
        chamfer_norm: chamfer / diag,
        reverse_hausdorff: reverse_max,
        reverse_hausdorff_norm: reverse_max / diag,
        reverse_chamfer,
        reverse_chamfer_norm: reverse_chamfer / diag,
        hausdorff_2way,
        hausdorff_2way_norm: hausdorff_2way / diag,
        chamfer_2way,
        chamfer_2way_norm: chamfer_2way / diag,
        coverage_fraction,
        total_volume,
        aabb_volume,
        volume_ratio: total_volume / aabb_volume,
        n_primitives: n_alive,
        by_kind,
        worst_prim_idx: worst_global,
        worst_prim_kind: worst_kind,
        worst_prim_max: worst_val,
        worst_prim_volume: worst_volume,
    }
}

/// Smallest index `i` such that `cum[i] >= x`. Cum must be non-decreasing.
fn upper_bound(cum: &[f32], x: f32) -> usize {
    let mut lo = 0usize;
    let mut hi = cum.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cum[mid] >= x {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}
