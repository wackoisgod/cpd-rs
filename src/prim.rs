use nalgebra::{Point3, Vector3};
use std::f32::consts::PI;

const MIN_HALF_EXTENT: f32 = 1e-3;
const TINY: f32 = 1e-6;

/// A fitted convex primitive. Each variant carries everything needed to
/// (a) compute its volume, (b) test point containment, (c) tessellate
/// it for visualization.
#[derive(Clone, Debug)]
pub enum Prim {
    Obb {
        center: Point3<f32>,
        axes: [Vector3<f32>; 3],
        half_extents: [f32; 3],
    },
    Sphere {
        center: Point3<f32>,
        r: f32,
    },
    /// Capped cylinder. p_cyl is a point on the axis (we store the center).
    /// h is the *full* axial length.
    Cylinder {
        center: Point3<f32>,
        axis: Vector3<f32>,
        h: f32,
        r: f32,
    },
    /// Capsule. h is the gap between the two hemisphere centers (so total
    /// axial extent is h + 2r).
    Capsule {
        center: Point3<f32>,
        axis: Vector3<f32>,
        h: f32,
        r: f32,
    },
    Frustum {
        center: Point3<f32>,
        axis: Vector3<f32>,
        h: f32,
        r_bot: f32,
        r_top: f32,
    },
    /// Isosceles trapezoidal prism. Cross-section in the (ay, az) plane is a
    /// trapezoid: half-z-width hzb at -ay side, hzt at +ay side. ax is the
    /// extruded direction with half-extent hx.
    Prism {
        center: Point3<f32>,
        axes: [Vector3<f32>; 3], // [ax, ay, az]
        hx: f32,
        hy: f32,
        hzt: f32,
        hzb: f32,
    },
}

impl Prim {
    /// True if the fit is "pancake degenerate": the smallest dimension
    /// has clamped to MIN_HALF_EXTENT *and* the largest dimension is much
    /// bigger. These arise when many near-coplanar faces from disparate
    /// parts of a mesh get merged into a single primitive whose
    /// thickness collapses to the clamp — surface area is metres but
    /// only a tiny fraction overlaps actual input geometry. We require
    /// the clamp condition (not just an aspect ratio) so legitimate
    /// thin features (e.g. a real 5mm-thick wall) aren't rejected.
    pub fn is_pancake(&self) -> bool {
        // Min-dim has to be sitting *at* the half-extent clamp (we leave
        // a 1.5× buffer to absorb any float noise from the fit code) and
        // the primitive must extend at least 1000× further in some other
        // direction before we call it a pancake. Looser thresholds (e.g.
        // 0.01) caught legitimate thin features like 1mm vehicle panels
        // and inflated their parent merge tree.
        const CLAMP_BUFFER: f32 = 1.5;
        const ASPECT: f32 = 0.001;
        let clamped = MIN_HALF_EXTENT * CLAMP_BUFFER;
        match *self {
            Prim::Obb { half_extents, .. } => {
                let min_h = half_extents[0].min(half_extents[1]).min(half_extents[2]);
                let max_h = half_extents[0].max(half_extents[1]).max(half_extents[2]);
                min_h < clamped && max_h > 0.0 && min_h / max_h < ASPECT
            }
            Prim::Prism {
                hx,
                hy,
                hzt,
                hzb,
                ..
            } => {
                let dims = [hx, hy, hzt, hzb];
                let min_h = dims.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_h = dims.iter().cloned().fold(0.0f32, f32::max);
                min_h < clamped && max_h > 0.0 && min_h / max_h < ASPECT
            }
            Prim::Cylinder { h, r, .. } | Prim::Capsule { h, r, .. } => {
                let max_d = h.max(r * 2.0);
                let min_d = h.min(r * 2.0);
                min_d < clamped && max_d > 0.0 && min_d / max_d < ASPECT
            }
            Prim::Frustum {
                h, r_bot, r_top, ..
            } => {
                let r_max = r_top.max(r_bot) * 2.0;
                let max_d = h.max(r_max);
                let min_d = h.min(r_max);
                min_d < clamped && max_d > 0.0 && min_d / max_d < ASPECT
            }
            // Spheres can't be degenerate (single radius).
            Prim::Sphere { .. } => false,
        }
    }

    pub fn volume(&self) -> f32 {
        match *self {
            Prim::Obb { half_extents: h, .. } => 8.0 * h[0] * h[1] * h[2],
            Prim::Sphere { r, .. } => (4.0 / 3.0) * PI * r * r * r,
            Prim::Cylinder { h, r, .. } => PI * r * r * h,
            Prim::Capsule { h, r, .. } => PI * r * r * h + (4.0 / 3.0) * PI * r * r * r,
            Prim::Frustum { h, r_bot, r_top, .. } => {
                (PI * h / 3.0) * (r_top * r_top + r_top * r_bot + r_bot * r_bot)
            }
            Prim::Prism {
                hx, hy, hzt, hzb, ..
            } => 4.0 * hx * hy * (hzt + hzb),
        }
    }

    /// Per-shape multiplier on volume during cost selection (paper §3.3).
    /// Smaller weights are preferred.
    pub fn weight(&self) -> f32 {
        match self {
            Prim::Obb { .. } => 1.0,
            Prim::Sphere { .. } => 1.0,
            Prim::Capsule { .. } => 1.0,
            Prim::Cylinder { .. } => 1.05,
            Prim::Prism { .. } => 1.4,
            Prim::Frustum { .. } => 2.1,
        }
    }

    pub fn weighted_volume(&self) -> f32 {
        self.volume() * self.weight()
    }

    pub fn kind(&self) -> PrimKind {
        match self {
            Prim::Obb { .. } => PrimKind::Obb,
            Prim::Sphere { .. } => PrimKind::Sphere,
            Prim::Cylinder { .. } => PrimKind::Cylinder,
            Prim::Capsule { .. } => PrimKind::Capsule,
            Prim::Frustum { .. } => PrimKind::Frustum,
            Prim::Prism { .. } => PrimKind::Prism,
        }
    }

    /// Conservative containment check: returns true iff `p` lies inside the
    /// closed primitive (with a small tolerance). Used by the redundant
    /// primitive cull, so false negatives are acceptable but false positives
    /// would let us drop primitives that shouldn't be dropped.
    pub fn contains(&self, p: Point3<f32>, tol: f32) -> bool {
        match *self {
            Prim::Obb {
                center,
                axes,
                half_extents: h,
            } => {
                let d = p - center;
                axes[0].dot(&d).abs() <= h[0] + tol
                    && axes[1].dot(&d).abs() <= h[1] + tol
                    && axes[2].dot(&d).abs() <= h[2] + tol
            }
            Prim::Sphere { center, r } => (p - center).norm() <= r + tol,
            Prim::Cylinder { center, axis, h, r } => {
                let d = p - center;
                let ax = axis.dot(&d);
                if ax.abs() > h * 0.5 + tol {
                    return false;
                }
                let radial = (d - axis * ax).norm();
                radial <= r + tol
            }
            Prim::Capsule { center, axis, h, r } => {
                let d = p - center;
                let ax = axis.dot(&d);
                let half_h = h * 0.5;
                let clamped = ax.clamp(-half_h, half_h);
                let radial_vec = d - axis * clamped;
                radial_vec.norm() <= r + tol
            }
            Prim::Frustum {
                center,
                axis,
                h,
                r_bot,
                r_top,
            } => {
                let d = p - center;
                let ax = axis.dot(&d);
                let half_h = h * 0.5;
                if ax.abs() > half_h + tol {
                    return false;
                }
                let u = ((ax + half_h) / h).clamp(0.0, 1.0);
                let r_at = r_bot * (1.0 - u) + r_top * u;
                let radial = (d - axis * ax).norm();
                radial <= r_at + tol
            }
            Prim::Prism {
                center,
                axes,
                hx,
                hy,
                hzt,
                hzb,
            } => {
                let d = p - center;
                let dx = axes[0].dot(&d);
                let dy = axes[1].dot(&d);
                let dz = axes[2].dot(&d);
                if dx.abs() > hx + tol || dy.abs() > hy + tol {
                    return false;
                }
                let u = ((dy + hy) / (2.0 * hy.max(TINY))).clamp(0.0, 1.0);
                let half_z = hzb * (1.0 - u) + hzt * u;
                dz.abs() <= half_z + tol
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimKind {
    Obb,
    Sphere,
    Cylinder,
    Capsule,
    Frustum,
    Prism,
}

// ---------------------------------------------------------------------------
// Fit functions
// ---------------------------------------------------------------------------

/// OBB axis-projected fit. Returns (center, axes, half_extents).
pub fn fit_obb(axes: [Vector3<f32>; 3], points: &[Point3<f32>]) -> Prim {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for p in points {
        for i in 0..3 {
            let d = axes[i].dot(&p.coords);
            if d < lo[i] {
                lo[i] = d;
            }
            if d > hi[i] {
                hi[i] = d;
            }
        }
    }
    let mut he = [0.0f32; 3];
    let mut c_axis = [0.0f32; 3];
    for i in 0..3 {
        he[i] = ((hi[i] - lo[i]) * 0.5).max(MIN_HALF_EXTENT);
        c_axis[i] = (hi[i] + lo[i]) * 0.5;
    }
    let center = axes[0] * c_axis[0] + axes[1] * c_axis[1] + axes[2] * c_axis[2];
    Prim::Obb {
        center: Point3::from(center),
        axes,
        half_extents: he,
    }
}

pub fn fit_sphere(center: Point3<f32>, points: &[Point3<f32>]) -> Prim {
    let mut r2 = 0.0f32;
    for p in points {
        let d = (*p - center).norm_squared();
        if d > r2 {
            r2 = d;
        }
    }
    let r = r2.sqrt().max(MIN_HALF_EXTENT);
    Prim::Sphere { center, r }
}

/// Fit a cylinder along a single axis a passing through the OBB center.
fn fit_cylinder_on_axis(center: Point3<f32>, axis: Vector3<f32>, points: &[Point3<f32>]) -> Prim {
    let mut max_r2 = 0.0f32;
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in points {
        let d = *p - center;
        let ax = axis.dot(&d);
        if ax < lo {
            lo = ax;
        }
        if ax > hi {
            hi = ax;
        }
        let radial = d - axis * ax;
        let r2 = radial.norm_squared();
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    let r = max_r2.sqrt().max(MIN_HALF_EXTENT);
    let h = (hi - lo).max(MIN_HALF_EXTENT);
    // Recenter along axis so center is the midpoint of the axial extent.
    let shift = (hi + lo) * 0.5;
    let recentered = center + axis * shift;
    Prim::Cylinder {
        center: recentered,
        axis,
        h,
        r,
    }
}

pub fn fit_cylinder_best(
    obb_center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
) -> Prim {
    // For each axis, the axial extent. We skip axes where the cloud is
    // essentially flat (extent << largest extent) — fitting a cylinder
    // along such an axis collapses to a thin disk whose tessellated rim
    // extends far from any actual point. Volume formula gives a misleading
    // tiny number (π·r²·MIN_HALF_EXTENT), so the disk would win selection.
    let extents = axial_extents(obb_center, axes, points);
    let max_e = extents.iter().cloned().fold(0.0f32, f32::max);
    let thresh = max_e * 0.05;

    let mut best: Option<Prim> = None;
    for (i, &a) in axes.iter().enumerate() {
        if extents[i] < thresh {
            continue;
        }
        let cand = fit_cylinder_on_axis(obb_center, a, points);
        match &best {
            None => best = Some(cand),
            Some(b) => {
                if cand.weighted_volume() < b.weighted_volume() {
                    best = Some(cand);
                }
            }
        }
    }
    best.unwrap_or_else(|| fit_cylinder_on_axis(obb_center, axes[0], points))
}

fn fit_capsule_on_axis(center: Point3<f32>, axis: Vector3<f32>, points: &[Point3<f32>]) -> Prim {
    // First pass: r = max radial distance from the axis.
    let mut max_r2 = 0.0f32;
    for p in points {
        let d = *p - center;
        let ax = axis.dot(&d);
        let radial = d - axis * ax;
        let r2 = radial.norm_squared();
        if r2 > max_r2 {
            max_r2 = r2;
        }
    }
    let r = max_r2.sqrt().max(MIN_HALF_EXTENT);

    // Second pass: place the two hemisphere centres so every point is
    // contained in either the cylinder body or one of the spheres.
    //
    // For point (ax, r_p) with s = √(r² − r_p²), the upper hemisphere
    // centre `top` must satisfy `top ≥ ax − s` (so the upper sphere
    // reaches up to `ax`), and the lower hemisphere centre `bot` must
    // satisfy `bot ≤ ax + s`. Taking max/min over points respectively
    // gives the tightest valid (top, bot). NB: the paper's formula
    // `h(p) = ax − s` with `height = max(h) − min(h)` is over-conservative —
    // it conflates the two constraints and yields a longer-than-needed
    // capsule.
    let mut top = f32::NEG_INFINITY;
    let mut bot = f32::INFINITY;
    for p in points {
        let d = *p - center;
        let ax = axis.dot(&d);
        let radial = d - axis * ax;
        let r2_p = radial.norm_squared();
        let inner = (r * r - r2_p).max(0.0);
        let s = inner.sqrt();
        let upper_lower_bound = ax - s; // top must be at least this
        let lower_upper_bound = ax + s; // bot must be at most this
        if upper_lower_bound > top {
            top = upper_lower_bound;
        }
        if lower_upper_bound < bot {
            bot = lower_upper_bound;
        }
    }
    // top can come out below bot if all points fit inside a single sphere
    // (all axial extremes covered by hemispheres alone). In that case we
    // collapse to a degenerate cylinder body of length zero.
    let (top, bot) = if top < bot {
        let mid = (top + bot) * 0.5;
        (mid, mid)
    } else {
        (top, bot)
    };
    let h = (top - bot).max(MIN_HALF_EXTENT);
    let shift = (top + bot) * 0.5;
    let recentered = center + axis * shift;
    Prim::Capsule {
        center: recentered,
        axis,
        h,
        r,
    }
}

pub fn fit_capsule_best(
    obb_center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
) -> Prim {
    // Same degeneracy guard as cylinders.
    let extents = axial_extents(obb_center, axes, points);
    let max_e = extents.iter().cloned().fold(0.0f32, f32::max);
    let thresh = max_e * 0.05;

    let mut best: Option<Prim> = None;
    for (i, &a) in axes.iter().enumerate() {
        if extents[i] < thresh {
            continue;
        }
        let cand = fit_capsule_on_axis(obb_center, a, points);
        match &best {
            None => best = Some(cand),
            Some(b) => {
                if cand.weighted_volume() < b.weighted_volume() {
                    best = Some(cand);
                }
            }
        }
    }
    best.unwrap_or_else(|| fit_capsule_on_axis(obb_center, axes[0], points))
}

fn axial_extents(
    obb_center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for (i, &a) in axes.iter().enumerate() {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for p in points {
            let v = a.dot(&(p - obb_center));
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        out[i] = (hi - lo).max(0.0);
    }
    out
}

/// Frustum fit along a fixed axis (paper Alg. 2). The radial profile is
/// linear from `r_bot` at u=0 to `r_top` at u=1, where u is the normalized
/// axial coordinate ∈ [0, 1].
fn fit_frustum_on_axis(center: Point3<f32>, axis: Vector3<f32>, points: &[Point3<f32>]) -> Prim {
    if points.is_empty() {
        return Prim::Frustum {
            center,
            axis,
            h: MIN_HALF_EXTENT,
            r_bot: MIN_HALF_EXTENT,
            r_top: MIN_HALF_EXTENT,
        };
    }
    // First, find the full axial extent.
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for p in points {
        let d = *p - center;
        let ax = axis.dot(&d);
        if ax < lo {
            lo = ax;
        }
        if ax > hi {
            hi = ax;
        }
    }
    let h = (hi - lo).max(MIN_HALF_EXTENT);
    let shift = (hi + lo) * 0.5;
    let new_center = center + axis * shift;

    // Second, project each point and run the iterative LP from Alg 2.
    let n = points.len();
    let mut u = Vec::with_capacity(n);
    let mut radial = Vec::with_capacity(n);
    for p in points {
        let d = *p - new_center;
        let ax = axis.dot(&d);
        u.push(((ax + 0.5 * h) / h).clamp(0.0, 1.0));
        let r_vec = d - axis * ax;
        radial.push(r_vec.norm());
    }

    let r_top_for = |r: f32, y: f32, r_bot: f32| -> f32 {
        if y <= TINY {
            return 0.0;
        }
        ((r - r_bot * (1.0 - y)) / y).max(0.0)
    };
    let r_bot_for = |r: f32, y: f32, r_top: f32| -> f32 {
        if y >= 1.0 - TINY {
            return 0.0;
        }
        ((r - r_top * y) / (1.0 - y)).max(0.0)
    };

    let mut r_top = 0.0f32;
    let mut r_bot = 0.0f32;
    let mut star_top = (0.0f32, 1.0f32); // (r, y)
    let mut star_bot = (0.0f32, 0.0f32);

    for i in 0..n {
        let yi = u[i];
        let ri = radial[i];
        if yi <= 0.5 {
            let next = r_bot_for(ri, yi, r_top);
            if next > r_bot {
                r_bot = next;
                star_bot = (ri, yi);
                r_top = r_top_for(star_top.0, star_top.1, r_bot);
            }
        } else {
            let next = r_top_for(ri, yi, r_bot);
            if next > r_top {
                r_top = next;
                star_top = (ri, yi);
                r_bot = r_bot_for(star_bot.0, star_bot.1, r_top);
            }
        }
    }

    // Final tightening pass to guarantee every point is enclosed.
    let mut tighten = |r_top: &mut f32, r_bot: &mut f32| {
        let mut new_top = 0.0f32;
        for i in 0..n {
            let v = r_top_for(radial[i], u[i], *r_bot);
            if v > new_top {
                new_top = v;
            }
        }
        *r_top = new_top;
        let mut new_bot = 0.0f32;
        for i in 0..n {
            let v = r_bot_for(radial[i], u[i], *r_top);
            if v > new_bot {
                new_bot = v;
            }
        }
        *r_bot = new_bot;
    };
    tighten(&mut r_top, &mut r_bot);

    Prim::Frustum {
        center: new_center,
        axis,
        h,
        r_bot: r_bot.max(MIN_HALF_EXTENT),
        r_top: r_top.max(MIN_HALF_EXTENT),
    }
}

pub fn fit_frustum(
    obb_center: Point3<f32>,
    cyl_axis: Vector3<f32>,
    points: &[Point3<f32>],
) -> Prim {
    fit_frustum_on_axis(obb_center, cyl_axis, points)
}

/// Trapezoidal prism fit for one axis ordering (paper Alg. 3).
fn fit_prism_on_axes(
    center: Point3<f32>,
    ax: Vector3<f32>,
    ay: Vector3<f32>,
    az: Vector3<f32>,
    points: &[Point3<f32>],
) -> Prim {
    let n = points.len();
    if n == 0 {
        return Prim::Prism {
            center,
            axes: [ax, ay, az],
            hx: MIN_HALF_EXTENT,
            hy: MIN_HALF_EXTENT,
            hzt: MIN_HALF_EXTENT,
            hzb: MIN_HALF_EXTENT,
        };
    }
    // Centered hx, hy from min/max projections; recenter along ay so the
    // trapezoid's normalized y coordinate truly spans [0, 1]. Also
    // recenter along ax to keep symmetry.
    let mut lox = f32::INFINITY;
    let mut hix = f32::NEG_INFINITY;
    let mut loy = f32::INFINITY;
    let mut hiy = f32::NEG_INFINITY;
    let mut loz = f32::INFINITY;
    let mut hiz = f32::NEG_INFINITY;
    for p in points {
        let d = *p - center;
        let dx = ax.dot(&d);
        let dy = ay.dot(&d);
        let dz = az.dot(&d);
        if dx < lox {
            lox = dx;
        }
        if dx > hix {
            hix = dx;
        }
        if dy < loy {
            loy = dy;
        }
        if dy > hiy {
            hiy = dy;
        }
        if dz < loz {
            loz = dz;
        }
        if dz > hiz {
            hiz = dz;
        }
    }
    let hx = ((hix - lox) * 0.5).max(MIN_HALF_EXTENT);
    let hy = ((hiy - loy) * 0.5).max(MIN_HALF_EXTENT);
    let shift_x = (hix + lox) * 0.5;
    let shift_y = (hiy + loy) * 0.5;
    let shift_z = (hiz + loz) * 0.5;
    let new_center = center + ax * shift_x + ay * shift_y + az * shift_z;

    // Compute u, z for the LP in the new coordinate frame.
    let mut u = Vec::with_capacity(n);
    let mut zabs = Vec::with_capacity(n);
    for p in points {
        let d = *p - new_center;
        let dy = ay.dot(&d);
        let dz = az.dot(&d);
        u.push(((dy + hy) / (2.0 * hy)).clamp(0.0, 1.0));
        zabs.push(dz.abs());
    }

    let h_zt_for = |z: f32, y: f32, h_zb: f32| -> f32 {
        if y <= TINY {
            return 0.0;
        }
        ((z - h_zb * (1.0 - y)) / y).max(0.0)
    };
    let h_zb_for = |z: f32, y: f32, h_zt: f32| -> f32 {
        if y >= 1.0 - TINY {
            return 0.0;
        }
        ((z - h_zt * y) / (1.0 - y)).max(0.0)
    };

    let mut h_zt = 0.0f32;
    let mut h_zb = 0.0f32;
    let mut star_zt = (0.0f32, 1.0f32);
    let mut star_zb = (0.0f32, 0.0f32);

    for i in 0..n {
        let yi = u[i];
        let zi = zabs[i];
        if yi <= 0.5 {
            let next = h_zb_for(zi, yi, h_zt);
            if next > h_zb {
                h_zb = next;
                star_zb = (zi, yi);
                h_zt = h_zt_for(star_zt.0, star_zt.1, h_zb);
            }
        } else {
            let next = h_zt_for(zi, yi, h_zb);
            if next > h_zt {
                h_zt = next;
                star_zt = (zi, yi);
                h_zb = h_zb_for(star_zb.0, star_zb.1, h_zt);
            }
        }
    }

    // FixSide passes: tighten one side against all points with the other
    // fixed, then swap. Guarantees every point is enclosed.
    let mut fix_side = |fix_top_first: bool, h_zt: &mut f32, h_zb: &mut f32| {
        let recompute_top = |h_zb: f32| -> f32 {
            let mut acc = 0.0f32;
            for i in 0..n {
                let v = h_zt_for(zabs[i], u[i], h_zb);
                if v > acc {
                    acc = v;
                }
            }
            acc
        };
        let recompute_bot = |h_zt: f32| -> f32 {
            let mut acc = 0.0f32;
            for i in 0..n {
                let v = h_zb_for(zabs[i], u[i], h_zt);
                if v > acc {
                    acc = v;
                }
            }
            acc
        };
        if fix_top_first {
            *h_zb = recompute_bot(*h_zt);
        } else {
            *h_zt = recompute_top(*h_zb);
        }
    };
    fix_side(h_zt >= h_zb, &mut h_zt, &mut h_zb);
    fix_side(h_zt < h_zb, &mut h_zt, &mut h_zb);

    Prim::Prism {
        center: new_center,
        axes: [ax, ay, az],
        hx,
        hy,
        hzt: h_zt.max(MIN_HALF_EXTENT),
        hzb: h_zb.max(MIN_HALF_EXTENT),
    }
}

/// Try all 6 permutations of the 3 axes (paper §3.2) and pick the min
/// weighted-volume prism.
pub fn fit_prism_best(
    obb_center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
) -> Prim {
    let perms = [
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ];
    let mut best: Option<Prim> = None;
    for (i, j, k) in perms {
        let cand = fit_prism_on_axes(obb_center, axes[i], axes[j], axes[k], points);
        match &best {
            None => best = Some(cand),
            Some(b) => {
                if cand.weighted_volume() < b.weighted_volume() {
                    best = Some(cand);
                }
            }
        }
    }
    best.unwrap()
}

/// All-candidate fit (no sphere skip), returns every primitive type as a
/// Vec so the caller can apply a custom selection criterion (e.g. a
/// Hausdorff-aware combined cost).
pub fn fit_all(
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
    enabled: PrimMask,
) -> Vec<Prim> {
    let obb = fit_obb(axes, points);
    let obb_center = match &obb {
        Prim::Obb { center, .. } => *center,
        _ => unreachable!(),
    };
    let mut out: Vec<Prim> = Vec::with_capacity(6);
    if enabled.obb {
        out.push(obb.clone());
    }
    if enabled.sphere {
        out.push(fit_sphere(obb_center, points));
    }
    if enabled.cylinder {
        out.push(fit_cylinder_best(obb_center, axes, points));
    }
    if enabled.capsule {
        out.push(fit_capsule_best(obb_center, axes, points));
    }
    if enabled.frustum {
        // frustum needs a cylinder axis seed
        let cyl = fit_cylinder_best(obb_center, axes, points);
        let cyl_axis = match &cyl {
            Prim::Cylinder { axis, .. } => *axis,
            _ => axes[0],
        };
        out.push(fit_frustum(obb_center, cyl_axis, points));
    }
    if enabled.prism {
        out.push(fit_prism_best(obb_center, axes, points));
    }
    if out.is_empty() {
        out.push(obb);
    }
    out
}

/// Try every enabled primitive type and return the one with the smallest
/// weighted volume. OBB is computed unconditionally because its centre
/// seeds the sphere / cylinder / capsule fits, but it's only *eligible*
/// to be returned when `enabled.obb == true`.
pub fn fit_best(
    axes: [Vector3<f32>; 3],
    points: &[Point3<f32>],
    enabled: PrimMask,
) -> Prim {
    let obb = fit_obb(axes, points);
    let obb_center = match &obb {
        Prim::Obb { center, .. } => *center,
        _ => unreachable!(),
    };

    let cylinder = if enabled.cylinder || enabled.frustum {
        Some(fit_cylinder_best(obb_center, axes, points))
    } else {
        None
    };

    let mut best: Option<Prim> = None;
    let consider = |cand: Prim, best: &mut Option<Prim>| match best {
        None => *best = Some(cand),
        Some(b) => {
            if cand.weighted_volume() < b.weighted_volume() {
                *best = Some(cand);
            }
        }
    };

    if enabled.obb {
        consider(obb.clone(), &mut best);
    }

    // Tiny primitives (singleton tri, two coplanar tris) don't have
    // enough geometry for cylinder/capsule/frustum/prism to beat OBB —
    // they collapse to the same box-shaped slab. Skip the per-call cost.
    const FANCY_FIT_MIN_VERTS: usize = 8;
    if points.len() >= FANCY_FIT_MIN_VERTS {
        // Sphere strictly loses to OBB on weighted volume (proof: r ≥
        // √(hx²+hy²+hz²) gives sphere ≥ 2.7× OBB). Only worth fitting
        // when OBB itself isn't a candidate.
        if enabled.sphere && !enabled.obb {
            consider(fit_sphere(obb_center, points), &mut best);
        }
        if enabled.cylinder {
            if let Some(c) = &cylinder {
                consider(c.clone(), &mut best);
            }
        }
        if enabled.capsule {
            consider(fit_capsule_best(obb_center, axes, points), &mut best);
        }
        if enabled.frustum {
            let cyl_axis = match cylinder.as_ref() {
                Some(Prim::Cylinder { axis, .. }) => *axis,
                _ => axes[0],
            };
            consider(fit_frustum(obb_center, cyl_axis, points), &mut best);
        }
        if enabled.prism {
            consider(fit_prism_best(obb_center, axes, points), &mut best);
        }
    }

    // Pathological: every type masked off. Return the OBB so we always
    // emit something.
    best.unwrap_or(obb)
}

#[derive(Clone, Copy, Debug)]
pub struct PrimMask {
    pub obb: bool,
    pub sphere: bool,
    pub cylinder: bool,
    pub capsule: bool,
    pub frustum: bool,
    pub prism: bool,
}

impl PrimMask {
    pub fn all() -> Self {
        Self {
            obb: true,
            sphere: true,
            cylinder: true,
            capsule: true,
            frustum: true,
            prism: true,
        }
    }
    pub fn obb_only() -> Self {
        Self {
            obb: true,
            sphere: false,
            cylinder: false,
            capsule: false,
            frustum: false,
            prism: false,
        }
    }
}

/// Axis-aligned bounding box of the primitive in world space. Used as the
/// sampling region for the empty-space-preservation check. Computed
/// analytically (O(1) per primitive, no allocation).
pub fn world_aabb(prim: &Prim) -> ([f32; 3], [f32; 3]) {
    match *prim {
        Prim::Obb {
            center,
            axes,
            half_extents,
        } => {
            // Each component of the world AABB extent is the dot product of
            // |axis_i| with half_extents.
            let mut ext = [0.0f32; 3];
            for i in 0..3 {
                for k in 0..3 {
                    ext[k] += axes[i][k].abs() * half_extents[i];
                }
            }
            (
                [center.x - ext[0], center.y - ext[1], center.z - ext[2]],
                [center.x + ext[0], center.y + ext[1], center.z + ext[2]],
            )
        }
        Prim::Sphere { center, r } => (
            [center.x - r, center.y - r, center.z - r],
            [center.x + r, center.y + r, center.z + r],
        ),
        Prim::Cylinder {
            center,
            axis,
            h,
            r,
        } => {
            let half_h = h * 0.5;
            let mut lo = [0.0f32; 3];
            let mut hi = [0.0f32; 3];
            for k in 0..3 {
                let along = axis[k].abs() * half_h;
                let perp = (1.0 - axis[k] * axis[k]).max(0.0).sqrt() * r;
                lo[k] = center[k] - along - perp;
                hi[k] = center[k] + along + perp;
            }
            (lo, hi)
        }
        Prim::Capsule {
            center,
            axis,
            h,
            r,
        } => {
            // Tight AABB of the union of two spheres of radius r at the two
            // hemisphere centers.
            let half_h = h * 0.5;
            let mut lo = [0.0f32; 3];
            let mut hi = [0.0f32; 3];
            for k in 0..3 {
                let top = center[k] + axis[k] * half_h;
                let bot = center[k] - axis[k] * half_h;
                lo[k] = top.min(bot) - r;
                hi[k] = top.max(bot) + r;
            }
            (lo, hi)
        }
        Prim::Frustum {
            center,
            axis,
            h,
            r_bot,
            r_top,
        } => {
            let half_h = h * 0.5;
            let r_max = r_top.max(r_bot);
            let mut lo = [0.0f32; 3];
            let mut hi = [0.0f32; 3];
            for k in 0..3 {
                let along = axis[k].abs() * half_h;
                let perp = (1.0 - axis[k] * axis[k]).max(0.0).sqrt() * r_max;
                lo[k] = center[k] - along - perp;
                hi[k] = center[k] + along + perp;
            }
            (lo, hi)
        }
        Prim::Prism {
            center,
            axes,
            hx,
            hy,
            hzt,
            hzb,
        } => {
            let mut lo = [f32::INFINITY; 3];
            let mut hi = [f32::NEG_INFINITY; 3];
            for sx in [-1.0f32, 1.0] {
                for (sy, hz) in [(-1.0f32, hzb), (1.0, hzt)] {
                    for sz in [-1.0f32, 1.0] {
                        let v = center.coords
                            + axes[0] * (sx * hx)
                            + axes[1] * (sy * hy)
                            + axes[2] * (sz * hz);
                        for k in 0..3 {
                            if v[k] < lo[k] {
                                lo[k] = v[k];
                            }
                            if v[k] > hi[k] {
                                hi[k] = v[k];
                            }
                        }
                    }
                }
            }
            (lo, hi)
        }
    }
}

// ---------------------------------------------------------------------------
// Tessellation for the viewer / OBJ output. Returns (verts, tris).
// ---------------------------------------------------------------------------

pub fn tessellate(prim: &Prim) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    match *prim {
        Prim::Obb {
            center,
            axes,
            half_extents,
        } => box_mesh(center, axes, half_extents),
        Prim::Sphere { center, r } => sphere_mesh(center, r, 16, 12),
        Prim::Cylinder {
            center,
            axis,
            h,
            r,
        } => cylinder_mesh(center, axis, h, r, 24),
        Prim::Capsule {
            center,
            axis,
            h,
            r,
        } => capsule_mesh(center, axis, h, r, 24, 8),
        Prim::Frustum {
            center,
            axis,
            h,
            r_bot,
            r_top,
        } => frustum_mesh(center, axis, h, r_bot, r_top, 24),
        Prim::Prism {
            center,
            axes,
            hx,
            hy,
            hzt,
            hzb,
        } => prism_mesh(center, axes, hx, hy, hzt, hzb),
    }
}

fn box_mesh(
    center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    he: [f32; 3],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut verts = Vec::with_capacity(8);
    for k in 0..8usize {
        let sx = if k & 1 != 0 { 1.0 } else { -1.0 };
        let sy = if k & 2 != 0 { 1.0 } else { -1.0 };
        let sz = if k & 4 != 0 { 1.0 } else { -1.0 };
        let v = center.coords + axes[0] * (sx * he[0]) + axes[1] * (sy * he[1]) + axes[2] * (sz * he[2]);
        verts.push([v.x, v.y, v.z]);
    }
    let q = |a, b, c, d| [(a, b, c), (a, c, d)];
    let faces = [
        q(0u32, 1, 3, 2),
        q(4, 6, 7, 5),
        q(0, 4, 5, 1),
        q(2, 3, 7, 6),
        q(0, 2, 6, 4),
        q(1, 5, 7, 3),
    ];
    let mut tris = Vec::with_capacity(12);
    for f in &faces {
        for t in f {
            tris.push([t.0, t.1, t.2]);
        }
    }
    (verts, tris)
}

fn build_axis_basis(axis: Vector3<f32>) -> (Vector3<f32>, Vector3<f32>) {
    let a = axis.normalize();
    let helper = if a.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let u = a.cross(&helper).normalize();
    let v = a.cross(&u).normalize();
    (u, v)
}

fn sphere_mesh(
    center: Point3<f32>,
    r: f32,
    sectors: u32,
    stacks: u32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut verts = Vec::new();
    for i in 0..=stacks {
        let theta = PI * (i as f32) / (stacks as f32);
        let st = theta.sin();
        let ct = theta.cos();
        for j in 0..=sectors {
            let phi = 2.0 * PI * (j as f32) / (sectors as f32);
            let x = r * st * phi.cos();
            let y = r * ct;
            let z = r * st * phi.sin();
            let p = center.coords + Vector3::new(x, y, z);
            verts.push([p.x, p.y, p.z]);
        }
    }
    let mut tris = Vec::new();
    let row = sectors + 1;
    for i in 0..stacks {
        for j in 0..sectors {
            let a = i * row + j;
            let b = a + row;
            tris.push([a, b, a + 1]);
            tris.push([a + 1, b, b + 1]);
        }
    }
    (verts, tris)
}

fn cylinder_mesh(
    center: Point3<f32>,
    axis: Vector3<f32>,
    h: f32,
    r: f32,
    sectors: u32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (u, v) = build_axis_basis(axis);
    let mut verts = Vec::new();
    let half = 0.5 * h;
    for j in 0..sectors {
        let phi = 2.0 * PI * (j as f32) / (sectors as f32);
        let radial = u * (r * phi.cos()) + v * (r * phi.sin());
        let bot = center.coords + radial - axis * half;
        let top = center.coords + radial + axis * half;
        verts.push([bot.x, bot.y, bot.z]);
        verts.push([top.x, top.y, top.z]);
    }
    let mut tris = Vec::new();
    for j in 0..sectors {
        let a = 2 * j;
        let b = 2 * ((j + 1) % sectors);
        tris.push([a, a + 1, b + 1]);
        tris.push([a, b + 1, b]);
    }
    // caps as triangle fans
    let bot_center_idx = verts.len() as u32;
    let bcv = center.coords - axis * half;
    verts.push([bcv.x, bcv.y, bcv.z]);
    let top_center_idx = verts.len() as u32;
    let tcv = center.coords + axis * half;
    verts.push([tcv.x, tcv.y, tcv.z]);
    for j in 0..sectors {
        let a = 2 * j;
        let b = 2 * ((j + 1) % sectors);
        tris.push([bot_center_idx, b, a]);
        tris.push([top_center_idx, a + 1, b + 1]);
    }
    (verts, tris)
}

fn capsule_mesh(
    center: Point3<f32>,
    axis: Vector3<f32>,
    h: f32,
    r: f32,
    sectors: u32,
    cap_stacks: u32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (u, v) = build_axis_basis(axis);
    let half = 0.5 * h;
    let mut verts: Vec<[f32; 3]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();

    // cylinder side
    let row_off = verts.len() as u32;
    for j in 0..=sectors {
        let phi = 2.0 * PI * (j as f32) / (sectors as f32);
        let radial = u * (r * phi.cos()) + v * (r * phi.sin());
        let bot = center.coords + radial - axis * half;
        let top = center.coords + radial + axis * half;
        verts.push([bot.x, bot.y, bot.z]);
        verts.push([top.x, top.y, top.z]);
    }
    for j in 0..sectors {
        let a = row_off + 2 * j;
        let b = row_off + 2 * (j + 1);
        tris.push([a, a + 1, b + 1]);
        tris.push([a, b + 1, b]);
    }
    // top hemisphere centered at center + half*axis
    let top_c = center.coords + axis * half;
    push_hemisphere(top_c, axis, u, v, r, sectors, cap_stacks, true, &mut verts, &mut tris);
    let bot_c = center.coords - axis * half;
    push_hemisphere(bot_c, axis, u, v, r, sectors, cap_stacks, false, &mut verts, &mut tris);
    (verts, tris)
}

fn push_hemisphere(
    apex_center: Vector3<f32>,
    axis: Vector3<f32>,
    u: Vector3<f32>,
    v: Vector3<f32>,
    r: f32,
    sectors: u32,
    stacks: u32,
    upward: bool,
    verts: &mut Vec<[f32; 3]>,
    tris: &mut Vec<[u32; 3]>,
) {
    let dir = if upward { 1.0 } else { -1.0 };
    let row_off = verts.len() as u32;
    for i in 0..=stacks {
        let theta = (PI * 0.5) * (i as f32) / (stacks as f32);
        let st = theta.sin();
        let ct = theta.cos();
        for j in 0..=sectors {
            let phi = 2.0 * PI * (j as f32) / (sectors as f32);
            let radial = u * (r * st * phi.cos()) + v * (r * st * phi.sin());
            let p = apex_center + radial + axis * (dir * r * ct);
            verts.push([p.x, p.y, p.z]);
        }
    }
    let row = sectors + 1;
    for i in 0..stacks {
        for j in 0..sectors {
            let a = row_off + i * row + j;
            let b = a + row;
            tris.push([a, b, a + 1]);
            tris.push([a + 1, b, b + 1]);
        }
    }
}

fn frustum_mesh(
    center: Point3<f32>,
    axis: Vector3<f32>,
    h: f32,
    r_bot: f32,
    r_top: f32,
    sectors: u32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let (u, v) = build_axis_basis(axis);
    let half = 0.5 * h;
    let mut verts = Vec::new();
    for j in 0..sectors {
        let phi = 2.0 * PI * (j as f32) / (sectors as f32);
        let cos = phi.cos();
        let sin = phi.sin();
        let bot = center.coords + u * (r_bot * cos) + v * (r_bot * sin) - axis * half;
        let top = center.coords + u * (r_top * cos) + v * (r_top * sin) + axis * half;
        verts.push([bot.x, bot.y, bot.z]);
        verts.push([top.x, top.y, top.z]);
    }
    let mut tris = Vec::new();
    for j in 0..sectors {
        let a = 2 * j;
        let b = 2 * ((j + 1) % sectors);
        tris.push([a, a + 1, b + 1]);
        tris.push([a, b + 1, b]);
    }
    let bot_c = verts.len() as u32;
    let bv = center.coords - axis * half;
    verts.push([bv.x, bv.y, bv.z]);
    let top_c = verts.len() as u32;
    let tv = center.coords + axis * half;
    verts.push([tv.x, tv.y, tv.z]);
    for j in 0..sectors {
        let a = 2 * j;
        let b = 2 * ((j + 1) % sectors);
        tris.push([bot_c, b, a]);
        tris.push([top_c, a + 1, b + 1]);
    }
    (verts, tris)
}

fn prism_mesh(
    center: Point3<f32>,
    axes: [Vector3<f32>; 3],
    hx: f32,
    hy: f32,
    hzt: f32,
    hzb: f32,
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    // Cross-section vertices in (ay, az) order, walking the trapezoid:
    //   bot-left: (-hy, -hzb), bot-right: (-hy, +hzb)
    //   top-right: (+hy, +hzt), top-left: (+hy, -hzt)
    let cross = [
        (-hy, -hzb),
        (-hy, hzb),
        (hy, hzt),
        (hy, -hzt),
    ];
    let mut verts = Vec::with_capacity(8);
    for &sx in &[-1.0f32, 1.0f32] {
        for &(y, z) in &cross {
            let p = center.coords + axes[0] * (sx * hx) + axes[1] * y + axes[2] * z;
            verts.push([p.x, p.y, p.z]);
        }
    }
    // 8 verts: 0..3 are -ax slab, 4..7 are +ax slab, both walking cross[0..3].
    let mut tris = Vec::new();
    // -ax cap (face) and +ax cap
    tris.extend_from_slice(&[[0, 1, 2], [0, 2, 3]]);
    tris.extend_from_slice(&[[4, 6, 5], [4, 7, 6]]);
    // sides (4 quads)
    let sides = [(0u32, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)];
    for s in &sides {
        tris.push([s.0, s.1, s.2]);
        tris.push([s.0, s.2, s.3]);
    }
    (verts, tris)
}
