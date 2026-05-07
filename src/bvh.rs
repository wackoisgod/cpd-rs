//! Simple AABB BVH over the input mesh's triangles. Supports nearest-face
//! query (closest point on any face + signed distance using face normal).
//! Used by the empty-space-preservation constraint.

use nalgebra::{Point3, Vector3};

#[derive(Clone, Copy)]
struct Node {
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    /// For leaves: face range start. For internal: left-child node index.
    a: u32,
    /// For leaves: number of faces (≥1). For internal: 0 sentinel.
    count: u32,
    /// For internal: right-child node index. Unused on leaves.
    b: u32,
}

pub struct Bvh {
    nodes: Vec<Node>,
    /// Permutation of face indices; leaves reference contiguous ranges.
    face_indices: Vec<u32>,
    /// Cached per-face data for fast queries.
    pub face_centroids: Vec<Point3<f32>>,
    pub face_normals: Vec<Vector3<f32>>,
}

impl Bvh {
    pub fn build(verts: &[Point3<f32>], tris: &[[u32; 3]]) -> Self {
        let nf = tris.len();
        let mut face_centroids = Vec::with_capacity(nf);
        let mut face_normals = Vec::with_capacity(nf);
        let mut face_aabb_min: Vec<[f32; 3]> = Vec::with_capacity(nf);
        let mut face_aabb_max: Vec<[f32; 3]> = Vec::with_capacity(nf);
        for t in tris {
            let a = verts[t[0] as usize];
            let b = verts[t[1] as usize];
            let c = verts[t[2] as usize];
            let centroid = Point3::from((a.coords + b.coords + c.coords) / 3.0);
            let n = (b - a).cross(&(c - a));
            let normal = if n.norm_squared() > 1e-20 {
                n.normalize()
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            face_centroids.push(centroid);
            face_normals.push(normal);
            let lo = [a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y), a.z.min(b.z).min(c.z)];
            let hi = [a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y), a.z.max(b.z).max(c.z)];
            face_aabb_min.push(lo);
            face_aabb_max.push(hi);
        }

        let mut face_indices: Vec<u32> = (0..nf as u32).collect();
        let mut nodes: Vec<Node> = Vec::with_capacity(nf * 2);
        build_recursive(
            0,
            nf,
            &mut face_indices,
            &face_centroids,
            &face_aabb_min,
            &face_aabb_max,
            &mut nodes,
        );
        Self {
            nodes,
            face_indices,
            face_centroids,
            face_normals,
        }
    }

    /// Any-hit ray query: does any face along the ray within `max_dist`
    /// intersect? Used for ambient-occlusion-style exposure scoring.
    pub fn any_hit(
        &self,
        verts: &[Point3<f32>],
        tris: &[[u32; 3]],
        origin: Point3<f32>,
        dir: Vector3<f32>,
        max_dist: f32,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        // pre-compute reciprocals for the slab test
        let inv_dir = Vector3::new(
            if dir.x.abs() > 1e-20 { 1.0 / dir.x } else { f32::INFINITY },
            if dir.y.abs() > 1e-20 { 1.0 / dir.y } else { f32::INFINITY },
            if dir.z.abs() > 1e-20 { 1.0 / dir.z } else { f32::INFINITY },
        );
        self.descend_ray(0, verts, tris, origin, dir, inv_dir, max_dist)
    }

    fn descend_ray(
        &self,
        node_idx: u32,
        verts: &[Point3<f32>],
        tris: &[[u32; 3]],
        origin: Point3<f32>,
        dir: Vector3<f32>,
        inv_dir: Vector3<f32>,
        max_dist: f32,
    ) -> bool {
        let node = &self.nodes[node_idx as usize];
        if !ray_aabb_hit(&node.aabb_min, &node.aabb_max, &origin, &inv_dir, max_dist) {
            return false;
        }
        if node.count > 0 {
            // leaf: ray-triangle test for each face
            let s = node.a as usize;
            let e = s + node.count as usize;
            for &fi in &self.face_indices[s..e] {
                let t = tris[fi as usize];
                if let Some(d) = ray_triangle(
                    origin,
                    dir,
                    verts[t[0] as usize],
                    verts[t[1] as usize],
                    verts[t[2] as usize],
                ) {
                    if d > 1e-5 && d <= max_dist {
                        return true;
                    }
                }
            }
            return false;
        }
        let l = node.a;
        let r = node.b;
        if self.descend_ray(l, verts, tris, origin, dir, inv_dir, max_dist) {
            return true;
        }
        self.descend_ray(r, verts, tris, origin, dir, inv_dir, max_dist)
    }

    /// Nearest-face query. Returns (closest_point_on_face, face_normal, signed_distance).
    /// Signed distance is positive when the query is on the side the face normal points
    /// toward (i.e., "outside" by the face's orientation), negative on the back side.
    pub fn nearest_face(
        &self,
        verts: &[Point3<f32>],
        tris: &[[u32; 3]],
        query: Point3<f32>,
    ) -> (Point3<f32>, Vector3<f32>, f32) {
        let mut best_d2 = f32::INFINITY;
        let mut best_pt = Point3::origin();
        let mut best_face: u32 = 0;
        if !self.nodes.is_empty() {
            self.descend(0, verts, tris, query, &mut best_d2, &mut best_pt, &mut best_face);
        }
        let n = self.face_normals[best_face as usize];
        let signed = (query - best_pt).dot(&n);
        let d = best_d2.sqrt();
        let signed = if signed.is_sign_negative() { -d } else { d };
        (best_pt, n, signed)
    }

    fn descend(
        &self,
        node_idx: u32,
        verts: &[Point3<f32>],
        tris: &[[u32; 3]],
        query: Point3<f32>,
        best_d2: &mut f32,
        best_pt: &mut Point3<f32>,
        best_face: &mut u32,
    ) {
        let node = &self.nodes[node_idx as usize];
        let dmin2 = aabb_dist_sq(&node.aabb_min, &node.aabb_max, &query);
        if dmin2 >= *best_d2 {
            return;
        }
        if node.count > 0 {
            // leaf
            let s = node.a as usize;
            let e = s + node.count as usize;
            for &fi in &self.face_indices[s..e] {
                let t = tris[fi as usize];
                let cp = closest_point_on_triangle(
                    query,
                    verts[t[0] as usize],
                    verts[t[1] as usize],
                    verts[t[2] as usize],
                );
                let d2 = (query - cp).norm_squared();
                if d2 < *best_d2 {
                    *best_d2 = d2;
                    *best_pt = cp;
                    *best_face = fi;
                }
            }
            return;
        }
        // internal: descend into the closer child first for better pruning
        let l = node.a;
        let r = node.b;
        let ln = &self.nodes[l as usize];
        let rn = &self.nodes[r as usize];
        let ld = aabb_dist_sq(&ln.aabb_min, &ln.aabb_max, &query);
        let rd = aabb_dist_sq(&rn.aabb_min, &rn.aabb_max, &query);
        if ld <= rd {
            self.descend(l, verts, tris, query, best_d2, best_pt, best_face);
            self.descend(r, verts, tris, query, best_d2, best_pt, best_face);
        } else {
            self.descend(r, verts, tris, query, best_d2, best_pt, best_face);
            self.descend(l, verts, tris, query, best_d2, best_pt, best_face);
        }
    }
}

fn aabb_dist_sq(lo: &[f32; 3], hi: &[f32; 3], p: &Point3<f32>) -> f32 {
    let mut d2 = 0.0f32;
    for i in 0..3 {
        let v = p[i];
        if v < lo[i] {
            let dv = lo[i] - v;
            d2 += dv * dv;
        } else if v > hi[i] {
            let dv = v - hi[i];
            d2 += dv * dv;
        }
    }
    d2
}

fn build_recursive(
    start: usize,
    end: usize,
    face_indices: &mut [u32],
    face_centroids: &[Point3<f32>],
    face_aabb_min: &[[f32; 3]],
    face_aabb_max: &[[f32; 3]],
    nodes: &mut Vec<Node>,
) -> u32 {
    let (aabb_min, aabb_max) = compute_aabb(start, end, face_indices, face_aabb_min, face_aabb_max);
    let count = end - start;
    if count <= 4 {
        let idx = nodes.len() as u32;
        nodes.push(Node {
            aabb_min,
            aabb_max,
            a: start as u32,
            count: count as u32,
            b: 0,
        });
        return idx;
    }
    // split on longest axis at midpoint of centroids
    let ext = [
        aabb_max[0] - aabb_min[0],
        aabb_max[1] - aabb_min[1],
        aabb_max[2] - aabb_min[2],
    ];
    let axis = if ext[0] >= ext[1] && ext[0] >= ext[2] {
        0usize
    } else if ext[1] >= ext[2] {
        1
    } else {
        2
    };
    let mid = 0.5 * (aabb_min[axis] + aabb_max[axis]);
    // partition in-place
    let mut i = start;
    let mut j = end;
    while i < j {
        if face_centroids[face_indices[i] as usize][axis] < mid {
            i += 1;
        } else {
            j -= 1;
            face_indices.swap(i, j);
        }
    }
    let mut split = i;
    if split == start || split == end {
        // bad split — fall back to median
        split = start + count / 2;
    }
    // reserve our slot first so children get later indices
    let our_idx = nodes.len() as u32;
    nodes.push(Node {
        aabb_min,
        aabb_max,
        a: 0,
        count: 0,
        b: 0,
    });
    let left = build_recursive(start, split, face_indices, face_centroids, face_aabb_min, face_aabb_max, nodes);
    let right = build_recursive(split, end, face_indices, face_centroids, face_aabb_min, face_aabb_max, nodes);
    nodes[our_idx as usize].a = left;
    nodes[our_idx as usize].b = right;
    nodes[our_idx as usize].count = 0;
    our_idx
}

fn compute_aabb(
    start: usize,
    end: usize,
    face_indices: &[u32],
    face_aabb_min: &[[f32; 3]],
    face_aabb_max: &[[f32; 3]],
) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for i in start..end {
        let f = face_indices[i] as usize;
        for k in 0..3 {
            if face_aabb_min[f][k] < lo[k] {
                lo[k] = face_aabb_min[f][k];
            }
            if face_aabb_max[f][k] > hi[k] {
                hi[k] = face_aabb_max[f][k];
            }
        }
    }
    (lo, hi)
}

/// Standard slab test: does the ray hit the AABB within [0, max_dist]?
fn ray_aabb_hit(
    lo: &[f32; 3],
    hi: &[f32; 3],
    origin: &Point3<f32>,
    inv_dir: &Vector3<f32>,
    max_dist: f32,
) -> bool {
    let mut tmin = 0.0f32;
    let mut tmax = max_dist;
    for i in 0..3 {
        let t1 = (lo[i] - origin[i]) * inv_dir[i];
        let t2 = (hi[i] - origin[i]) * inv_dir[i];
        let (lo_t, hi_t) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        if lo_t > tmin {
            tmin = lo_t;
        }
        if hi_t < tmax {
            tmax = hi_t;
        }
        if tmin > tmax {
            return false;
        }
    }
    true
}

/// Möller-Trumbore ray-triangle intersection. Returns the parametric `t`
/// along the ray to the hit, if any.
fn ray_triangle(
    origin: Point3<f32>,
    dir: Vector3<f32>,
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let h = dir.cross(&edge2);
    let det = edge1.dot(&h);
    if det.abs() < 1e-12 {
        return None; // parallel
    }
    let inv_det = 1.0 / det;
    let s = origin - a;
    let u = inv_det * s.dot(&h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = s.cross(&edge1);
    let v = inv_det * dir.dot(&q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = inv_det * edge2.dot(&q);
    if t > 0.0 {
        Some(t)
    } else {
        None
    }
}

/// Ericson, "Real-Time Collision Detection". Returns the point on triangle
/// abc nearest to p (clamped to triangle interior or one of its edges).
fn closest_point_on_triangle(
    p: Point3<f32>,
    a: Point3<f32>,
    b: Point3<f32>,
    c: Point3<f32>,
) -> Point3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a + ab * v;
    }
    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a + ac * w;
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a + ab * v + ac * w
}
