use anyhow::{Context, Result};
use nalgebra::{Matrix4, Point3, Vector3};
use std::collections::HashMap;
use std::path::Path;

pub struct Mesh {
    pub verts: Vec<Point3<f32>>,
    pub tris: Vec<[u32; 3]>,
}

pub fn load_glb(path: &Path) -> Result<Mesh> {
    let (doc, buffers, _images) =
        gltf::import(path).with_context(|| format!("loading {}", path.display()))?;

    let mut out = Mesh {
        verts: Vec::new(),
        tris: Vec::new(),
    };
    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .context("glTF has no scene")?;
    for node in scene.nodes() {
        process_node(&node, Matrix4::identity(), &buffers, &mut out);
    }
    Ok(out)
}

fn process_node(
    node: &gltf::Node,
    parent: Matrix4<f32>,
    buffers: &[gltf::buffer::Data],
    out: &mut Mesh,
) {
    let local = Matrix4::from_column_slice(&flatten_4x4(node.transform().matrix()));
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                eprintln!(
                    "skipping primitive with mode {:?} (only triangles supported)",
                    prim.mode()
                );
                continue;
            }
            let reader = prim.reader(|b| Some(&buffers[b.index()]));
            let positions: Vec<[f32; 3]> = match reader.read_positions() {
                Some(p) => p.collect(),
                None => continue,
            };
            let base = out.verts.len() as u32;
            for p in &positions {
                let v = world.transform_point(&Point3::new(p[0], p[1], p[2]));
                out.verts.push(v);
            }
            match reader.read_indices() {
                Some(idx) => {
                    let idx: Vec<u32> = idx.into_u32().collect();
                    for tri in idx.chunks_exact(3) {
                        out.tris.push([base + tri[0], base + tri[1], base + tri[2]]);
                    }
                }
                None => {
                    let n = positions.len() as u32;
                    let mut i = 0u32;
                    while i + 3 <= n {
                        out.tris.push([base + i, base + i + 1, base + i + 2]);
                        i += 3;
                    }
                }
            }
        }
    }

    for child in node.children() {
        process_node(&child, world, buffers, out);
    }
}

fn flatten_4x4(m: [[f32; 4]; 4]) -> [f32; 16] {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = m[c][r];
        }
    }
    out
}

/// Collapse vertices that share the same quantized position. Game-mesh
/// imports (e.g. .glb) duplicate vertices at UV/normal seams so the
/// topology looks disconnected; this restores the physical adjacency.
pub fn weld_vertices(mesh: &mut Mesh, eps: f32) -> usize {
    let inv_eps = 1.0 / eps;
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut new_verts: Vec<Point3<f32>> = Vec::new();
    let mut remap: Vec<u32> = Vec::with_capacity(mesh.verts.len());
    for v in &mesh.verts {
        let key = (
            (v.x * inv_eps).round() as i64,
            (v.y * inv_eps).round() as i64,
            (v.z * inv_eps).round() as i64,
        );
        let idx = match map.get(&key) {
            Some(&i) => i,
            None => {
                let i = new_verts.len() as u32;
                new_verts.push(*v);
                map.insert(key, i);
                i
            }
        };
        remap.push(idx);
    }
    let collapsed = mesh.verts.len() - new_verts.len();
    mesh.verts = new_verts;
    let mut new_tris = Vec::with_capacity(mesh.tris.len());
    for t in &mesh.tris {
        let a = remap[t[0] as usize];
        let b = remap[t[1] as usize];
        let c = remap[t[2] as usize];
        if a == b || b == c || a == c {
            continue; // skip triangles that became degenerate after welding
        }
        new_tris.push([a, b, c]);
    }
    mesh.tris = new_tris;
    collapsed
}

pub struct Adjacency {
    pub neighbors: Vec<Vec<u32>>,
}

pub fn build_adjacency(tris: &[[u32; 3]]) -> Adjacency {
    let mut edge_to_tri: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ti, t) in tris.iter().enumerate() {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if e.0 < e.1 { (e.0, e.1) } else { (e.1, e.0) };
            edge_to_tri.entry(key).or_default().push(ti as u32);
        }
    }
    let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); tris.len()];
    for tris_on_edge in edge_to_tri.values() {
        for &a in tris_on_edge {
            for &b in tris_on_edge {
                if a != b && !neighbors[a as usize].contains(&b) {
                    neighbors[a as usize].push(b);
                }
            }
        }
    }
    Adjacency { neighbors }
}

/// Per-face ambient-occlusion-style exposure score in [0, 1].
/// 1.0 = face is unobstructed (clear sky in its outward hemisphere).
/// 0.0 = face is buried in mesh interior (every outward direction blocked).
///
/// Used by the shell-aware orientation refinement: interior geometry on
/// kitbashed/scanned assets pollutes the area-weighted normal quadric and
/// PCA. Down-weighting buried faces in Q and filtering them out of PCA
/// makes the orientation reflect the visible shell, while containment
/// fitting still uses every subsumed vertex (paper enclosure guarantee).
pub fn compute_face_exposure(mesh: &Mesh, bvh: &crate::bvh::Bvh, n_dirs: usize) -> Vec<f32> {
    let nf = mesh.tris.len();
    let diag = aabb_diag(&mesh.verts).max(1.0);
    let max_dist = diag * 0.5;
    let eps_offset = diag * 1e-4;

    // Stratified directions on the unit sphere via Fibonacci spiral.
    let golden = std::f32::consts::PI * (3.0 - (5.0_f32).sqrt());
    let dirs: Vec<Vector3<f32>> = (0..n_dirs)
        .map(|i| {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n_dirs as f32;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let theta = golden * i as f32;
            Vector3::new(r * theta.cos(), r * theta.sin(), z)
        })
        .collect();

    use rayon::prelude::*;
    (0..nf)
        .into_par_iter()
        .map(|fi| {
            let t = mesh.tris[fi];
            let a = mesh.verts[t[0] as usize];
            let b = mesh.verts[t[1] as usize];
            let c = mesh.verts[t[2] as usize];
            let centroid = Point3::from((a.coords + b.coords + c.coords) / 3.0);
            let n = (b - a).cross(&(c - a));
            let n_len2 = n.norm_squared();
            if n_len2 < 1e-20 {
                return 0.5;
            }
            let normal = n / n_len2.sqrt();
            // Origin slightly above the face along its normal so we don't
            // self-intersect with the source face.
            let origin = centroid + normal * eps_offset;
            let mut total = 0u32;
            let mut exposed = 0u32;
            for d in &dirs {
                let cosang = d.dot(&normal);
                // Only sample the +normal hemisphere — interior side
                // exposure isn't relevant.
                if cosang <= 0.0 {
                    continue;
                }
                total += 1;
                if !bvh.any_hit(&mesh.verts, &mesh.tris, origin, *d, max_dist) {
                    exposed += 1;
                }
            }
            if total == 0 {
                0.5
            } else {
                exposed as f32 / total as f32
            }
        })
        .collect()
}

/// Find connected components of the topology graph using DSU. Returns
/// a face → component-id vector. Used by proximity-merge generation to
/// pair up nearby disconnected components.
pub fn find_components(adj: &Adjacency) -> Vec<u32> {
    let nf = adj.neighbors.len();
    let mut dsu = crate::dsu::Dsu::new(nf);
    for (fi, ns) in adj.neighbors.iter().enumerate() {
        for &nj in ns {
            dsu.union(fi as u32, nj);
        }
    }
    // Compress to consecutive component ids.
    let mut canonical: HashMap<u32, u32> = HashMap::new();
    let mut comp_id = Vec::with_capacity(nf);
    for fi in 0..nf {
        let root = dsu.find(fi as u32);
        let id = match canonical.get(&root) {
            Some(&id) => id,
            None => {
                let id = canonical.len() as u32;
                canonical.insert(root, id);
                id
            }
        };
        comp_id.push(id);
    }
    comp_id
}

/// Per-component pre-computation used by proximity-merge generation.
pub struct ComponentSummary {
    /// Number of components.
    pub n: usize,
    /// face_idx → component id
    pub face_comp: Vec<u32>,
    /// per-component centroid (area-weighted)
    pub centroids: Vec<Point3<f32>>,
    /// per-component AABB (lo, hi)
    pub aabbs: Vec<(Point3<f32>, Point3<f32>)>,
    /// per-component dominant normal (largest eigenvector of summed Q),
    /// roughly the "facing direction" of the surface
    pub normals: Vec<Vector3<f32>>,
    /// face indices grouped by component
    pub faces_per: Vec<Vec<u32>>,
}

pub fn summarize_components(mesh: &Mesh, adj: &Adjacency) -> ComponentSummary {
    let face_comp = find_components(adj);
    let n = (face_comp.iter().copied().max().map(|m| m + 1).unwrap_or(0)) as usize;
    let mut faces_per: Vec<Vec<u32>> = vec![Vec::new(); n];
    for (fi, &c) in face_comp.iter().enumerate() {
        faces_per[c as usize].push(fi as u32);
    }

    // Per-component area-weighted centroid + AABB
    let mut centroids: Vec<Point3<f32>> = vec![Point3::origin(); n];
    let mut aabbs: Vec<(Point3<f32>, Point3<f32>)> = vec![
        (
            Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        );
        n
    ];
    let mut total_area: Vec<f32> = vec![0.0; n];
    let mut summed_q: Vec<nalgebra::Matrix3<f32>> = vec![nalgebra::Matrix3::zeros(); n];

    for (fi, t) in mesh.tris.iter().enumerate() {
        let c = face_comp[fi] as usize;
        let a = mesh.verts[t[0] as usize];
        let b = mesh.verts[t[1] as usize];
        let cp = mesh.verts[t[2] as usize];
        let cross = (b - a).cross(&(cp - a));
        let area = 0.5 * cross.norm();
        if area < 1e-20 {
            continue;
        }
        let face_centroid = (a.coords + b.coords + cp.coords) / 3.0;
        centroids[c].coords += face_centroid * area;
        total_area[c] += area;
        for &v in &[a, b, cp] {
            for k in 0..3 {
                if v[k] < aabbs[c].0[k] {
                    aabbs[c].0[k] = v[k];
                }
                if v[k] > aabbs[c].1[k] {
                    aabbs[c].1[k] = v[k];
                }
            }
        }
        let n_vec = cross / cross.norm();
        summed_q[c] += area * n_vec * n_vec.transpose();
    }
    for c in 0..n {
        if total_area[c] > 0.0 {
            centroids[c].coords /= total_area[c];
        }
    }

    let mut normals: Vec<Vector3<f32>> = Vec::with_capacity(n);
    for q in &summed_q {
        let sym = (q + q.transpose()) * 0.5;
        let dec = nalgebra::SymmetricEigen::new(sym);
        // largest eigenvalue's eigenvector is the area-weighted normal
        let mut max_i = 0;
        for i in 1..3 {
            if dec.eigenvalues[i].abs() > dec.eigenvalues[max_i].abs() {
                max_i = i;
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

    ComponentSummary {
        n,
        face_comp,
        centroids,
        aabbs,
        normals,
        faces_per,
    }
}

/// Build proximity edges between disconnected components: for each
/// component, find its `k` nearest neighbors by AABB-to-AABB distance,
/// reject pairs whose dominant normals differ by more than `max_angle_rad`,
/// and produce one representative face-edge per accepted pair (closest
/// face-centroid pair).
pub fn build_proximity_edges(
    mesh: &Mesh,
    summary: &ComponentSummary,
    k: usize,
    max_dist: f32,
    max_angle_rad: f32,
) -> Vec<(u32, u32)> {
    if summary.n < 2 {
        return Vec::new();
    }

    // Per-component face centroids — precomputed once for closest-pair
    // search inside accepted component pairs.
    let mut face_centroid: Vec<Point3<f32>> = Vec::with_capacity(mesh.tris.len());
    for t in &mesh.tris {
        let a = mesh.verts[t[0] as usize].coords;
        let b = mesh.verts[t[1] as usize].coords;
        let c = mesh.verts[t[2] as usize].coords;
        face_centroid.push(Point3::from((a + b + c) / 3.0));
    }

    let cos_min = max_angle_rad.cos();
    let mut edges: HashMap<(u32, u32), f32> = HashMap::new();

    // For each component, find k-nearest neighbours by AABB-to-AABB
    // distance. O(C^2) but C is in the hundreds at most.
    for i in 0..summary.n {
        let mut dists: Vec<(usize, f32)> = (0..summary.n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    j,
                    aabb_to_aabb_distance(&summary.aabbs[i], &summary.aabbs[j]),
                )
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, d) in dists.iter().take(k) {
            if d > max_dist {
                continue;
            }
            // Orientation guard: components with dominant normals that
            // differ too much shouldn't merge (e.g. roof-vs-wall across
            // a gap).
            let cos = summary.normals[i].dot(&summary.normals[j]).abs();
            if cos < cos_min {
                continue;
            }
            // Find the closest face-centroid pair between the two components.
            let faces_i = &summary.faces_per[i];
            let faces_j = &summary.faces_per[j];
            let mut best: (u32, u32, f32) = (faces_i[0], faces_j[0], f32::INFINITY);
            for &fi in faces_i {
                for &fj in faces_j {
                    let dd =
                        (face_centroid[fi as usize] - face_centroid[fj as usize]).norm_squared();
                    if dd < best.2 {
                        best = (fi, fj, dd);
                    }
                }
            }
            let key = if best.0 < best.1 {
                (best.0, best.1)
            } else {
                (best.1, best.0)
            };
            edges.entry(key).or_insert(best.2);
        }
    }

    edges.into_iter().map(|(k, _)| k).collect()
}

fn aabb_to_aabb_distance(a: &(Point3<f32>, Point3<f32>), b: &(Point3<f32>, Point3<f32>)) -> f32 {
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

/// Per-mesh detection of "sharp" feature edges: edges whose two incident
/// faces meet at a dihedral angle above some threshold (i.e., a crease).
/// The directions of these edges are useful as a third orientation
/// candidate during post-merge refit — building corners, hood-line creases,
/// stair tread edges, etc. align primitives well.
pub struct SharpEdges {
    /// For each face, the (unit) directions of its sharp edges.
    pub per_face: Vec<Vec<Vector3<f32>>>,
}

/// Compute a per-mesh adaptive dihedral threshold for sharp-edge
/// detection. The fixed 30° cutoff catches too many small ridges on
/// organic meshes (rocks, terrain) where most dihedrals are tiny; the
/// 95th percentile of the actual dihedral distribution is mesh-aware
/// without needing user tuning. Clamped to [30°, 60°] to keep behaviour
/// in a sane range.
pub fn adaptive_sharp_threshold(mesh: &Mesh) -> f32 {
    let face_normals: Vec<Vector3<f32>> = mesh
        .tris
        .iter()
        .map(|t| {
            let a = mesh.verts[t[0] as usize];
            let b = mesh.verts[t[1] as usize];
            let c = mesh.verts[t[2] as usize];
            let n = (b - a).cross(&(c - a));
            if n.norm_squared() > 1e-20 {
                n.normalize()
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            }
        })
        .collect();

    let mut edge_to_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (fi, t) in mesh.tris.iter().enumerate() {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if e.0 < e.1 { (e.0, e.1) } else { (e.1, e.0) };
            edge_to_faces.entry(key).or_default().push(fi as u32);
        }
    }
    let mut angles: Vec<f32> = Vec::new();
    for faces in edge_to_faces.values() {
        if faces.len() != 2 {
            continue;
        }
        let dot = face_normals[faces[0] as usize]
            .dot(&face_normals[faces[1] as usize])
            .clamp(-1.0, 1.0);
        angles.push(dot.acos());
    }
    if angles.is_empty() {
        return std::f32::consts::FRAC_PI_6;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = angles[(angles.len() * 95) / 100];
    let min_t = std::f32::consts::FRAC_PI_6; // 30°
    let max_t = std::f32::consts::FRAC_PI_3; // 60°
    p95.clamp(min_t, max_t)
}

pub fn build_sharp_edges(
    mesh: &Mesh,
    dihedral_threshold_rad: f32,
    face_shell_mask: Option<&[bool]>,
) -> SharpEdges {
    let nf = mesh.tris.len();
    let face_normals: Vec<Vector3<f32>> = mesh
        .tris
        .iter()
        .map(|t| {
            let a = mesh.verts[t[0] as usize];
            let b = mesh.verts[t[1] as usize];
            let c = mesh.verts[t[2] as usize];
            let n = (b - a).cross(&(c - a));
            if n.norm_squared() > 1e-20 {
                n.normalize()
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            }
        })
        .collect();

    let mut edge_to_faces: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (fi, t) in mesh.tris.iter().enumerate() {
        for e in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let key = if e.0 < e.1 { (e.0, e.1) } else { (e.1, e.0) };
            edge_to_faces.entry(key).or_default().push(fi as u32);
        }
    }

    let cos_threshold = dihedral_threshold_rad.cos();
    let mut per_face: Vec<Vec<Vector3<f32>>> = vec![Vec::new(); nf];
    for (key, faces) in &edge_to_faces {
        if faces.len() != 2 {
            continue; // boundary or non-manifold edge — treat as not sharp
        }
        // When shell-mask is provided, skip edges with any interior face —
        // creases between two interior faces aren't features we want to
        // align primitives to.
        if let Some(mask) = face_shell_mask {
            if !mask[faces[0] as usize] || !mask[faces[1] as usize] {
                continue;
            }
        }
        let n0 = &face_normals[faces[0] as usize];
        let n1 = &face_normals[faces[1] as usize];
        if n0.dot(n1) < cos_threshold {
            let u = key.0 as usize;
            let v = key.1 as usize;
            let dir = (mesh.verts[v] - mesh.verts[u]).normalize();
            per_face[faces[0] as usize].push(dir);
            per_face[faces[1] as usize].push(dir);
        }
    }
    SharpEdges { per_face }
}

pub fn aabb(verts: &[Point3<f32>]) -> (Point3<f32>, Point3<f32>) {
    if verts.is_empty() {
        return (Point3::origin(), Point3::origin());
    }
    let mut lo = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut hi = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for v in verts {
        for i in 0..3 {
            if v[i] < lo[i] {
                lo[i] = v[i];
            }
            if v[i] > hi[i] {
                hi[i] = v[i];
            }
        }
    }
    (lo, hi)
}

pub fn aabb_diag(verts: &[Point3<f32>]) -> f32 {
    if verts.is_empty() {
        return 0.0;
    }
    let mut lo = Vector3::repeat(f32::INFINITY);
    let mut hi = Vector3::repeat(f32::NEG_INFINITY);
    for v in verts {
        for i in 0..3 {
            if v[i] < lo[i] {
                lo[i] = v[i];
            }
            if v[i] > hi[i] {
                hi[i] = v[i];
            }
        }
    }
    (hi - lo).norm()
}
