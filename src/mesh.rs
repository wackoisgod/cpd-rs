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

    let mut out = Mesh { verts: Vec::new(), tris: Vec::new() };
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

/// Per-mesh detection of "sharp" feature edges: edges whose two incident
/// faces meet at a dihedral angle above some threshold (i.e., a crease).
/// The directions of these edges are useful as a third orientation
/// candidate during post-merge refit — building corners, hood-line creases,
/// stair tread edges, etc. align primitives well.
pub struct SharpEdges {
    /// For each face, the (unit) directions of its sharp edges.
    pub per_face: Vec<Vec<Vector3<f32>>>,
}

pub fn build_sharp_edges(mesh: &Mesh, dihedral_threshold_rad: f32) -> SharpEdges {
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
