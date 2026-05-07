use crate::decomp::Primitive;
use crate::prim;
use anyhow::Result;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn write_obbs_obj(path: &Path, prims: &[Primitive]) -> Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    writeln!(w, "# convex primitive decomposition output")?;
    let mut vbase: usize = 0;
    let mut emitted = 0usize;
    for (pi, p) in prims.iter().enumerate() {
        if !p.alive {
            continue;
        }
        let (verts, tris) = prim::tessellate(&p.prim);
        writeln!(w, "g {}_{}", kind_tag(&p.prim), pi)?;
        for v in &verts {
            writeln!(w, "v {} {} {}", v[0], v[1], v[2])?;
        }
        for t in &tris {
            writeln!(
                w,
                "f {} {} {}",
                vbase + t[0] as usize + 1,
                vbase + t[1] as usize + 1,
                vbase + t[2] as usize + 1
            )?;
        }
        vbase += verts.len();
        emitted += 1;
    }
    eprintln!("wrote {} primitives to {}", emitted, path.display());
    Ok(())
}

fn kind_tag(prim: &prim::Prim) -> &'static str {
    match prim {
        prim::Prim::Obb { .. } => "obb",
        prim::Prim::Sphere { .. } => "sphere",
        prim::Prim::Cylinder { .. } => "cyl",
        prim::Prim::Capsule { .. } => "cap",
        prim::Prim::Frustum { .. } => "frustum",
        prim::Prim::Prism { .. } => "prism",
    }
}
