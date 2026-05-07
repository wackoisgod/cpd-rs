mod bvh;
mod decomp;
mod dsu;
mod io_obj;
mod mesh;
mod metrics;
mod prim;
mod viewer;

use anyhow::{Context, Result};
use prim::PrimMask;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug)]
struct CliArgs {
    mesh_path: PathBuf,
    target_n: usize,
    out_obj: PathBuf,
    viewer_html: Option<PathBuf>,
    volume_threshold_frac: f32, // fraction of input AABB volume
    enable_mask: PrimMask,
    cull_redundant: bool,
    empty_space: Option<(f32, f32)>, // (max_bridge_fraction, dist_threshold_frac_of_diag)
    refine_orient: bool,
    quality_beta: f32,
    shell_aware: bool,
    proximity: Option<(f32, usize, f32)>, // (max_dist_frac, k, max_angle_rad)
    weighted_cost: bool,
    rebalance: Option<usize>,
    reject_pancakes: bool,
    strip_thin_obbs: Option<f32>,
    feasibility: Option<f32>,
    split_worst: Option<(f32, usize)>,
    cull_overlap: Option<f32>,
    axis_align: bool,
    world_axis_align: bool,
    tangent_eps: f32,
    metrics: bool,
    metrics_json: Option<PathBuf>,
}

fn parse_args() -> Result<CliArgs> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut viewer_html: Option<PathBuf> = None;
    let mut volume_threshold_frac = f32::INFINITY;
    let mut mask = PrimMask::all();
    let mut obb_only = false;
    let mut cull_redundant = true;
    let mut empty_space: Option<(f32, f32)> = None;
    let mut refine_orient = true;
    let mut quality_beta: f32 = 0.0;
    let mut shell_aware = false;
    let mut proximity: Option<(f32, usize, f32)> = None;
    let mut weighted_cost = false;
    let mut rebalance: Option<usize> = None;
    let mut reject_pancakes = false;
    let mut strip_thin_obbs: Option<f32> = None;
    let mut feasibility: Option<f32> = None;
    let mut split_worst: Option<(f32, usize)> = None;
    let mut cull_overlap: Option<f32> = None;
    let mut axis_align = false;
    let mut world_axis_align = false;
    let mut tangent_eps: f32 = 0.01; // paper §3.4 default
    let mut metrics_flag = false;
    let mut metrics_json: Option<PathBuf> = None;
    let mut positional: Vec<String> = Vec::new();

    while let Some(arg) = args.first().cloned() {
        match arg.as_str() {
            "--viewer" => {
                args.remove(0);
                let p = args.first().cloned().context("--viewer needs a path")?;
                args.remove(0);
                viewer_html = Some(PathBuf::from(p));
            }
            "--volume-threshold" => {
                args.remove(0);
                let v: String = args.first().cloned().context("--volume-threshold needs f32")?;
                args.remove(0);
                volume_threshold_frac = v.parse().context("not a float")?;
            }
            "--obb-only" => {
                args.remove(0);
                obb_only = true;
            }
            "--no-cull" => {
                args.remove(0);
                cull_redundant = false;
            }
            "--no-refine" => {
                args.remove(0);
                refine_orient = false;
            }
            "--quality" => {
                args.remove(0);
                let v = args.first().cloned().context("--quality needs a beta value")?;
                args.remove(0);
                quality_beta = v.parse().context("--quality beta must be f32")?;
            }
            "--shell" => {
                args.remove(0);
                shell_aware = true;
            }
            "--weighted-cost" => {
                args.remove(0);
                weighted_cost = true;
            }
            "--rebalance" => {
                args.remove(0);
                rebalance = Some(rebalance.unwrap_or(5));
            }
            "--reject-pancakes" => {
                args.remove(0);
                reject_pancakes = true;
            }
            "--strip-thin" => {
                args.remove(0);
                strip_thin_obbs = Some(strip_thin_obbs.unwrap_or(1e-4));
            }
            "--strip-thin-threshold" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--strip-thin-threshold needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                strip_thin_obbs = Some(f);
            }
            "--feasibility" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--feasibility needs a max-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                feasibility = Some(f);
            }
            "--split-worst" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--split-worst needs a threshold-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev_max = split_worst.map(|(_, m)| m).unwrap_or(32);
                split_worst = Some((f, prev_max));
            }
            "--split-worst-max" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--split-worst-max needs an integer")?;
                args.remove(0);
                let m: usize = v.parse().context("not an integer")?;
                let prev_frac = split_worst.map(|(f, _)| f).unwrap_or(0.05);
                split_worst = Some((prev_frac, m));
            }
            "--cull-overlap" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--cull-overlap needs a 0..1 fraction")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                cull_overlap = Some(f);
            }
            "--axis-align" => {
                args.remove(0);
                axis_align = true;
            }
            "--world-axis" => {
                args.remove(0);
                world_axis_align = true;
            }
            "--no-tangent-eps" => {
                args.remove(0);
                tangent_eps = 0.0;
            }
            "--tangent-eps" => {
                args.remove(0);
                let v = args.first().cloned().context("--tangent-eps needs f32")?;
                args.remove(0);
                tangent_eps = v.parse().context("not a float")?;
            }
            "--rebalance-passes" => {
                args.remove(0);
                let v = args.first().cloned().context("--rebalance-passes needs usize")?;
                args.remove(0);
                rebalance = Some(v.parse().context("not an integer")?);
            }
            "--proximity" => {
                args.remove(0);
                // Default knobs: 5% of diag, k=2 nearest, 45° angle limit.
                proximity = Some((0.05, 2, std::f32::consts::FRAC_PI_4));
            }
            "--proximity-r" => {
                args.remove(0);
                let v = args.first().cloned().context("--proximity-r needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = proximity.unwrap_or((0.05, 2, std::f32::consts::FRAC_PI_4));
                proximity = Some((f, prev.1, prev.2));
            }
            "--proximity-k" => {
                args.remove(0);
                let v = args.first().cloned().context("--proximity-k needs usize")?;
                args.remove(0);
                let k: usize = v.parse().context("not an integer")?;
                let prev = proximity.unwrap_or((0.05, 2, std::f32::consts::FRAC_PI_4));
                proximity = Some((prev.0, k, prev.2));
            }
            "--proximity-angle" => {
                args.remove(0);
                let v = args.first().cloned().context("--proximity-angle needs degrees")?;
                args.remove(0);
                let deg: f32 = v.parse().context("not a float")?;
                let prev = proximity.unwrap_or((0.05, 2, std::f32::consts::FRAC_PI_4));
                proximity = Some((prev.0, prev.1, deg.to_radians()));
            }
            "--metrics" => {
                args.remove(0);
                metrics_flag = true;
            }
            "--metrics-json" => {
                args.remove(0);
                let p = args.first().cloned().context("--metrics-json needs a path")?;
                args.remove(0);
                metrics_json = Some(PathBuf::from(p));
                metrics_flag = true;
            }
            "--empty-space" => {
                args.remove(0);
                // sensible defaults: reject merges that bridge >25% interior
                // mass into empty space, with "empty" defined as >1% of
                // scene diagonal outside the input mesh.
                empty_space = Some((0.25, 0.01));
            }
            "--empty-space-fraction" => {
                args.remove(0);
                let v = args.first().cloned().context("--empty-space-fraction needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = empty_space.unwrap_or((0.25, 0.01));
                empty_space = Some((f, prev.1));
            }
            "--empty-space-distance" => {
                args.remove(0);
                let v = args.first().cloned().context("--empty-space-distance needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = empty_space.unwrap_or((0.25, 0.01));
                empty_space = Some((prev.0, f));
            }
            "--no-sphere" => { args.remove(0); mask.sphere = false; }
            "--no-cylinder" => { args.remove(0); mask.cylinder = false; }
            "--no-capsule" => { args.remove(0); mask.capsule = false; }
            "--no-frustum" => { args.remove(0); mask.frustum = false; }
            "--no-prism" => { args.remove(0); mask.prism = false; }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {
                positional.push(arg);
                args.remove(0);
            }
        }
    }

    if positional.len() < 2 {
        print_usage();
        anyhow::bail!("missing positional args");
    }
    let mesh_path = PathBuf::from(&positional[0]);
    let target_n: usize = positional[1].parse().context("target_n must be a positive integer")?;
    let out_obj = positional
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("obbs.obj"));

    if obb_only {
        mask = PrimMask::obb_only();
    }

    Ok(CliArgs {
        mesh_path,
        target_n,
        out_obj,
        viewer_html,
        volume_threshold_frac,
        enable_mask: mask,
        cull_redundant,
        empty_space,
        refine_orient,
        quality_beta,
        shell_aware,
        proximity,
        weighted_cost,
        rebalance,
        reject_pancakes,
        strip_thin_obbs,
        feasibility,
        split_worst,
        cull_overlap,
        axis_align,
        world_axis_align,
        tangent_eps,
        metrics: metrics_flag,
        metrics_json,
    })
}

fn print_usage() {
    eprintln!(
        "usage: cpd <mesh.glb> <target_n> [out.obj]
       [--viewer <viewer.html>]
       [--volume-threshold <fraction-of-AABB-volume>]
       [--obb-only]
       [--no-cull]
       [--no-refine]   disable post-merge orientation refit (default on)
       [--quality <beta>]   Hausdorff-aware refit. Combined cost is
                            volume * (1 + beta * h/diag). Try 0.5–5.0.
                            Lets sphere compete; tightens low-N fits.
       [--shell]            shell-aware orientation. Pre-computes per-face
                            ambient-occlusion exposure; weights Q and PCA by
                            it so interior geometry doesn't bias axes. Best
                            for kitbashed / scanned assets.
       [--weighted-cost]    PQ ordering uses weighted volume (cost = ΔwV).
                            Trades surface fit for runtime/memory cost.
                            Helps near-convex / organic meshes (rocks,
                            terrain) by 10-20% Hausdorff. Hurts detail-
                            heavy meshes (vehicles) by similar amounts.
       [--rebalance]        Lloyd-style face migration after greedy. Keeps
                            N constant; tries to escape greedy local
                            minima. Default 5 passes.
           [--rebalance-passes <N>]   override iteration count
       [--reject-pancakes]  push merges that produce a 1mm-thick × Nm-wide
                            slab to the bottom of the PQ. Targets
                            architecture meshes with rooftops / wall
                            collisions; can hurt vehicles with long thin
                            panels at the same threshold.
       [--strip-thin]       paper appendix Fig 22 postprocess: after merge
                            + cull, delete any OBB whose smallest half-
                            extent ≤ 1e-4 × mesh_diag. Different from
                            --reject-pancakes (which fights the merge);
                            this lets the merge converge then strips slabs
                            after the fact. Paper's recipe for environment
                            scenes with planar-but-not-rectangular walls.
           [--strip-thin-threshold <f>]   override 1e-4
       [--feasibility <f>]  reject any popped merge whose result has local
                            Hausdorff > f × mesh_diag. Direct attack on the
                            slab-merge failure mode (cost function can't
                            see surface drift). Try 0.05 to 0.15. Cost:
                            ~24 BVH queries per realized merge.
       [--split-worst <f>]  post-merge: repeatedly find the live primitive
                            with highest local Hausdorff > f × diag and
                            split it along the longest PCA axis (median
                            split) into two new primitives. Each accepted
                            split adds 1 primitive. Targets the OBB-on-
                            non-rectangular-region limit.
           [--split-worst-max <int>]  cap on splits (default 32)
       [--axis-align]       lock all primitive orientations to the mesh's
                            dominant axes (eigendecomposition of global Q).
                            Eliminates the rotated-slab failure mode on
                            architecture; off-by-default since it removes
                            per-primitive orientation refinement that helps
                            organic / vehicle / rotated meshes.
       [--no-tangent-eps]   set Q's tangent-term coefficient to 0 (paper
                            §3.4 says decided-per-mesh). Removes the
                            rotated-OBB failure mode on large flat regions;
                            can hurt meshes that depend on the eigendecomp's
                            in-plane stabilisation (vehicles, organic).
       [--tangent-eps <e>]  override default 0.01 with custom ε.
       [--proximity]        spatial-proximity merges between disconnected
                            components. Adds candidate edges between nearby
                            components in the initial PQ, with cost ordering
                            so small fragments merge first. Replaces the
                            all-pairs failure mode for kitbashed assets.
           [--proximity-r <frac-of-diag>]   default 0.05
           [--proximity-k <int>]            default 2
           [--proximity-angle <deg>]        default 45
       [--metrics]     compute one-way Hausdorff/Chamfer (paper §4.4) + volume ratio
       [--metrics-json <path>]   also write metrics as JSON to <path>
       [--empty-space]   coarse heuristic — reliably flags large bridges
                         (stairwells, vents, slots) but also rejects most
                         OBBs that wrap curved geometry. Best on
                         hole-dominant meshes (mazes, towers, environments).
           [--empty-space-fraction <0..1>]   default 0.25
           [--empty-space-distance <frac-of-diag>]   default 0.01
       [--no-sphere | --no-cylinder | --no-capsule | --no-frustum | --no-prism]"
    );
}

fn main() -> Result<()> {
    let args = parse_args()?;

    let t0 = Instant::now();
    let mut m = mesh::load_glb(&args.mesh_path)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let diag = mesh::aabb_diag(&m.verts);
    eprintln!(
        "loaded {}: {} verts, {} tris, aabb diag {:.3} ({:.1} ms)",
        args.mesh_path.display(),
        m.verts.len(),
        m.tris.len(),
        diag,
        load_ms,
    );

    if m.tris.is_empty() {
        anyhow::bail!("mesh has no triangles");
    }

    let weld_eps = (diag.max(1.0)) * 1e-6;
    let collapsed = mesh::weld_vertices(&mut m, weld_eps);
    eprintln!(
        "welded {} duplicate verts (eps {:.2e}); now {} verts / {} tris",
        collapsed,
        weld_eps,
        m.verts.len(),
        m.tris.len(),
    );

    // AABB volume for the relative volume threshold knob.
    let (lo, hi) = mesh::aabb(&m.verts);
    let aabb_vol = ((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z)).max(1e-12);
    let abs_vol_threshold = if args.volume_threshold_frac.is_finite() {
        args.volume_threshold_frac * aabb_vol
    } else {
        f32::INFINITY
    };
    eprintln!(
        "aabb volume {:.4}, volume-threshold {} → abs {}",
        aabb_vol,
        args.volume_threshold_frac,
        abs_vol_threshold,
    );

    let t1 = Instant::now();
    let adj = mesh::build_adjacency(&m.tris);
    let adj_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let neighbor_total: usize = adj.neighbors.iter().map(|v| v.len()).sum();
    eprintln!(
        "adjacency: {:.1} ms, avg neighbors/tri = {:.2}",
        adj_ms,
        neighbor_total as f64 / m.tris.len() as f64,
    );

    let t2 = Instant::now();
    let empty_space = args.empty_space.map(|(frac, dist_frac)| {
        let abs_dist = dist_frac * diag.max(1.0);
        eprintln!(
            "empty-space check: max bridge fraction {:.2}, dist threshold {:.4} ({:.2}% of diag)",
            frac,
            abs_dist,
            dist_frac * 100.0,
        );
        (frac, abs_dist)
    });
    let result = decomp::run(
        &m,
        &adj,
        decomp::DecompOpts {
            target_n: args.target_n,
            volume_threshold: abs_vol_threshold,
            enabled: args.enable_mask,
            cull_redundant: args.cull_redundant,
            empty_space,
            refine_orient: args.refine_orient,
            quality_beta: args.quality_beta,
            shell_aware: args.shell_aware,
            proximity: args.proximity,
            weighted_cost: args.weighted_cost,
            rebalance: args.rebalance,
            reject_pancakes: args.reject_pancakes,
            strip_thin_obbs: args.strip_thin_obbs,
            feasibility: args.feasibility,
            split_worst: args.split_worst,
            cull_overlap: args.cull_overlap,
            axis_align: args.axis_align,
            world_axis_align: args.world_axis_align,
            tangent_eps: args.tangent_eps,
        },
    );
    let merge_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let alive: usize = result.primitives.iter().filter(|p| p.alive).count();
    let total_vol: f32 = result
        .primitives
        .iter()
        .filter(|p| p.alive)
        .map(|p| p.volume)
        .sum();
    let by_kind = count_by_kind(&result.primitives);
    eprintln!(
        "merge: {:.1} ms, {} merges, {} stale, {} empty-rejected, {} feasibility-rejected, all-pairs={}, culled={}, overlap-culled={}, splits={}, thin-stripped={}, {} primitives, total vol {:.3}",
        merge_ms,
        result.merges_done,
        result.merges_skipped_stale,
        result.merges_rejected_empty,
        result.merges_rejected_feasibility,
        result.all_pairs_used,
        result.redundant_culled,
        result.overlap_culled,
        result.splits_done,
        result.thin_stripped,
        alive,
        total_vol,
    );
    eprintln!("by kind: {}", by_kind);

    io_obj::write_obbs_obj(&args.out_obj, &result.primitives)?;

    if let Some(html_path) = &args.viewer_html {
        viewer::write_viewer(html_path, &m, &result.primitives)?;
    }

    if args.metrics {
        let t = Instant::now();
        let met = metrics::compute(&result.primitives, &m, 10_000);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        eprintln!("{} ({:.1} ms)", met.human(), ms);

        // Detailed dump of the Hausdorff-driving primitive.
        let wp = &result.primitives[met.worst_prim_idx].prim;
        eprintln!("  worst primitive details: {:#?}", wp);

        if let Some(p) = &args.metrics_json {
            std::fs::write(p, met.json())
                .with_context(|| format!("writing metrics to {}", p.display()))?;
            eprintln!("wrote metrics json to {}", p.display());
        }
    }

    Ok(())
}

fn count_by_kind(prims: &[decomp::Primitive]) -> String {
    use prim::PrimKind;
    let mut counts = [0usize; 6];
    for p in prims {
        if !p.alive {
            continue;
        }
        let i = match p.prim.kind() {
            PrimKind::Obb => 0,
            PrimKind::Sphere => 1,
            PrimKind::Cylinder => 2,
            PrimKind::Capsule => 3,
            PrimKind::Frustum => 4,
            PrimKind::Prism => 5,
        };
        counts[i] += 1;
    }
    format!(
        "obb={} sphere={} cyl={} cap={} frustum={} prism={}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]
    )
}
