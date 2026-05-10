mod bvh;
mod decomp;
mod dsu;
mod io_obj;
mod mesh;
mod metrics;
mod prim;
mod viewer;

use anyhow::{Context, Result};
use nalgebra::Point3;
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
    subdivide_bad_faces: Option<(f32, usize)>,
    strip_thin_obbs: Option<f32>,
    feasibility: Option<f32>,
    outside_space: Option<f32>,
    outside_fit_beta: f32,
    split_worst: Option<(f32, usize)>,
    repair_bad_slabs: Option<(f32, usize)>,
    post_merge_budget: Option<(usize, f32)>,
    shrink_high_error: Option<f32>,
    split_error_region: bool,
    split_debug_json: Option<PathBuf>,
    split_debug_primitive: Option<u32>,
    cull_overlap: Option<f32>,
    collision_simplify: Option<f32>,
    collision_target_scale: Option<f32>,
    collision_ignore_detail: Option<f32>,
    collision_support_planes: Option<f32>,
    axis_align: bool,
    world_axis_align: bool,
    refine_search: Option<(f32, usize)>,
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
    let mut subdivide_bad_faces: Option<(f32, usize)> = None;
    let mut strip_thin_obbs: Option<f32> = None;
    let mut feasibility: Option<f32> = None;
    let mut outside_space: Option<f32> = None;
    let mut outside_fit_beta: f32 = 0.0;
    let mut split_worst: Option<(f32, usize)> = None;
    let mut repair_bad_slabs: Option<(f32, usize)> = None;
    let mut post_merge_budget: Option<usize> = None;
    let mut post_merge_hausdorff: Option<f32> = None;
    let mut shrink_high_error: Option<f32> = None;
    let mut split_error_region = false;
    let mut split_debug_json: Option<PathBuf> = None;
    let mut split_debug_primitive: Option<u32> = None;
    let mut cull_overlap: Option<f32> = None;
    let mut collision_simplify: Option<f32> = None;
    let mut collision_target_scale: Option<f32> = None;
    let mut collision_ignore_detail: Option<f32> = None;
    let mut collision_support_planes: Option<f32> = None;
    let mut axis_align = false;
    let mut world_axis_align = false;
    let mut refine_search: Option<(f32, usize)> = None;
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
                let v: String = args
                    .first()
                    .cloned()
                    .context("--volume-threshold needs f32")?;
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
                let v = args
                    .first()
                    .cloned()
                    .context("--quality needs a beta value")?;
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
            "--subdivide-bad-faces" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--subdivide-bad-faces needs a threshold-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev_depth = subdivide_bad_faces.map(|(_, d)| d).unwrap_or(1);
                subdivide_bad_faces = Some((f, prev_depth));
            }
            "--subdivide-bad-faces-max-depth" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--subdivide-bad-faces-max-depth needs an integer")?;
                args.remove(0);
                let depth: usize = v.parse().context("not an integer")?;
                let prev_frac = subdivide_bad_faces.map(|(f, _)| f).unwrap_or(0.05);
                subdivide_bad_faces = Some((prev_frac, depth));
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
            "--outside-space" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--outside-space needs a max-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                outside_space = Some(f);
            }
            "--outside-fit" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--outside-fit needs a beta value")?;
                args.remove(0);
                outside_fit_beta = v.parse().context("not a float")?;
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
            "--repair-bad-slabs" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--repair-bad-slabs needs a threshold-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev_max = repair_bad_slabs.map(|(_, m)| m).unwrap_or(32);
                repair_bad_slabs = Some((f, prev_max));
            }
            "--repair-bad-slabs-max" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--repair-bad-slabs-max needs an integer")?;
                args.remove(0);
                let m: usize = v.parse().context("not an integer")?;
                let prev_frac = repair_bad_slabs.map(|(f, _)| f).unwrap_or(0.03);
                repair_bad_slabs = Some((prev_frac, m));
            }
            "--post-merge-budget" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--post-merge-budget needs a primitive count")?;
                args.remove(0);
                post_merge_budget = Some(v.parse().context("not an integer")?);
            }
            "--post-merge-hausdorff" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--post-merge-hausdorff needs a threshold-frac-of-diag value")?;
                args.remove(0);
                post_merge_hausdorff = Some(v.parse().context("not a float")?);
            }
            "--shrink-high-error" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--shrink-high-error needs a threshold-frac-of-diag value")?;
                args.remove(0);
                shrink_high_error = Some(v.parse().context("not a float")?);
            }
            "--split-error-region" => {
                args.remove(0);
                split_error_region = true;
                split_worst = Some(split_worst.unwrap_or((0.05, 32)));
            }
            "--debug-splits" => {
                args.remove(0);
                let p = args
                    .first()
                    .cloned()
                    .context("--debug-splits needs an output .jsonl path")?;
                args.remove(0);
                split_debug_json = Some(PathBuf::from(p));
                split_worst = Some(split_worst.unwrap_or((0.05, 32)));
            }
            "--debug-primitive" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--debug-primitive needs a primitive id")?;
                args.remove(0);
                split_debug_primitive = Some(v.parse().context("not an integer")?);
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
            "--collision-simplify" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--collision-simplify needs a tolerance-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                collision_simplify = Some(f);
            }
            "--collision-target-scale" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--collision-target-scale needs a positive scale")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                collision_target_scale = Some(f);
            }
            "--collision-ignore-detail" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--collision-ignore-detail needs a feature-scale fraction")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                collision_ignore_detail = Some(f);
            }
            "--collision-support-planes" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--collision-support-planes needs a plane-distance fraction")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                collision_support_planes = Some(f);
            }
            "--axis-align" => {
                args.remove(0);
                axis_align = true;
            }
            "--world-axis" => {
                args.remove(0);
                world_axis_align = true;
            }
            "--refine-search" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--refine-search needs a threshold-frac-of-diag value")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = refine_search.map(|(_, m)| m).unwrap_or(20);
                refine_search = Some((f, prev));
            }
            "--refine-search-iters" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--refine-search-iters needs an integer")?;
                args.remove(0);
                let m: usize = v.parse().context("not an integer")?;
                let prev = refine_search.map(|(f, _)| f).unwrap_or(0.05);
                refine_search = Some((prev, m));
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
                let v = args
                    .first()
                    .cloned()
                    .context("--rebalance-passes needs usize")?;
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
                let v = args
                    .first()
                    .cloned()
                    .context("--proximity-angle needs degrees")?;
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
                let p = args
                    .first()
                    .cloned()
                    .context("--metrics-json needs a path")?;
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
                let v = args
                    .first()
                    .cloned()
                    .context("--empty-space-fraction needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = empty_space.unwrap_or((0.25, 0.01));
                empty_space = Some((f, prev.1));
            }
            "--empty-space-distance" => {
                args.remove(0);
                let v = args
                    .first()
                    .cloned()
                    .context("--empty-space-distance needs f32")?;
                args.remove(0);
                let f: f32 = v.parse().context("not a float")?;
                let prev = empty_space.unwrap_or((0.25, 0.01));
                empty_space = Some((prev.0, f));
            }
            "--no-sphere" => {
                args.remove(0);
                mask.sphere = false;
            }
            "--no-cylinder" => {
                args.remove(0);
                mask.cylinder = false;
            }
            "--no-capsule" => {
                args.remove(0);
                mask.capsule = false;
            }
            "--no-frustum" => {
                args.remove(0);
                mask.frustum = false;
            }
            "--no-prism" => {
                args.remove(0);
                mask.prism = false;
            }
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
    if split_debug_primitive.is_some() && split_debug_json.is_none() {
        anyhow::bail!("--debug-primitive requires --debug-splits <path>");
    }
    if let Some((frac, depth)) = subdivide_bad_faces {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--subdivide-bad-faces must be a positive finite fraction");
        }
        if depth == 0 {
            anyhow::bail!("--subdivide-bad-faces-max-depth must be at least 1");
        }
    }
    if let Some(frac) = collision_simplify {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--collision-simplify must be a positive finite fraction");
        }
    }
    if let Some(frac) = outside_space {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--outside-space must be a positive finite fraction");
        }
    }
    if let Some((frac, max_repairs)) = repair_bad_slabs {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--repair-bad-slabs must be a positive finite fraction");
        }
        if max_repairs == 0 {
            anyhow::bail!("--repair-bad-slabs-max must be at least 1");
        }
    }
    if let Some(target) = post_merge_budget {
        if target == 0 {
            anyhow::bail!("--post-merge-budget must be at least 1");
        }
    }
    if let Some(frac) = post_merge_hausdorff {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--post-merge-hausdorff must be a positive finite fraction");
        }
        if post_merge_budget.is_none() {
            anyhow::bail!("--post-merge-hausdorff requires --post-merge-budget <n>");
        }
    }
    if let Some(frac) = shrink_high_error {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--shrink-high-error must be a positive finite fraction");
        }
    }
    if outside_fit_beta != 0.0 {
        if outside_fit_beta <= 0.0 || !outside_fit_beta.is_finite() {
            anyhow::bail!("--outside-fit must be a positive finite beta");
        }
        if outside_space.is_none() {
            anyhow::bail!("--outside-fit requires --outside-space <f>");
        }
    }
    if let Some(scale) = collision_target_scale {
        if scale <= 0.0 || !scale.is_finite() {
            anyhow::bail!("--collision-target-scale must be a positive finite scale");
        }
    }
    if let Some(frac) = collision_ignore_detail {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--collision-ignore-detail must be a positive finite fraction");
        }
    }
    if let Some(frac) = collision_support_planes {
        if frac <= 0.0 || !frac.is_finite() {
            anyhow::bail!("--collision-support-planes must be a positive finite fraction");
        }
    }
    let mesh_path = PathBuf::from(&positional[0]);
    let target_n: usize = positional[1]
        .parse()
        .context("target_n must be a positive integer")?;
    let out_obj = positional
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("obbs.obj"));

    if obb_only {
        mask = PrimMask::obb_only();
    }
    let post_merge_budget = post_merge_budget.map(|target| {
        let threshold = post_merge_hausdorff.unwrap_or_else(|| {
            repair_bad_slabs
                .map(|(f, _)| f)
                .or_else(|| split_worst.map(|(f, _)| f))
                .unwrap_or(0.03)
        });
        (target, threshold)
    });

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
        subdivide_bad_faces,
        strip_thin_obbs,
        feasibility,
        outside_space,
        outside_fit_beta,
        split_worst,
        repair_bad_slabs,
        post_merge_budget,
        shrink_high_error,
        split_error_region,
        split_debug_json,
        split_debug_primitive,
        cull_overlap,
        collision_simplify,
        collision_target_scale,
        collision_ignore_detail,
        collision_support_planes,
        axis_align,
        world_axis_align,
        refine_search,
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
       [--subdivide-bad-faces <f>]  pre-decomp: recursively split a single
                            triangle into 3 triangles when its initial
                            one-face primitive has local Hausdorff > f × diag.
                            Targets oversized planar triangles; default
                            max depth 1.
           [--subdivide-bad-faces-max-depth <int>]  override depth
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
       [--outside-space <f>]  reject any popped merge whose sampled surface
                            is farther than f × diag on the outward signed
                            side of the nearest input face; also biases
                            split/refine toward reducing outward protrusion.
                            Experimental; assumes mostly outward winding.
       [--outside-fit <beta>] during merge-time refit, score candidate
                            orientations/types with an outside-space penalty.
                            Requires --outside-space. Try 1 to 3.
       [--split-worst <f>]  post-merge: repeatedly find the live primitive
                            with highest local Hausdorff > f × diag and
                            split it along the longest PCA axis (median
                            split) into two new primitives. Each accepted
                            split adds 1 primitive. Targets the OBB-on-
                            non-rectangular-region limit.
           [--split-worst-max <int>]  cap on splits (default 32)
       [--repair-bad-slabs <f>] final collision repair: only medium
                            slab-like OBBs with sampled protrusion > f × diag;
                            tries corner/footprint splits, plus OBB-only
                            sub-triangle trims for one-face slabs. Try
                            0.025 to 0.035.
           [--repair-bad-slabs-max <int>]  cap on repairs (default 32)
       [--post-merge-budget <n>] final gated compression pass to n primitives
           [--post-merge-hausdorff <f>] max accepted merge Hausdorff as f × diag
       [--shrink-high-error <f>] final OBB shrink pass for primitives whose
                            dense local Hausdorff remains above f × diag.
           [--split-error-region]  opt-in repair mode for --split-worst:
                            also try candidate splits derived from the
                            primitive's farthest sampled error point.
                            Implies --split-worst 0.05 if omitted.
           [--debug-splits <path.jsonl>]  write per-candidate split scores.
                            Implies --split-worst 0.05 if omitted.
           [--debug-primitive <id>]  restrict --debug-splits to one viewer/
                            metrics primitive id.
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
       [--collision-simplify <f>]  collision-oriented simplification: prefer
                         cleaner primitive types during fitting/merging, bias
                         thin detail to merge into nearby larger shapes, then
                         drop supported thin/small leftovers within f × diag.
           [--collision-target-scale <f>]  when collision simplification is
                         enabled, merge to target_n × f before cleanup. Values
                         below 1 trade detail fidelity for cleaner colliders.
       [--collision-ignore-detail <f>]  collision fit vertices ignore faces
                         whose sqrt(area) is below f × diag when a merged
                         primitive also has larger support faces. Try 0.001-0.005.
       [--collision-support-planes <f>]  detect broad support planes and project
                         nearby detail faces onto them during collision fitting.
                         The value is plane distance as a fraction of diag.
                         Try 0.003-0.005.
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

    if let Some((threshold_frac, max_depth)) = args.subdivide_bad_faces {
        let t = Instant::now();
        let stats = subdivide_bad_faces(
            &mut m,
            threshold_frac,
            max_depth,
            args.enable_mask,
            args.tangent_eps,
        );
        eprintln!(
            "subdivide-bad-faces: split {} faces, +{} tris, max depth {}, now {} verts / {} tris ({:.1} ms)",
            stats.split_faces,
            stats.added_tris,
            stats.max_depth_reached,
            m.verts.len(),
            m.tris.len(),
            t.elapsed().as_secs_f64() * 1000.0,
        );
    }

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
        aabb_vol, args.volume_threshold_frac, abs_vol_threshold,
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
            outside_space: args.outside_space,
            outside_fit_beta: args.outside_fit_beta,
            split_worst: args.split_worst,
            repair_bad_slabs: args.repair_bad_slabs,
            post_merge_budget: args.post_merge_budget,
            shrink_high_error: args.shrink_high_error,
            split_error_region: args.split_error_region,
            debug_splits: args
                .split_debug_json
                .as_ref()
                .map(|_| decomp::SplitDebugOpts {
                    primitive: args.split_debug_primitive,
                }),
            cull_overlap: args.cull_overlap,
            collision_simplify: args.collision_simplify,
            collision_target_scale: args.collision_target_scale,
            collision_ignore_detail: args.collision_ignore_detail,
            collision_support_planes: args.collision_support_planes,
            axis_align: args.axis_align,
            world_axis_align: args.world_axis_align,
            refine_search: args.refine_search,
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
        "merge: {:.1} ms, {} merges, {} stale, {} empty-rejected, {} feasibility-rejected, {} outside-rejected, all-pairs={}, culled={}, overlap-culled={}, post-budget-merges={}, collision-simplified={}, splits={}, slab-repairs={}, high-error-shrunk={}, thin-stripped={}, {} primitives, total vol {:.3}",
        merge_ms,
        result.merges_done,
        result.merges_skipped_stale,
        result.merges_rejected_empty,
        result.merges_rejected_feasibility,
        result.merges_rejected_outside,
        result.all_pairs_used,
        result.redundant_culled,
        result.overlap_culled,
        result.post_budget_merges,
        result.collision_simplified,
        result.splits_done,
        result.slab_repairs_done,
        result.high_error_shrunk,
        result.thin_stripped,
        alive,
        total_vol,
    );
    eprintln!("by kind: {}", by_kind);

    if let Some(path) = &args.split_debug_json {
        let mut lines = String::new();
        for row in &result.split_debug {
            lines.push_str(&row.json_line());
            lines.push('\n');
        }
        std::fs::write(path, lines)
            .with_context(|| format!("writing split debug jsonl to {}", path.display()))?;
        eprintln!(
            "wrote {} split debug rows to {}",
            result.split_debug.len(),
            path.display()
        );
    }

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

struct SubdivideStats {
    split_faces: usize,
    added_tris: usize,
    max_depth_reached: usize,
}

fn subdivide_bad_faces(
    mesh: &mut mesh::Mesh,
    threshold_frac: f32,
    max_depth: usize,
    enabled: PrimMask,
    tangent_eps: f32,
) -> SubdivideStats {
    let start_tris = mesh.tris.len();
    let reference = mesh::Mesh {
        verts: mesh.verts.clone(),
        tris: mesh.tris.clone(),
    };
    let bvh = bvh::Bvh::build(&reference.verts, &reference.tris);
    let threshold = threshold_frac * mesh::aabb_diag(&reference.verts).max(1e-6);
    let mut depths = vec![0usize; mesh.tris.len()];
    let mut split_faces = 0usize;

    for _ in 0..max_depth {
        let mut changed = false;
        let mut next_tris: Vec<[u32; 3]> = Vec::with_capacity(mesh.tris.len());
        let mut next_depths: Vec<usize> = Vec::with_capacity(depths.len());

        for (ti, tri) in mesh.tris.iter().copied().enumerate() {
            let depth = depths[ti];
            if depth >= max_depth {
                next_tris.push(tri);
                next_depths.push(depth);
                continue;
            }

            let p0 = mesh.verts[tri[0] as usize];
            let p1 = mesh.verts[tri[1] as usize];
            let p2 = mesh.verts[tri[2] as usize];
            let area2 = (p1 - p0).cross(&(p2 - p0)).norm();
            if area2 < 1e-12 {
                next_tris.push(tri);
                next_depths.push(depth);
                continue;
            }

            let h = decomp::single_triangle_dense_hausdorff(
                p0,
                p1,
                p2,
                &bvh,
                &reference,
                enabled,
                tangent_eps,
                None,
            );
            if h <= threshold {
                next_tris.push(tri);
                next_depths.push(depth);
                continue;
            }

            let centroid = Point3::from((p0.coords + p1.coords + p2.coords) / 3.0);
            let ci = mesh.verts.len() as u32;
            mesh.verts.push(centroid);
            let next_depth = depth + 1;
            next_tris.push([tri[0], tri[1], ci]);
            next_depths.push(next_depth);
            next_tris.push([tri[1], tri[2], ci]);
            next_depths.push(next_depth);
            next_tris.push([tri[2], tri[0], ci]);
            next_depths.push(next_depth);
            split_faces += 1;
            changed = true;
        }

        mesh.tris = next_tris;
        depths = next_depths;
        if !changed {
            break;
        }
    }

    SubdivideStats {
        split_faces,
        added_tris: mesh.tris.len().saturating_sub(start_tris),
        max_depth_reached: depths.iter().copied().max().unwrap_or(0),
    }
}
