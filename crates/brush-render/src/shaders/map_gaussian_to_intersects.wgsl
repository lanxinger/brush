#import helpers;

@group(0) @binding(0) var<storage, read> uniforms: helpers::RenderUniforms;
@group(0) @binding(1) var<storage, read> projected: array<helpers::ProjectedSplat>;

#ifdef PREPASS
    @group(0) @binding(2) var<storage, read_write> splat_intersect_counts: array<atomic<i32>>;
    @group(0) @binding(3) var<storage, read_write> tile_intersect_counts: array<atomic<i32>>;
#else
    @group(0) @binding(2) var<storage, read> splat_cum_hit_counts: array<i32>;
    @group(0) @binding(3) var<storage, read_write> tile_id_from_isect: array<i32>;
    @group(0) @binding(4) var<storage, read_write> compact_gid_from_isect: array<i32>;
#endif


@compute
@workgroup_size(256, 1, 1) // Relatively small workgroup, as work amount is quite variable, so rather not hold up a whole SM.
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let compact_gid = gid.x;

    if i32(compact_gid) >= uniforms.num_visible {
        return;
    }

    let projected = projected[compact_gid];
    let mean2d = vec2f(projected.xy_x, projected.xy_y);

    let opac = projected.color_a;
    let power_threshold = log(opac * 255.0f);

    // Compute the splat's bounding box in tile coordinates.
    let conic = vec3f(projected.conic_x, projected.conic_y, projected.conic_z);
    let cov2d = helpers::inverse(mat2x2f(conic.x, conic.y, conic.y, conic.z));
    let extent = helpers::compute_bbox_extent(cov2d, power_threshold);
    let tile_bbox = helpers::get_tile_bbox(mean2d, extent, uniforms.tile_bounds);
    let tile_bbox_min = tile_bbox.xy;
    let tile_bbox_max = tile_bbox.zw;

    var num_tiles_hit = 0;

    #ifdef PREPASS
        var base_isect_id = 0;
    #else
        var base_isect_id = splat_cum_hit_counts[compact_gid];
    #endif

    // Nb: It's really really important here the two dispatches
    // of this kernel arrive at the exact same num_tiles_hit count. Otherwise
    // we might not be writing some intersection data.
    // This is a bit scary given potential optimizations that might happen depending
    // on which version is being ran.
    let tile_bbox_width = tile_bbox_max.x - tile_bbox_min.x;
    let num_tiles_bbox = (tile_bbox_max.y - tile_bbox_min.y) * tile_bbox_width;
    for (var tile_idx = 0u; tile_idx < num_tiles_bbox; tile_idx++) {
        let tx = (tile_idx % tile_bbox_width) + tile_bbox_min.x;
        let ty = (tile_idx / tile_bbox_width) + tile_bbox_min.y;
        if helpers::will_primitive_contribute(vec2u(tx, ty), mean2d, conic, power_threshold) {
            let tile_id = tx + ty * uniforms.tile_bounds.x;

        #ifdef PREPASS
            // TODO: Want to bail here if the tile is saturated with gaussians, but not clear
            // how the prepass and final pass would agree.
            atomicAdd(&tile_intersect_counts[tile_id + 1u], 1);
        #else
            let isect_id = base_isect_id + num_tiles_hit;
            // Nb: isect_id MIGHT be out of bounds here for degenerate cases.
            // These kernels should be launched with bounds checking, so that these
            // writes are ignored. This will skip these intersections.
            tile_id_from_isect[isect_id] = i32(tile_id);
            compact_gid_from_isect[isect_id] = i32(compact_gid);
        #endif

            num_tiles_hit += 1;
        }
    }

    #ifdef PREPASS
        splat_intersect_counts[compact_gid + 1u] = num_tiles_hit;
    #endif
}
