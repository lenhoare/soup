"""GLSL sources. Colour work happens in OKLCH -> linear sRGB in the scene pass; the
composite pass does bloom add, chromatic aberration, vignette, filmic tonemap and
sRGB encoding."""

FULLSCREEN_VS = """
#version 330
out vec2 v_uv;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

COMMON = """
vec3 oklch(float L, float C, float h) {
    float a = C * cos(h), b = C * sin(h);
    float l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    float m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    float s_ = L - 0.0894841775 * a - 1.2914855480 * b;
    float l = l_ * l_ * l_, m = m_ * m_ * m_, s = s_ * s_ * s_;
    return max(vec3(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s), 0.0);
}
float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
"""

SCENE_FS = """
#version 330
uniform isampler2D u_ids;    // r: slot gid (-1 empty), g: lineage root
uniform sampler2D u_cell;    // energy, state0, state1, heat
uniform sampler2D u_env;     // light, fx_gold (HGT), fx_blue (heat damage), fx_white (spore)
uniform sampler2D u_morph;   // 4 morphogens
uniform sampler2D u_slots;   // per slot: hue, chroma, depth_norm, dist_norm  (256-wide)
uniform vec2 u_view;         // viewport size, px
uniform vec2 u_cam;          // camera centre, cells
uniform float u_zoom;        // px per cell
uniform ivec2 u_grid;        // W, H
uniform int u_mode;          // 1..6
uniform float u_time;
uniform int u_sel;           // selected slot or -1
uniform vec2 u_cursor;       // hovered cell (cells), x < 0 if none
in vec2 v_uv;
out vec4 frag;
""" + COMMON + """
ivec2 wrapc(ivec2 c) { return ivec2(mod(vec2(c), vec2(u_grid))); }
ivec2 idsAt(ivec2 c) { return texelFetch(u_ids, wrapc(c), 0).rg; }
vec4 slotAt(int g) { return texelFetch(u_slots, ivec2(g % 256, g / 256), 0); }

vec3 lineageColour(ivec2 c, out float alive) {
    int g = idsAt(c).r;
    alive = g >= 0 ? 1.0 : 0.0;
    if (g < 0) return vec3(0.0);
    vec4 cell = texelFetch(u_cell, wrapc(c), 0);
    vec4 sl = slotAt(g);
    return oklch(0.24 + 0.36 * cell.r + 0.06 * cell.g, sl.g, sl.r + 0.35 * cell.b);
}

vec3 ramp(float t, float hue) {          // single-hue sequential ramp, dark -> light
    t = clamp(t, 0.0, 1.0);
    return oklch(0.22 + 0.62 * t, 0.04 + 0.11 * t, hue);
}

void main() {
    vec2 px = gl_FragCoord.xy;
    vec2 w = vec2(px.x - 0.5 * u_view.x, 0.5 * u_view.y - px.y) / u_zoom + u_cam;
    ivec2 c = wrapc(ivec2(floor(w)));
    vec2 f = fract(w);
    ivec2 id = idsAt(c);
    int g = id.r;
    bool alive = g >= 0;
    vec4 cell = texelFetch(u_cell, c, 0);
    vec4 env = texelFetch(u_env, c, 0);
    float energy = cell.r, heat = cell.a, light = env.r;

    // ---- empty space: near-black ink with ember speckle ~ heat
    vec3 col = oklch(0.13, 0.012, 4.9);
    float tick = floor(u_time * 3.0);
    float ember = step(hash12(vec2(c) + tick * 17.31), 0.07 * heat * heat);
    // zoomed in, an ember is a small glowing dot inside its cell, not a whole square
    ember *= mix(1.0, smoothstep(0.22, 0.1, length(f - 0.5)), smoothstep(3.0, 8.0, u_zoom));
    col += ember * oklch(0.62, 0.16, 0.85) * (0.6 + 0.8 * heat) * (0.6 + 0.4 * sin(u_time * 9.0 + float(c.x)));

    vec4 sl = alive ? slotAt(g) : vec4(0.0);
    bool sel = u_sel >= 0;
    bool mine = alive && g == u_sel;

    if (u_mode == 1) {                    // lineage
        float a;
        // hue from genome embedding, interior texture: state1 -> hue, state0 -> lightness
        if (alive) col = lineageColour(c, a);
        // zoomed out: blend with the 3x3 neighbourhood so lattice-like bodies read as
        // tissue (density -> brightness); zoomed in: exact cells
        float far = 1.0 - smoothstep(5.0, 10.0, u_zoom);
        if (far > 0.0) {
            vec3 acc = vec3(0.0);
            float n = 0.0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    acc += lineageColour(c + ivec2(dx, dy), a);
                    n += a;
                }
            if (n > 0.0) {
                vec3 tissue = acc / n * (0.45 + 0.55 * n / 9.0);
                col = mix(col, tissue, far * (alive ? 0.7 : 0.85));
            }
        }
    } else if (u_mode == 2) {             // energy
        if (alive) col = ramp(energy, 1.55);
    } else if (u_mode == 3) {             // morphogens (also diffuse into empty space)
        vec4 m = texelFetch(u_morph, c, 0);
        vec4 mm = m / (1.0 + m);                         // compress: morph can reach 2
        col = 0.32 * vec3(mm.r, mm.g, mm.b) + 0.12 * mm.a + (alive ? 0.02 : 0.0);
    } else if (u_mode == 4) {             // genome distance from selected
        if (alive) col = sel ? ramp(1.0 - sl.a, 3.5) : oklch(0.35, 0.0, 0.0);
    } else if (u_mode == 5) {             // heat / light fields
        col = heat * oklch(0.55, 0.17, 0.6) * 0.6 + light * oklch(0.6, 0.12, 2.3) * 0.4;
        if (alive) col += 0.04;
    } else if (u_mode == 6) {             // lineage depth
        if (alive) col = ramp(sl.b, 5.2);
    }

    // ---- membranes
    if (alive && (u_mode == 1 || u_mode == 2 || u_mode == 6)) {
        // three kinds of edge: other lineage (bright membrane), sibling organism of
        // the same lineage (faint), empty space (soft surface rim)
        float ew = clamp(1.25 / u_zoom, 0.0, 0.2);
        float lin = 0.0, sib = 0.0, surf = 0.0, anyLin = 0.0;
        ivec2 d[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
        float near[4] = float[4](step(f.x, ew), step(1.0 - ew, f.x), step(f.y, ew), step(1.0 - ew, f.y));
        for (int i = 0; i < 4; i++) {
            ivec2 n = idsAt(c + d[i]);
            float isLin = (n.r >= 0 && n.g != id.g) ? 1.0 : 0.0;
            float isSib = (n.r >= 0 && n.g == id.g && n.r != g) ? 1.0 : 0.0;
            float isEmpty = n.r < 0 ? 1.0 : 0.0;
            anyLin = max(anyLin, isLin);
            lin = max(lin, near[i] * isLin);
            sib = max(sib, near[i] * isSib);
            surf = max(surf, near[i] * isEmpty);
        }
        float close = smoothstep(6.0, 12.0, u_zoom);         // sub-cell lines when zoomed in
        vec3 rim = oklch(0.88, sl.g * 0.8, sl.r);
        col = mix(col, rim * 1.6, close * lin);
        col = mix(col, rim, close * sib * 0.35);
        col = mix(col, col * 1.5 + 0.02, close * surf * 0.6);
        if (u_mode == 1)                                     // zoomed out: lineage borders only
            col = mix(col, rim, (1.0 - close) * anyLin * 0.4);
    }

    // ---- selection
    if (sel && u_mode != 4 && alive && !mine) col *= 0.45;
    if (mine) col *= 1.15;

    // ---- event flashes (HDR, picked up by bloom)
    col += env.g * oklch(0.9, 0.12, 1.4) * 3.0;     // HGT: gold-white
    col += env.b * oklch(0.75, 0.14, 4.3) * 2.2;    // heat damage: cold blue
    col += env.a * vec3(0.7);                       // spore pop

    // ---- crosshair on hovered cell
    if (u_cursor.x >= 0.0 && u_zoom >= 4.0) {
        ivec2 hc = wrapc(ivec2(floor(u_cursor)));
        float ew = 1.0 / u_zoom;
        if (c == hc && (f.x < ew || f.x > 1.0 - ew || f.y < ew || f.y > 1.0 - ew)) col = vec3(0.9);
    }
    frag = vec4(col, 1.0);
}
"""

BRIGHT_FS = """
#version 330
uniform sampler2D u_src;
uniform float u_threshold;
in vec2 v_uv;
out vec4 frag;
void main() {
    vec3 c = texture(u_src, v_uv).rgb;
    float l = max(max(c.r, c.g), c.b);
    frag = vec4(c * max(l - u_threshold, 0.0) / max(l, 1e-4), 1.0);
}
"""

BLUR_FS = """
#version 330
uniform sampler2D u_src;
uniform vec2 u_dir;     // texel step
in vec2 v_uv;
out vec4 frag;
void main() {
    float w[5] = float[5](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 acc = texture(u_src, v_uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        acc += texture(u_src, v_uv + u_dir * float(i) * 1.5).rgb * w[i];
        acc += texture(u_src, v_uv - u_dir * float(i) * 1.5).rgb * w[i];
    }
    frag = vec4(acc, 1.0);
}
"""

COMPOSITE_FS = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_k;
uniform float u_ca;          // chromatic aberration strength
uniform float u_vignette;
uniform float u_exposure;
in vec2 v_uv;
out vec4 frag;
vec3 aces(vec3 x) {          // Narkowicz fit
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 srgb(vec3 c) {
    return mix(12.92 * c, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}
void main() {
    vec2 d = v_uv - 0.5;
    vec2 off = d * u_ca;
    vec3 c = vec3(texture(u_scene, v_uv + off).r, texture(u_scene, v_uv).g,
                  texture(u_scene, v_uv - off).b);
    c += u_bloom_k * texture(u_bloom, v_uv).rgb;
    c *= u_exposure * (1.0 - u_vignette * dot(d, d) * 2.0);
    frag = vec4(srgb(aces(c)), 1.0);
}
"""

OVERLAY_FS = """
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 frag;
void main() { frag = texture(u_tex, vec2(v_uv.x, 1.0 - v_uv.y)); }
"""
