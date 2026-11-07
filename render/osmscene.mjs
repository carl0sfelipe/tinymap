// Cidade 3D a partir do OpenStreetMap, para onde o Google 3D Tiles não tem malha.
//
// Prédios extrudados pela altura (height / building:levels / tipo), ruas, água e áreas verdes,
// com cores chapadas e uma luz de sol. Fica "maquete", não foto, e é de propósito: o modelo de
// estilo redesenha tudo em pixel art, e uma entrada limpa é mais fácil de redesenhar do que uma
// malha derretida. A geometria fica no MESMO referencial (ECEF) do Google 3D Tiles, então a câmera,
// o planejamento, a máscara de água e a projeção do mapa continuam idênticos.
//
// Usado no navegador (main.js, view.source = "osm") e no Node (teste).
import {
  BufferGeometry, Float32BufferAttribute, Group, Mesh, MeshLambertMaterial, MeshBasicMaterial, Color, ShapeUtils,
  Vector2, Vector3, Matrix4, MathUtils, DirectionalLight, AmbientLight,
} from "three";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";
import { WGS84_ELLIPSOID } from "3d-tiles-renderer";

const D2R = MathUtils.DEG2RAD;
const ROOFS = ["#b5603f", "#9a9a96", "#c9c3b5", "#7d8a91", "#a34a33", "#d8d2c4", "#6f7f6a", "#8c6e5a"];
const WALLS = ["#efe9dc", "#e4ddd0", "#f4efe6", "#d9d4cb", "#ece2cf"];
const SAT = { water: "#3f7fc4", road: "#8d8d8a", major: "#77777a" }; // cores chapadas sobre o satélite
const FLAT = new Set(Object.values({ water: "#3f7fc4", road: "#8d8d8a", major: "#77777a" }));
const COL = {
  ground: "#cfcab9", road: "#6c6c6c", major: "#585858", water: "#4f86b8", green: "#7fa760",
  forest: "#557f43", sand: "#d9c68f", rail: "#6b5e55",
};

function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}

export const GROUND = COL.ground;

// Satélite (osm/sat.json + pedaços de 4096 px): lat/lon -> pixel da mosaica Web Mercator.
function satProj(sat) {
  const n = 2 ** sat.meta.z, ox = sat.meta.x0 * 256, oy = sat.meta.y0 * 256;
  const toPx = (lat, lon) => {
    const x = ((lon + 180) / 360) * n * 256 - ox;
    const y = ((1 - Math.asinh(Math.tan(lat * D2R)) / Math.PI) / 2) * n * 256 - oy;
    return [x, y];
  };
  const toLL = (x, y) => {
    const X = (x + ox) / (n * 256), Y = (y + oy) / (n * 256);
    return [Math.atan(Math.sinh(Math.PI * (1 - 2 * Y))) / D2R, X * 360 - 180];
  };
  const grid = new Map(sat.meta.chunks.map((c, k) => [`${Math.floor(c.py / 4096)},${Math.floor(c.px / 4096)}`, k]));
  const chunkAt = (x, y) => grid.get(`${Math.floor(y / 4096)},${Math.floor(x / 4096)}`) ?? -1;
  return { toPx, toLL, chunkAt };
}

export function buildOsmScene(view, data, sat = null) {
  const ground = view.target_height_m;
  const enu = new Matrix4();
  WGS84_ELLIPSOID.getEastNorthUpFrame(view.lat * D2R, view.lon * D2R, ground, enu);
  const inv = enu.clone().invert();
  const tmp = new Vector3();
  // lat/lon na altura do chão -> coordenadas locais ENU (com a curvatura: z cai ~4 m a 7 km)
  const local = ([lat, lon], dh = 0) => {
    WGS84_ELLIPSOID.getCartographicToPosition(lat * D2R, lon * D2R, ground + dh, tmp);
    tmp.applyMatrix4(inv);
    return [tmp.x, tmp.y, tmp.z];
  };

  const buckets = new Map(); // cor -> [geometria]
  const sp = sat ? satProj(sat) : null;
  const satRoofs = sat ? sat.meta.chunks.map(() => ({ pos: [], uv: [] })) : null;
  const push = (color, geom) => {
    if (!buckets.has(color)) buckets.set(color, []);
    buckets.get(color).push(geom);
  };

  // polígono plano (água, verde, chão) numa altura relativa dh
  function flat(rings, holes, dh, color) {
    for (const ring of rings) {
      const pts = ring.map((p) => local(p, dh));
      const hs = (holes ?? []).map((h) => h.map((p) => local(p, dh)));
      const tris = ShapeUtils.triangulateShape(pts.map(([x, y]) => new Vector2(x, y)), hs.map((h) => h.map(([x, y]) => new Vector2(x, y))));
      const all = pts.concat(...hs);
      const pos = [];
      for (const t of tris) for (const i of t) pos.push(...all[i]);
      if (!pos.length) continue;
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(pos, 3));
      g.computeVertexNormals();
      push(color, g);
    }
  }

  // faixa de rua como quads ao longo da linha
  function strip(line, width, dh, color) {
    const pts = line.map((p) => local(p, dh));
    const pos = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const [ax, ay, az] = pts[i], [bx, by, bz] = pts[i + 1];
      const dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1;
      const nx = (-dy / L) * width / 2, ny = (dx / L) * width / 2;
      pos.push(ax + nx, ay + ny, az, ax - nx, ay - ny, az, bx + nx, by + ny, bz,
        bx + nx, by + ny, bz, ax - nx, ay - ny, az, bx - nx, by - ny, bz);
    }
    if (!pos.length) return;
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(pos, 3));
    g.computeVertexNormals();
    push(color, g);
  }

  function building(b) {
    const k = hash(b.id ?? JSON.stringify(b.rings[0][0]));
    const roof = b.roof_colour || ROOFS[k % ROOFS.length];
    const wall = b.colour || WALLS[(k >> 4) % WALLS.length];
    const h = Math.max(2.5, b.h), base = Math.max(0, b.min_h ?? 0);
    for (const ring of b.rings) {
      if (ring.length < 4) continue;
      // anel em sentido anti-horário para as normais das paredes apontarem para fora
      let pts = ring.slice(0, -1).map((p) => local(p, 0));
      let area = 0;
      for (let i = 0; i < pts.length; i++) {
        const [x1, y1] = pts[i], [x2, y2] = pts[(i + 1) % pts.length];
        area += x1 * y2 - x2 * y1;
      }
      if (area < 0) pts = pts.reverse();
      const wallPos = [];
      for (let i = 0; i < pts.length; i++) {
        const [x1, y1, z1] = pts[i], [x2, y2, z2] = pts[(i + 1) % pts.length];
        wallPos.push(x1, y1, z1 + base, x2, y2, z2 + base, x2, y2, z2 + h,
          x1, y1, z1 + base, x2, y2, z2 + h, x1, y1, z1 + h);
      }
      const gw = new BufferGeometry();
      gw.setAttribute("position", new Float32BufferAttribute(wallPos, 3));
      gw.computeVertexNormals();
      push(wall, gw);
      const tris = ShapeUtils.triangulateShape(pts.map(([x, y]) => new Vector2(x, y)), []);
      if (sp) {
        // telhado com a foto do satélite no mesmo lugar (pedaço escolhido pelo centro do prédio)
        const ll = ring.slice(0, -1), ord = area < 0 ? [...ll].reverse() : ll;
        const pxs = ord.map(([la, lo]) => sp.toPx(la, lo));
        const cx = pxs.reduce((a, p) => a + p[0], 0) / pxs.length, cy = pxs.reduce((a, p) => a + p[1], 0) / pxs.length;
        const k = sp.chunkAt(cx, cy);
        if (k >= 0) {
          const c = sat.meta.chunks[k], R = satRoofs[k];
          for (const t of tris) for (const i of t) {
            R.pos.push(pts[i][0], pts[i][1], pts[i][2] + h);
            R.uv.push((pxs[i][0] - c.px) / c.w, 1 - (pxs[i][1] - c.py) / c.h);
          }
          continue;
        }
      }
      const roofPos = [];
      for (const t of tris) for (const i of t) roofPos.push(pts[i][0], pts[i][1], pts[i][2] + h);
      if (roofPos.length) {
        const gr = new BufferGeometry();
        gr.setAttribute("position", new Float32BufferAttribute(roofPos, 3));
        gr.computeVertexNormals();
        push(roof, gr);
      }
    }
  }

  // chão: o retângulo da área com 0,01° (~1 km) de folga, um pouco abaixo de tudo. Sem a folga, os
  // shots da borda mostram o fundo, e fundo azul vira "rio" no redesenho da IA.
  if (data.bbox) {
    const pad = 0.01;
    const { south: s0, west: w0, north: n0, east: e0 } = data.bbox;
    const [s, w, n, e] = [s0 - pad, w0 - pad, n0 + pad, e0 + pad];
    flat([[[s, w], [s, e], [n, e], [n, w], [s, w]]], [], -0.3, COL.ground);
  }
  if (!sat) {
    for (const f of data.green ?? []) flat(f.rings, f.holes, -0.2, COL[f.kind] ?? COL.green);
    for (const f of data.water ?? []) flat(f.rings, f.holes, -0.1, COL.water);
    for (const r of data.roads ?? []) strip(r.line, r.w, 0, r.major ? COL.major : r.kind === "rail" ? COL.rail : COL.road);
  } else {
    // Por cima da foto: rio do OSM em azul de mapa e ruas em cinza. Sem isso o modelo lia rua de terra
    // (vermelha) como canal e sombra de mata (azul-escura) como lago. Sem luz: cor chapada, igual à foto.
    for (const f of data.water ?? []) flat(f.rings, f.holes, -0.1, SAT.water);
    for (const r of data.roads ?? []) if (r.w >= 3) strip(r.line, r.w, -0.05, r.major ? SAT.major : SAT.road); // sem trilhas
  }
  for (const b of data.buildings ?? []) building(b);

  const group = new Group();
  // Satélite em pedaços carregados sob demanda: um material por pedaço, sem textura até useChunks().
  const chunkMats = [], chunkMeshes = [];
  if (sat) {
    // chão: uma malha 48x48 por pedaço da mosaica, com UV exato (Mercator não é linear em lat/lon)
    const G = 48;
    sat.meta.chunks.forEach((c, k) => {
      const pos = [], uv = [], idx = [];
      for (let iy = 0; iy <= G; iy++) for (let ix = 0; ix <= G; ix++) {
        const x = c.px + (c.w * ix) / G, y = c.py + (c.h * iy) / G;
        pos.push(...local(sp.toLL(x, y), -0.2));
        uv.push(ix / G, 1 - iy / G);
      }
      for (let iy = 0; iy < G; iy++) for (let ix = 0; ix < G; ix++) {
        const a = iy * (G + 1) + ix, b = a + 1, d = a + G + 1, e = d + 1;
        idx.push(a, d, b, b, d, e);
      }
      const g = new BufferGeometry();
      g.setAttribute("position", new Float32BufferAttribute(pos, 3));
      g.setAttribute("uv", new Float32BufferAttribute(uv, 2));
      g.setIndex(idx);
      const mat = new MeshBasicMaterial({ map: sat.textures?.[k] ?? null });
      const ms = [new Mesh(g, mat)];
      const R = satRoofs[k];
      if (R.pos.length) {
        const gr = new BufferGeometry();
        gr.setAttribute("position", new Float32BufferAttribute(R.pos, 3));
        gr.setAttribute("uv", new Float32BufferAttribute(R.uv, 2));
        ms.push(new Mesh(gr, mat));
      }
      for (const m of ms) { m.visible = !!mat.map; group.add(m); }
      chunkMats.push(mat); chunkMeshes.push(ms);
    });
  }
  for (const [color, geoms] of buckets) {
    for (let i = 0; i < geoms.length; i += 20000) {
      const merged = mergeGeometries(geoms.slice(i, i + 20000), false);
      const Mat = sat && FLAT.has(color) ? MeshBasicMaterial : MeshLambertMaterial;
      if (merged) group.add(new Mesh(merged, new Mat({ color: new Color(color) })));
    }
  }
  group.matrixAutoUpdate = false;
  group.matrix.copy(enu); // ENU local -> ECEF, o mesmo referencial dos 3D Tiles
  group.matrixWorldNeedsUpdate = true;

  // sol do fim da tarde vindo do noroeste, alto o bastante para não esconder ruas
  const sunEnu = new Vector3(-0.45, 0.55, 0.7).normalize();
  const sun = new DirectionalLight(0xffffff, 2.2);
  const rot = new Matrix4().extractRotation(enu);
  sun.position.copy(sunEnu.applyMatrix4(rot).multiplyScalar(1e4)).add(new Vector3().setFromMatrixPosition(enu));
  sun.target.position.setFromMatrixPosition(enu);
  const amb = new AmbientLight(0xffffff, 0.9);
  return { group, lights: [sun, sun.target, amb], chunkMats, chunkMeshes,
    stats: { buckets: buckets.size, buildings: (data.buildings ?? []).length } };
}
