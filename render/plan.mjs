// Plan the canvas for a region: project the bbox corners with the real camera, snap the
// canvas to whole shots and list the shots that touch the area.
// usage: node plan.mjs <run_dir> [--pilot]
// in: <run_dir>/region.json (written by `tinymap plan`)  out: <run_dir>/view.json
import { readFileSync, writeFileSync } from "node:fs";
import { projector } from "./camera.mjs";

const run = process.argv[2];
const pilot = process.argv.includes("--pilot");
const region = JSON.parse(readFileSync(`${run}/region.json`));
const box = pilot && region.pilot_bbox ? region.pilot_bbox : region.bbox;
const cam = region.camera;
const view = {
  name: region.name,
  lat: region.center.lat, lon: region.center.lon,
  azimuth_deg: cam.azimuth_deg, elevation_deg: cam.elevation_deg,
  target_height_m: cam.target_height_m, meters_per_px: cam.meters_per_px,
  shot_px: cam.shot_px, error_target: cam.error_target ?? 2,
};
const shot = view.shot_px;
const toPx = projector(view, [0, 0]); // camera-centred pixels

// sample the bbox outline densely: the projection of a lat/lon box is not a rectangle
const ring = [];
const N = 64;
for (let i = 0; i <= N; i++) {
  const t = i / N;
  ring.push([box.south, box.west + t * (box.east - box.west)]);
  ring.push([box.north, box.west + t * (box.east - box.west)]);
  ring.push([box.south + t * (box.north - box.south), box.west]);
  ring.push([box.south + t * (box.north - box.south), box.east]);
}
const h = cam.target_height_m;
const pts = ring.map(([la, lo]) => toPx(la, lo, h));
const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
// tall buildings lean "up" the image; leave 1 shot of headroom above
const x0 = Math.floor(Math.min(...xs) / shot) * shot;
const y0 = Math.floor((Math.min(...ys) - shot) / shot) * shot;
const x1 = Math.ceil(Math.max(...xs) / shot) * shot;
const y1 = Math.ceil(Math.max(...ys) / shot) * shot;
view.origin_px = [x0, y0];
view.canvas_px = [x1 - x0, y1 - y0];
view.cols = view.canvas_px[0] / shot;
view.rows = view.canvas_px[1] / shot;

// area polygon in canvas px (convex hull is enough for a box) and the shots that touch it
const poly = hull(pts.map(([x, y]) => [x - x0, y - y0]));
view.area_px = poly;
view.shots = [];
for (let r = 0; r < view.rows; r++) {
  for (let c = 0; c < view.cols; c++) {
    if (rectTouches(poly, c * shot, r * shot - shot, (c + 1) * shot, (r + 1) * shot)) view.shots.push([c, r]);
  }
}
writeFileSync(`${run}/view.json`, JSON.stringify(view, null, 1));
const gpx = (view.canvas_px[0] * view.canvas_px[1]) / 1e9;
console.log(`canvas ${view.canvas_px.join(" x ")} px (${gpx.toFixed(2)} Gpx), ${view.shots.length}/${view.cols * view.rows} shots`);

function hull(p) {
  p = [...p].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length > 1 && cross(lo.at(-2), lo.at(-1), q) <= 0) lo.pop(); lo.push(q); }
  for (const q of p.reverse()) { while (up.length > 1 && cross(up.at(-2), up.at(-1), q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1)).map(([x, y]) => [Math.round(x), Math.round(y)]);
}
function inside(poly, x, y) {
  let c = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}
function rectTouches(poly, ax, ay, bx, by) {
  if (poly.some(([x, y]) => x >= ax && x <= bx && y >= ay && y <= by)) return true;
  for (const [x, y] of [[ax, ay], [bx, ay], [ax, by], [bx, by], [(ax + bx) / 2, (ay + by) / 2]]) if (inside(poly, x, y)) return true;
  return false;
}
