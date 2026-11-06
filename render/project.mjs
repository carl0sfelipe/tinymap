// Project lat/lon geometry into canvas pixels with the renderer's exact camera.
// usage: node project.mjs <run_dir> <in.json> <out.json>
// in:  {"h": 110, "features": [{"id":..., "kind":..., "rings": [[[lat,lon],...],...], "holes": [...]}]}
//   or {"points": [{"name":..., "lat":..., "lon":..., "h":...}]}
import { readFileSync, writeFileSync } from "node:fs";
import { projector } from "./camera.mjs";

const [run, inp, out] = process.argv.slice(2);
const view = JSON.parse(readFileSync(`${run}/view.json`));
const toPx = projector(view);
const data = JSON.parse(readFileSync(inp));
const h = data.h ?? view.target_height_m;
const r1 = (n) => Math.round(n * 10) / 10;
const ring = (r) => r.map(([la, lo]) => toPx(la, lo, h).map(r1));
if (data.features) {
  data.features = data.features.map((f) => ({ ...f, rings: f.rings.map(ring), holes: (f.holes ?? []).map(ring) }));
}
if (data.points) {
  data.points = data.points.map((p) => ({ ...p, px: toPx(p.lat, p.lon, p.h ?? view.target_height_m).map(r1) }));
}
writeFileSync(out, JSON.stringify(data));
