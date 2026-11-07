// Render shots of the canvas to <run>/render/r{row}_c{col}.png (resumable, parallel).
// usage:
//   node capture.mjs <run_dir> [--workers 4]
//   node capture.mjs <run_dir> --probe lat,lon[,name] ...   one 1 km square per point -> <run>/probe/
//   node capture.mjs <run_dir> --overview                    whole canvas in one 2048 px image
import { createServer } from "vite";
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { projector } from "./camera.mjs";

const args = process.argv.slice(2);
const run = args[0];
const opt = (k, d) => (args.includes(k) ? args[args.indexOf(k) + 1] : d);
const view = JSON.parse(readFileSync(`${run}/view.json`));
loadEnv(`${run}/../../.env`);
loadEnv(".env");
loadEnv("../.env");
// maquete OSM (tinymap osm3d) quando view.source = "osm": servida em /scene.json pelo próprio
// servidor do vite (pode ter centenas de MB; não cabe no addInitScript)
const scenePath = `${run}/osm/scene_ll.json`;
if (view.source === "osm" && !existsSync(scenePath)) throw new Error(`rode antes: tinymap osm3d (falta ${scenePath})`);
const auth = { google: process.env.GOOGLE_TILES_API_KEY || undefined, ion: process.env.CESIUM_ION_TOKEN || undefined };

const root = new URL(".", import.meta.url).pathname;
const port = 5175 + Math.floor(Math.random() * 500);
const server = await createServer({
  root, server: { port },
  plugins: [{
    name: "tinymap-scene",
    configureServer(srv) {
      srv.middlewares.use("/scene.json", (_req, res) => {
        res.setHeader("Content-Type", "application/json");
        res.end(readFileSync(scenePath));
      });
      // satélite (tinymap satellite): osm/sat.json e os pedaços osm/sat_r*_c*.jpg
      srv.middlewares.use("/sat.json", (_req, res) => {
        const p = `${run}/osm/sat.json`;
        if (!existsSync(p)) { res.statusCode = 404; return res.end("{}"); }
        res.setHeader("Content-Type", "application/json");
        res.end(readFileSync(p));
      });
      srv.middlewares.use("/sat/", (req, res) => {
        const name = (req.url || "").replace(/^\//, "").split("?")[0];
        const p = `${run}/osm/${name}`;
        if (!/^sat_r\d+_c\d+\.jpg$/.test(name) || !existsSync(p)) { res.statusCode = 404; return res.end(); }
        res.setHeader("Content-Type", "image/jpeg");
        res.end(readFileSync(p));
      });
    },
  }],
});
await server.listen();
const browser = await chromium.launch({
  // CHROME_PATH usa um Chrome já instalado (útil onde o download do Chromium do Playwright é bloqueado)
  ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}),
  args: ["--enable-gpu", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=vulkan", "--enable-unsafe-swiftshader"],
});

async function newPage() {
  const page = await browser.newPage({ viewport: { width: view.shot_px, height: view.shot_px } });
  page.on("pageerror", (e) => console.error("page error:", e.message));
  await page.addInitScript(([v, a]) => { window.VIEW = v; window.AUTH = a; }, [view, auth]);
  await page.goto(`http://localhost:${port}/`);
  await page.waitForFunction(() => window.SCENE_READY === true, null, { timeout: 300_000 });
  return page;
}

// Pedaços do satélite que cada shot enxerga: cantos de cada pedaço projetados no canvas, com folga
// de 512 px (prédios altos inclinam a imagem para cima).
const satMeta = existsSync(`${run}/osm/sat.json`) && view.source === "osm" ? JSON.parse(readFileSync(`${run}/osm/sat.json`)) : null;
const chunkBoxes = satMeta ? (() => {
  const toPx = projector(view), n = 2 ** satMeta.z;
  const ll = (x, y) => {
    const X = (x + satMeta.x0 * 256) / (n * 256), Y = (y + satMeta.y0 * 256) / (n * 256);
    return [Math.atan(Math.sinh(Math.PI * (1 - 2 * Y))) * 180 / Math.PI, X * 360 - 180];
  };
  return satMeta.chunks.map((c) => {
    const pts = [[c.px, c.py], [c.px + c.w, c.py], [c.px, c.py + c.h], [c.px + c.w, c.py + c.h]]
      .map(([x, y]) => toPx(...ll(x, y), view.target_height_m));
    return [Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1])),
      Math.max(...pts.map((p) => p[0])), Math.max(...pts.map((p) => p[1]))];
  });
})() : null;
function chunksFor(spec) {
  const s = view.shot_px;
  const [x0, y0, size] = spec.box ?? [spec.col * s, spec.row * s, s];
  const P = 512;
  return chunkBoxes.map((b, k) => (b[0] < x0 + size + P && b[2] > x0 - P && b[1] < y0 + size + P && b[3] > y0 - P ? k : -1))
    .filter((k) => k >= 0);
}

async function shoot(page, spec, file) {
  if (chunkBoxes) await page.evaluate((idx) => window.useChunks(idx), chunksFor(spec));
  await page.evaluate((s) => window.setShot(s), spec);
  await page.waitForFunction(() => window.TILES_LOADED === true, null, { timeout: 600_000, polling: 500 });
  const url = await page.evaluate(() => window.capture());
  writeFileSync(file, Buffer.from(url.split(",")[1], "base64"));
}

if (args.includes("--overview")) {
  const page = await newPage();
  const size = Math.max(...view.canvas_px);
  await shoot(page, { box: [0, 0, size] }, `${run}/overview.png`);
  console.log(`${run}/overview.png`);
} else if (args.includes("--probe")) {
  mkdirSync(`${run}/probe`, { recursive: true });
  const toPx = projector(view);
  const page = await newPage();
  const side = Math.round(1000 / view.meters_per_px);
  for (const p of args.slice(args.indexOf("--probe") + 1).filter((a) => !a.startsWith("--"))) {
    const [lat, lon, name = `${lat}_${lon}`] = p.split(",");
    const [x, y] = toPx(+lat, +lon, view.target_height_m);
    const file = `${run}/probe/${name.replace(/\W+/g, "_")}.png`;
    await shoot(page, { box: [Math.round(x - side / 2), Math.round(y - side / 2), side] }, file);
    console.log(file);
  }
} else {
  mkdirSync(`${run}/render`, { recursive: true });
  const todo = view.shots.filter(([c, r]) => !existsSync(`${run}/render/r${r}_c${c}.png`));
  const workers = Number(opt("--workers", 2));
  console.log(`${todo.length} shots to render with ${workers} workers`);
  let done = 0;
  await Promise.all(Array.from({ length: workers }, async () => {
    const page = await newPage();
    while (todo.length) {
      const [c, r] = todo.shift();
      const t0 = Date.now();
      try {
        await shoot(page, { col: c, row: r }, `${run}/render/r${r}_c${c}.png`);
        console.log(`[${++done}] r${r} c${c} ${((Date.now() - t0) / 1000).toFixed(1)}s`);
      } catch (e) {
        console.error(`r${r} c${c} failed: ${e.message}; will retry on next run`);
      }
    }
  }));
}
await browser.close();
await server.close();

function loadEnv(path) {
  if (!existsSync(path)) return;
  for (const line of readFileSync(path, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
}
