// One orthographic camera for the whole canvas. A shot is a sub-rectangle of its frustum,
// so shots line up to the pixel without moving the camera (method of isometric.nyc / tiny.london).
// capture.mjs injects window.VIEW and window.AUTH before the page loads.
import { TilesRenderer } from "3d-tiles-renderer";
import {
  GoogleCloudAuthPlugin, CesiumIonAuthPlugin, GLTFExtensionsPlugin, TileCompressionPlugin,
} from "3d-tiles-renderer/plugins";
import { Scene, WebGLRenderer, OrthographicCamera, TextureLoader, SRGBColorSpace } from "three";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import { cameraWorld } from "./camera.mjs";
import { buildOsmScene, GROUND } from "./osmscene.mjs";

const view = window.VIEW;
const auth = window.AUTH ?? {};
const shot = view.shot_px;
const renderer = new WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setClearColor(0x87ceeb);
renderer.setPixelRatio(1);
renderer.setSize(shot, shot);
document.body.appendChild(renderer.domElement);

const scene = new Scene();
const camera = new OrthographicCamera(-1, 1, 1, -1, 1, 20000);
cameraWorld(view).decompose(camera.position, camera.quaternion, camera.scale);

let stableSince = 0;
let frames = 0;
let texBusy = 0; // texturas do satélite carregando
window.TILES_LOADED = false;

// Move the frustum to shot (col,row) of the canvas, or to a free box (overview / probe).
window.setShot = ({ col, row, box }) => {
  const s = view.meters_per_px;
  const [ox, oy] = view.origin_px;
  let left, top, size;
  if (box) {           // box = [x0, y0, size] in canvas px, rendered into `shot` px
    [left, top, size] = [(ox + box[0]) * s, (-oy - box[1]) * s, box[2] * s];
  } else {
    [left, top, size] = [(ox + col * shot) * s, (-oy - row * shot) * s, shot * s];
  }
  Object.assign(camera, { left, right: left + size, top, bottom: top - size });
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  window.TILES_LOADED = false;
  stableSince = 0;
  frames = 0;
};

// Fonte da cidade: Google 3D Tiles (padrão) ou maquete OSM (view.source = "osm"), injetada por
// capture.mjs em /scene.json. Com OSM não há streaming: a cena está pronta no primeiro quadro.
const useOsm = view.source === "osm";
let tiles = null;
if (useOsm) {
  const data = await (await fetch("/scene.json")).json();
  // satélite projetado no chão e nos telhados, quando existir (tinymap satellite)
  let sat = null;
  const sr = await fetch("/sat.json");
  if (sr.ok) sat = { meta: await sr.json(), textures: [] };
  const built = buildOsmScene(view, data, sat, { satOnly: !!window.OPTS?.satOnly });
  const { group, lights } = built;
  if (sat) {
    // Só os pedaços de satélite que o shot enxerga ficam na GPU (a região toda tem ~150 de 4096 px).
    const loader = new TextureLoader(), cache = new Map(), MAX = 8;
    window.useChunks = async (idx) => {
      texBusy++;
      try {
        await Promise.all(idx.map(async (k) => {
          if (cache.has(k)) { const t = cache.get(k); cache.delete(k); cache.set(k, t); return; }
          const t = await loader.loadAsync(`/sat/${sat.meta.chunks[k].file}`);
          t.colorSpace = SRGBColorSpace;
          t.anisotropy = renderer.capabilities.getMaxAnisotropy();
          cache.set(k, t);
        }));
        while (cache.size > MAX) {
          const [k, t] = cache.entries().next().value;
          if (idx.includes(k)) break;
          cache.delete(k); t.dispose();
        }
        built.chunkMats.forEach((m, k) => {
          const on = idx.includes(k) && cache.has(k);
          m.map = on ? cache.get(k) : null; m.needsUpdate = true;
          for (const me of built.chunkMeshes[k]) me.visible = on;
        });
      } finally { texBusy--; }
    };
  }
  scene.add(group, ...lights);
  renderer.setClearColor(GROUND); // o que sobrar fora do chão é terra, nunca céu/água
} else {
  tiles = new TilesRenderer();
}
if (useOsm) {
  // nada a configurar
} else if (auth.google) {
  tiles.registerPlugin(new GoogleCloudAuthPlugin({ apiToken: auth.google, autoRefreshToken: true }));
} else if (auth.ion) {
  // Google Photorealistic 3D Tiles through Cesium ion (asset 2275207)
  tiles.registerPlugin(new CesiumIonAuthPlugin({ apiToken: auth.ion, assetId: "2275207", autoRefreshToken: true }));
} else {
  throw new Error("set GOOGLE_TILES_API_KEY or CESIUM_ION_TOKEN in .env, or use source: osm");
}
if (tiles) {
  tiles.registerPlugin(new TileCompressionPlugin());
  tiles.registerPlugin(new GLTFExtensionsPlugin({
    dracoLoader: new DRACOLoader().setDecoderPath("https://unpkg.com/three@0.153.0/examples/jsm/libs/draco/gltf/"),
  }));
  tiles.errorTarget = view.error_target ?? 2;
  tiles.lruCache.maxSize = 40000;
  tiles.lruCache.minSize = 20000;
  if ("maxBytesSize" in tiles.lruCache) tiles.lruCache.maxBytesSize = 4e9;
  tiles.setCamera(camera);
  tiles.setResolution(camera, shot, shot);
  scene.add(tiles.group);
}

function animate() {
  requestAnimationFrame(animate);
  tiles?.update();
  renderer.render(scene, camera);
  frames++;
  const busy = (tiles ? tiles.stats.downloading + tiles.stats.parsing + (tiles.stats.queued ?? 0) : 0) + texBusy;
  if (busy === 0 && frames > 60) {
    stableSince ||= performance.now();
    window.TILES_LOADED = performance.now() - stableSince > 2000;
  } else {
    stableSince = 0;
    window.TILES_LOADED = false;
  }
}
window.setShot({ col: 0, row: 0 });
animate();
window.capture = () => renderer.domElement.toDataURL("image/png");
window.SCENE_READY = true; // cena (e texturas do satélite) carregada
