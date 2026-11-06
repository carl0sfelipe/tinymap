// The one camera shared by the renderer, the planner and every projection.
// Canvas pixel (x, y) is measured from the canvas top-left; view.origin_px is that
// top-left relative to the camera's centre point.
import { WGS84_ELLIPSOID, CAMERA_FRAME } from "3d-tiles-renderer";
import { Matrix4, Vector3, MathUtils } from "three";

export function cameraWorld(view) {
  const m = new Matrix4();
  WGS84_ELLIPSOID.getObjectFrame(
    view.lat * MathUtils.DEG2RAD, view.lon * MathUtils.DEG2RAD, view.target_height_m,
    view.azimuth_deg * MathUtils.DEG2RAD, view.elevation_deg * MathUtils.DEG2RAD, 0,
    m, CAMERA_FRAME,
  );
  // pull the camera back along its view axis so nothing tall is clipped
  return m.multiply(new Matrix4().makeTranslation(0, 0, 5000));
}

// Returns (lat, lon, h) -> [x, y] in canvas pixels. Pass origin [0, 0] for camera-centred pixels.
export function projector(view, origin = view.origin_px ?? [0, 0]) {
  const inv = cameraWorld(view).invert();
  const s = view.meters_per_px;
  const v = new Vector3();
  return (lat, lon, h) => {
    WGS84_ELLIPSOID.getCartographicToPosition(lat * MathUtils.DEG2RAD, lon * MathUtils.DEG2RAD, h, v);
    v.applyMatrix4(inv);
    return [v.x / s - origin[0], -v.y / s - origin[1]];
  };
}
