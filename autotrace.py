"""
Auto-trace module: extract KM curve coordinates from image using OpenCV.

Pipeline:
1. Color masking in HSV space
2. Morphological cleanup
3. Skeletonization
4. Per-column coordinate extraction
5. Douglas-Peucker simplification
"""

import cv2
import numpy as np


def pick_color_mask(image_bgr, target_rgb, tolerance=30):
    """
    Create a binary mask for pixels matching the target color in HSV space.

    Parameters
    ----------
    image_bgr : np.ndarray (H, W, 3) BGR image
    target_rgb : tuple (R, G, B) 0-255
    tolerance : int, hue tolerance in HSV space

    Returns
    -------
    mask : np.ndarray (H, W) binary mask
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Convert target RGB to HSV
    target_bgr = np.uint8([[list(reversed(target_rgb))]])
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]

    h, s, v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])

    # HSV ranges
    h_lo = max(0, h - tolerance)
    h_hi = min(179, h + tolerance)
    s_lo = max(0, s - 80)
    s_hi = min(255, s + 80)
    v_lo = max(0, v - 80)
    v_hi = min(255, v + 80)

    # Handle hue wrap-around (red hues near 0/180)
    if h_lo < 0 or h_hi > 179:
        mask1 = cv2.inRange(hsv, np.array([0, s_lo, v_lo]), np.array([h_hi % 180, s_hi, v_hi]))
        mask2 = cv2.inRange(hsv, np.array([h_lo % 180, s_lo, v_lo]), np.array([179, s_hi, v_hi]))
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]), np.array([h_hi, s_hi, v_hi]))

    return mask


def morphological_cleanup(mask, kernel_size=3):
    """Remove noise and fill small gaps."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def skeletonize(mask):
    """Thin the mask to 1px width."""
    try:
        return cv2.ximgproc.thinning(mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    except AttributeError:
        # Fallback: scikit-image skeletonize
        from skimage.morphology import skeletonize as ski_skel
        binary = mask > 0
        skel = ski_skel(binary)
        return (skel.astype(np.uint8) * 255)


def _get_x_bounds(shape, x_range=None):
    """Normalize x-range bounds for a mask/skeleton image."""
    _, w = shape
    if x_range:
        x_start, x_end = max(0, int(x_range[0])), min(w, int(x_range[1]))
    else:
        x_start, x_end = 0, w
    return x_start, x_end


def extract_coordinates_topmost(skeleton, x_range=None):
    """
    Extract (x, y) pixel coordinates from skeleton.
    For each x column, take the topmost (for survival) skeleton pixel.

    Parameters
    ----------
    skeleton : np.ndarray (H, W) binary skeleton
    x_range : tuple (x_min, x_max) pixel range to scan, or None for full width

    Returns
    -------
    points : list of (x, y) pixel coordinates
    """
    x_start, x_end = _get_x_bounds(skeleton.shape, x_range)

    points = []
    for x in range(x_start, x_end):
        col = skeleton[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) > 0:
            # Take topmost pixel (smallest y = highest on screen = highest survival)
            y = int(ys.min())
            points.append((x, y))

    return points


def _nearest_y_in_column(skeleton, x, ref_y):
    """Return the curve pixel in column x nearest to ref_y."""
    col = skeleton[:, x]
    ys = np.where(col > 0)[0]
    if len(ys) == 0:
        return None
    return int(ys[np.argmin(np.abs(ys - ref_y))])


def _find_seed_pixel(skeleton, seed_point, x_start, x_end, search_radius=10):
    """Find the skeleton pixel nearest to the clicked seed point."""
    if not seed_point or len(seed_point) < 2:
        return None

    seed_x = int(round(seed_point[0]))
    seed_y = int(round(seed_point[1]))
    seed_x = min(max(seed_x, x_start), x_end - 1)

    best = None
    best_score = None
    seen = set()

    for dx in range(search_radius + 1):
        for x in (seed_x - dx, seed_x + dx):
            if x in seen or x < x_start or x >= x_end:
                continue
            seen.add(x)

            y = _nearest_y_in_column(skeleton, x, seed_y)
            if y is None:
                continue

            score = abs(x - seed_x) * 4 + abs(y - seed_y)
            if best_score is None or score < best_score:
                best = (x, y)
                best_score = score

    return best


def _interpolate_gap(x0, y0, x1, y1):
    """Linearly fill missing columns between two accepted points."""
    if x0 == x1:
        return []

    step = 1 if x1 > x0 else -1
    span = abs(x1 - x0)
    filled = []
    for i in range(1, span):
        x = x0 + step * i
        t = i / span
        y = int(round(y0 + (y1 - y0) * t))
        filled.append((x, y))
    return filled


def _follow_curve_direction(skeleton, start_point, direction, x_start, x_end,
                            max_jump=32, max_gap=60):
    """
    Follow the same curve left or right by choosing the y nearest to the
    previously accepted point in each column.
    """
    points = []
    prev_x, prev_y = start_point
    x = prev_x + direction

    while x_start <= x < x_end:
        y = _nearest_y_in_column(skeleton, x, prev_y)
        if y is not None and abs(y - prev_y) <= max_jump:
            points.append((x, y))
            prev_x, prev_y = x, y
            x += direction
            continue

        found = None
        for gap in range(1, max_gap + 1):
            nx = x + direction * gap
            if nx < x_start or nx >= x_end:
                break

            ny = _nearest_y_in_column(skeleton, nx, prev_y)
            if ny is None:
                continue

            allowed_jump = max_jump + min(18, gap * 2)
            if abs(ny - prev_y) <= allowed_jump:
                found = (nx, ny)
                break

        if found is None:
            break

        nx, ny = found
        points.extend(_interpolate_gap(prev_x, prev_y, nx, ny))
        points.append((nx, ny))
        prev_x, prev_y = nx, ny
        x = nx + direction

    return points


def extract_coordinates(skeleton, x_range=None, seed_point=None):
    """
    Extract curve coordinates from a binary skeleton.

    If a seed point is provided, follow the clicked curve continuously across
    the image. Otherwise, fall back to the topmost pixel per column.
    """
    x_start, x_end = _get_x_bounds(skeleton.shape, x_range)

    seed = _find_seed_pixel(skeleton, seed_point, x_start, x_end)
    if seed is None:
        return extract_coordinates_topmost(skeleton, x_range=x_range)

    left = _follow_curve_direction(
        skeleton, seed, direction=-1, x_start=x_start, x_end=x_end
    )
    right = _follow_curve_direction(
        skeleton, seed, direction=1, x_start=x_start, x_end=x_end
    )

    ordered = {}
    for x, y in reversed(left):
        ordered[x] = y
    ordered[seed[0]] = seed[1]
    for x, y in right:
        ordered[x] = y

    return sorted(ordered.items())


def douglas_peucker_simplify(points, epsilon=2.0):
    """
    Simplify point list using Douglas-Peucker algorithm via OpenCV.

    Parameters
    ----------
    points : list of (x, y) tuples
    epsilon : float, simplification tolerance in pixels

    Returns
    -------
    simplified : list of (x, y) tuples
    """
    if len(points) < 3:
        return points

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(pts, epsilon, closed=False)
    return [(int(p[0][0]), int(p[0][1])) for p in simplified]


def calibrate_points(pixel_points, cal_x, cal_y):
    """
    Convert pixel coordinates to data coordinates using calibration.

    Parameters
    ----------
    pixel_points : list of (px, py) tuples
    cal_x : dict with 'pixel_a', 'pixel_b', 'value_a', 'value_b'
    cal_y : dict with 'pixel_a', 'pixel_b', 'value_a', 'value_b'

    Returns
    -------
    data_points : list of (time, survival) tuples
    """
    px_a_x = cal_x["pixel_a"]
    px_b_x = cal_x["pixel_b"]
    val_a_x = cal_x["value_a"]
    val_b_x = cal_x["value_b"]

    px_a_y = cal_y["pixel_a"]
    px_b_y = cal_y["pixel_b"]
    val_a_y = cal_y["value_a"]
    val_b_y = cal_y["value_b"]

    x_scale = (val_b_x - val_a_x) / (px_b_x - px_a_x) if (px_b_x - px_a_x) != 0 else 1
    y_scale = (val_b_y - val_a_y) / (px_b_y - px_a_y) if (px_b_y - px_a_y) != 0 else 1

    data_points = []
    for px, py in pixel_points:
        data_x = val_a_x + (px - px_a_x) * x_scale
        data_y = val_a_y + (py - px_a_y) * y_scale
        data_points.append((round(data_x, 6), round(data_y, 6)))

    return data_points


def autotrace(image_bgr, target_rgb, calibration, tolerance=30, epsilon=2.0,
              x_pixel_range=None, seed_point=None):
    """
    Full auto-trace pipeline.

    Parameters
    ----------
    image_bgr : np.ndarray BGR image
    target_rgb : tuple (R, G, B) - picked curve color
    calibration : dict with 'x' and 'y' calibration dicts
    tolerance : int, color tolerance
    epsilon : float, Douglas-Peucker simplification tolerance
    x_pixel_range : tuple (x_min, x_max) or None

    Returns
    -------
    dict with:
        'pixel_points': list of (px, py) simplified pixel coords
        'data_points': list of (time, survival) calibrated coords
        'mask_preview': base64 PNG of the color mask (for debugging)
    """
    import base64

    # 1. Color mask
    mask = pick_color_mask(image_bgr, target_rgb, tolerance)

    # 2. Morphological cleanup
    mask_clean = morphological_cleanup(mask)

    # 3. Skeletonize
    try:
        skel = skeletonize(mask_clean)
    except Exception:
        # Fallback: use cleaned mask directly
        skel = mask_clean

    # 4. Extract per-column coordinates
    raw_points = extract_coordinates(
        skel, x_range=x_pixel_range, seed_point=seed_point
    )

    if not raw_points:
        return {
            "pixel_points": [],
            "data_points": [],
            "mask_preview": None,
            "error": "No curve pixels found for the selected color"
        }

    # 5. Simplify
    simplified = douglas_peucker_simplify(raw_points, epsilon)

    # 6. Calibrate
    data_points = calibrate_points(simplified, calibration["x"], calibration["y"])

    # 7. Mask preview for debugging
    _, mask_png = cv2.imencode(".png", mask_clean)
    mask_b64 = base64.b64encode(mask_png).decode("utf-8")

    return {
        "pixel_points": simplified,
        "data_points": data_points,
        "mask_preview": mask_b64,
    }
