#!/usr/bin/env python3
"""Offline square slideshow image processor."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import tempfile
import ctypes
import time
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter, ImageStat

SIZE = 720
FACE_LIBRARY = Path(__file__).resolve().parent / "build" / "libfacepython.so"

def detect_faces(rgb, detector_size=480, min_confidence=0.70):
    """Run native libfacedetection on a thumbnail, returning original-image boxes.

    libfacepython.so must be alongside libfacedetection.so (or otherwise
    discoverable by the Android dynamic linker).
    """
    w, h = rgb.size
    scale = min(1.0, detector_size / max(w, h))
    iw, ih = max(1, round(w * scale)), max(1, round(h * scale))
    small = rgb.resize((iw, ih), Image.Resampling.BILINEAR)
    # libfacedetection expects interleaved BGR, not Pillow's normal RGB.
    pixels = small.tobytes("raw", "BGR")
    data = (ctypes.c_ubyte * len(pixels)).from_buffer_copy(pixels)
    native = ctypes.CDLL(str(FACE_LIBRARY))
    fn = native.detect_faces_bgr
    fn.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int,
                   ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    fn.restype = ctypes.c_int
    max_faces = 100
    results = (ctypes.c_int * (5 * max_faces))()
    start = time.perf_counter()
    count = fn(data, iw, ih, results, max_faces)
    print(f'Native face detection: {time.perf_counter()-start:.3f}s ({iw}x{ih})')
    if count < 0:
        raise RuntimeError(f'Native detector failed: {count}')
    faces = []
    for i in range(count):
        confidence, x, y, fw, fh = results[i*5:i*5+5]
        confidence /= 100.0  # libfacedetection returns integer 0..100
        print(
            f"  raw face {i+1}: "
            f"confidence={confidence:.3f}, "
            f"x={x}, y={y}, w={fw}, h={fh}"
        )
        if confidence >= min_confidence and fw > 0 and fh > 0:
            faces.append((x*w/iw, y*h/ih, fw*w/iw, fh*h/ih, confidence))
    return faces

def detect_faces_multiscale(rgb, sizes, min_confidence):
    # Run larger passes only when requested; merge duplicate boxes by IoU.
    all_faces = []
    for size in dict.fromkeys(sizes):
        detected = detect_faces(rgb, size, min_confidence)
        print(f'  {size}px pass: {len(detected)} accepted face(s)')
        all_faces.extend(detected)
    accepted = []
    for face in sorted(all_faces, key=lambda f: f[4], reverse=True):
        x, y, w, h, _ = face
        duplicate = False
        for a, b, aw, ah, _ in accepted:
            intersection = max(0, min(x+w, a+aw)-max(x, a)) * max(0, min(y+h, b+ah)-max(y, b))
            union = w*h + aw*ah - intersection
            if union > 0 and intersection/union > 0.35:
                duplicate = True
                break
        if not duplicate:
            accepted.append(face)
    return accepted

def choose_crop(image, faces):
    w, h = image.size
    side = min(w, h)
    if w == h:
        return (0, 0, w, h)
    # Score candidate square windows. Prioritize keeping all faces, then their centers.
    # Add a mild central bias for aesthetic composition.
    max_offset = max(w, h) - side
    candidates = sorted(set(round(i * max_offset / max(1, min(300, max_offset))) for i in range(min(300, max_offset) + 1)))
    if faces:
        best = None
        for offset in candidates:
            left, top = (int(offset), 0) if w > h else (0, int(offset))
            score = 0.0
            for x, y, fw, fh, confidence in faces:
                # Give full-face preservation more weight than centering.
                overlap_w = max(0, min(left + side, x + fw) - max(left, x))
                overlap_h = max(0, min(top + side, y + fh) - max(top, y))
                visible = (overlap_w * overlap_h) / max(1, fw * fh)
                cx, cy = x + fw / 2, y + fh / 2
                center_inside = left <= cx <= left + side and top <= cy <= top + side
                weight = (fw * fh) ** 0.5
                score += weight * (10 * visible**2 + (2 if center_inside else 0))
                # Prefer some breathing room around faces.
                margin = min(cx - left, left + side - cx, cy - top, top + side - cy)
                score += weight * max(0, min(margin / side, 0.25))
            score -= 0.08 * abs(offset - max_offset / 2) / max(1, max_offset)
            if best is None or (score, -abs(offset - max_offset / 2)) > best[0]:
                best = ((score, -abs(offset - max_offset / 2)), int(offset))
        offset = best[1]
    else:
        # Pillow-only edge saliency: no NumPy/OpenCV dependency.
        preview = image.convert('L')
        preview.thumbnail((384, 384))
        edges = preview.filter(ImageFilter.FIND_EDGES)
        ew, eh = edges.size

        scores = []
        for candidate in candidates:
            left, top = (candidate, 0) if w > h else (0, candidate)
            rect = (
                round(left * ew / w),
                round(top * eh / h),
                max(round((left + side) * ew / w),
                    round(left * ew / w) + 1),
                max(round((top + side) * eh / h),
                    round(top * eh / h) + 1)
            )
            scores.append(ImageStat.Stat(edges.crop(rect)).sum[0])

        # Normalize saliency across candidate crops.
        low, high = min(scores), max(scores)
        center = max_offset / 2
        best_score = float('-inf')
        offset = round(center)

        for candidate, saliency in zip(candidates, scores):
            normalized = ((saliency - low) / (high - low)
                          if high > low else 0.0)

            # 0 at center, 1 at either extreme.
            distance = abs(candidate - center) / max(1, center)

            # Saliency can shift the crop, but not without a benefit.
            score = normalized - 0.65 * distance

            if score > best_score:
                best_score = score
                offset = candidate
    return (offset, 0, offset + side, side) if w > h else (0, offset, side, offset + side)

def load_font(font_path, size):
    paths = [font_path] if font_path else [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf',
        '/system/fonts/Roboto-Regular.ttf',
    ]
    for path in paths:
        if path and Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def caption_layout(caption, font_path=None):
    caption = ' '.join(caption.split())
    if not caption:
        return None
    draw = ImageDraw.Draw(Image.new('RGB', (SIZE, SIZE)))
    for size in range(38, 15, -2):
        font = load_font(font_path, size)
        lines, line = [], ''
        for word in caption.split():
            proposed = (line + ' ' + word).strip()
            if draw.textbbox((0, 0), proposed, font=font, stroke_width=1)[2] <= SIZE-48 or not line:
                line = proposed
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        line_height = int(size * 1.35)
        if len(lines)*line_height <= SIZE*0.37 and all(
            draw.textbbox((0, 0), t, font=font)[2] <= SIZE-48 for t in lines
        ):
            return (lines, font, line_height, len(lines)*line_height+34)
    raise ValueError('Caption too long to fit legibly; shorten the caption')

def caption_image(image, layout, position='bottom'):
    if layout is None:
        return image
    lines, font, line_height, panel_height = layout
    panel_top = 0 if position == 'top' else SIZE-panel_height
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, panel_top, SIZE, panel_top+panel_height), fill=(0, 0, 0, 175))
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    rd = ImageDraw.Draw(result)
    y = panel_top + 15
    for line in lines:
        bbox = rd.textbbox((0, 0), line, font=font, stroke_width=1)
        x = (SIZE-(bbox[2]-bbox[0]))/2
        rd.text((x, y), line, font=font, fill='white', stroke_width=1, stroke_fill='black')
        y += line_height
    return result.convert('RGB')

def caption_cost(image, crop, faces, panel_height, position):
    """Lower is better: protect faces first, then visually busy regions."""
    left, top, right, bottom = crop
    side = right-left
    y0 = top if position == 'top' else bottom - side*panel_height/SIZE
    y1 = top + side*panel_height/SIZE if position == 'top' else bottom
    cost = 0.0
    for x, y, fw, fh, confidence in faces:
        visible_x = max(0, min(right, x+fw)-max(left, x))
        visible_y = max(0, min(bottom, y+fh)-max(top, y))
        cost += 1000*confidence*(1-visible_x*visible_y/max(1, fw*fh))
        overlap_x = max(0, min(right, x+fw)-max(left, x))
        overlap_y = max(0, min(y1, y+fh)-max(y0, y))
        cost += 1000 * confidence * (overlap_x*overlap_y/max(1, fw*fh))
    # Tiny thumbnail for inexpensive edge density, not full-resolution processing.
    thumb = image.crop((left, y0, right, y1)).convert('L')
    thumb.thumbnail((128, 64))
    if thumb.width > 2 and thumb.height > 2:
        edge = thumb.filter(ImageFilter.FIND_EDGES)
        cost += ImageStat.Stat(edge).mean[0] / 255.0
    return cost

def choose_caption_placement(image, crop, faces, panel_height):
    """Prefer bottom; for portrait photos try a small vertical crop shift."""
    w, h = image.size
    side = min(w, h)
    options = [crop]
    if h > w:
        # Move portrait crop downward to move subjects higher in the output,
        # potentially creating more room for a bottom caption.
        original_top = crop[1]
        for delta in (0.05, 0.10, 0.15):
            new_top = min(h-side, original_top + round(side*delta))
            options.append((0, new_top, side, new_top+side))
    best = None
    for candidate in dict.fromkeys(options):
        for position in ('bottom', 'top'):
            cost = caption_cost(image, candidate, faces, panel_height, position)
            # Modest preference for bottom and minimal deviation from chosen crop.
            cost += (0 if position == 'bottom' else 0.12)
            cost += 0.5*abs(candidate[1]-crop[1])/side
            if best is None or cost < best[0]:
                best = (cost, candidate, position)
    return best[1], best[2]

def write_debug(image, faces, crop, debug_path, caption_position=None, panel_height=0):
    """Save a diagnostic copy with detected faces and the selected crop."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    width = max(3, round(min(image.size) / 250))
    for i, (x, y, fw, fh, confidence) in enumerate(faces, 1):
        draw.rectangle((round(x), round(y), round(x + fw), round(y + fh)),
                       outline='#00ff44', width=width)
        label = f'{i}: {confidence:.3f}'
        font = load_font(None, max(15, round(min(image.size) / 60)))
        tx, ty = round(x), max(0, round(y) - 27)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill='black')
        draw.text((tx, ty), label, font=font, fill='#00ff44')
    draw.rectangle(crop, outline='#ff3030', width=width * 2)
    if caption_position and panel_height:
        side = crop[2]-crop[0]
        band = side*panel_height/SIZE
        y0 = crop[1] if caption_position == 'top' else crop[3]-band
        y1 = y0+band
        draw.rectangle((crop[0], y0, crop[2], y1), outline='#ffff00', width=width)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(debug_path, 'JPEG', quality=90)


def process(src, dst, caption='', detector_sizes=(480,), min_confidence=0.70, font=None, debug_path=None):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert('RGB')
    faces = detect_faces_multiscale(rgb, detector_sizes, min_confidence)
    crop = choose_crop(rgb, faces)
    layout = caption_layout(caption, font)
    position = None
    if layout:
        crop, position = choose_caption_placement(rgb, crop, faces, layout[3])
    if debug_path is not None:
        write_debug(rgb, faces, crop, debug_path, position, layout[3] if layout else 0)
    result = rgb.crop(crop).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if layout:
        result = caption_image(result, layout, position)
    dst = Path(dst)
    fd, tmp = tempfile.mkstemp(prefix='.processing_', suffix='.tmp', dir=dst.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            result.save(output, 'JPEG', quality=92, optimize=True)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return crop, faces, position

def filename_timestamp(src):
    """Prefer filesystem birth time; fall back to modification time.

    The fallback is not guaranteed to be the file's original creation time.
    """
    stat = Path(src).stat()
    timestamp = getattr(stat, 'st_birthtime', None)
    if timestamp is None:
        timestamp = stat.st_mtime
    return datetime.fromtimestamp(timestamp).strftime('%Y%m%d_%H%M%S_%f')


def automatic_output(src, output_dir):
    """Reserve a timestamp filename without overwriting an existing photo."""
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = filename_timestamp(src)
    for n in range(10000):
        suffix = '' if n == 0 else f'_{n}'
        target = folder / f'{stem}{suffix}.jpg'
        try:
            with target.open('xb'):
                pass
            return target
        except FileExistsError:
            continue
    raise RuntimeError('Too many timestamp filename collisions')


def main():
    p = argparse.ArgumentParser(description='Offline smart-crop to 720x720 JPEG for slideshow.digital')
    p.add_argument('input', type=Path, help='Source image')
    p.add_argument('--output-dir', type=Path, required=True, help='Required slideshow output directory')
    p.add_argument('--caption', default='')
    p.add_argument('--detector-size', type=int, default=480, help='Maximum detection dimension (default 480)')
    p.add_argument('--detector-sizes', type=int, nargs='+', help='Optional multiple passes, e.g. 320 640')
    p.add_argument('--min-confidence', type=float, default=0.70, help='Minimum native face confidence, 0..1 (default 0.70)')
    p.add_argument('--font', type=Path)
    p.add_argument('--debug', action='store_true', help='Save face boxes and crop outline in a separate debug folder')
    p.add_argument('--debug-dir', type=Path, help='Debug output directory (default: sibling of output-dir)')
    args = p.parse_args()
    dst = automatic_output(args.input, args.output_dir)
    debug_path = None
    if args.debug:
        debug_dir = args.debug_dir or (args.output_dir.parent / (args.output_dir.name + '_debug'))
        debug_path = debug_dir / (dst.stem + '_debug.jpg')
    try:
        crop, faces, position = process(args.input, dst, args.caption, args.detector_sizes or [args.detector_size], args.min_confidence, args.font, debug_path)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    print(f'Saved {dst}; detected {len(faces)} face(s); crop={crop}; caption={position}')
    for i, (x, y, w, h, confidence) in enumerate(faces, 1):
        print(f'  face {i}: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}, confidence={confidence:.3f}')
    if debug_path:
        print(f'Debug image: {debug_path}')

if __name__ == '__main__':
    main()
