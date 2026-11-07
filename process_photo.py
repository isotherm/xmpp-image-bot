#!/usr/bin/env python3
"""Offline square slideshow image processor."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import tempfile
import ctypes
import time
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

SIZE = 720
FACE_LIBRARY = Path(__file__).resolve().parent / "build" / "libfacepython.so"

def detect_faces(rgb, detector_size=320, min_confidence=0.85):
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
            if best is None or score > best[0]:
                best = (score, int(offset))
        offset = best[1]
    else:
        # Pillow-only edge saliency: no NumPy/OpenCV dependency.
        preview = image.convert('L')
        preview.thumbnail((384, 384))
        edges = preview.filter(ImageFilter.FIND_EDGES)
        ew, eh = edges.size
        from PIL import ImageStat
        best = None
        for candidate in candidates:
            left, top = (candidate, 0) if w > h else (0, candidate)
            rect = (round(left*ew/w), round(top*eh/h),
                    max(round((left+side)*ew/w), round(left*ew/w)+1),
                    max(round((top+side)*eh/h), round(top*eh/h)+1))
            score = ImageStat.Stat(edges.crop(rect)).mean[0]
            score -= 0.5 * abs(candidate - max_offset/2) / max(1, max_offset)
            if best is None or score > best[0]:
                best = (score, candidate)
        offset = best[1]
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

def caption_image(image, caption, font_path=None):
    caption = ' '.join(caption.split())
    if not caption:
        return image
    max_width = SIZE - 48
    draw = ImageDraw.Draw(image)
    for size in range(38, 15, -2):
        font = load_font(font_path, size)
        words = caption.split()
        lines = []
        line = ''
        for word in words:
            proposed = (line + ' ' + word).strip()
            if draw.textbbox((0, 0), proposed, font=font, stroke_width=1)[2] <= max_width or not line:
                line = proposed
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        # Max caption area: 40% of picture.
        line_height = int(size * 1.35)
        if len(lines) * line_height <= SIZE * 0.37 and all(draw.textbbox((0,0), line, font=font)[2] <= max_width for line in lines):
            break
    else:
        raise ValueError('Caption too long to fit legibly; shorten the caption')
    panel_height = len(lines) * line_height + 34
    overlay = Image.new('RGBA', image.size, (0,0,0,0))
    od = ImageDraw.Draw(overlay)
    od.rectangle((0, 0, SIZE, panel_height), fill=(0,0,0,175))
    result = Image.alpha_composite(image.convert('RGBA'), overlay)
    rd = ImageDraw.Draw(result)
    y = 15
    for line in lines:
        bbox = rd.textbbox((0,0), line, font=font, stroke_width=1)
        x = (SIZE - (bbox[2] - bbox[0])) / 2
        rd.text((x, y), line, font=font, fill='white', stroke_width=1, stroke_fill='black')
        y += line_height
    return result.convert('RGB')

def write_debug(image, faces, crop, debug_path):
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
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(debug_path, 'JPEG', quality=90)


def process(src, dst, caption='', detector_size=320, min_confidence=0.85, font=None, debug_path=None):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert('RGB')
    faces = detect_faces(rgb, detector_size, min_confidence)
    crop = choose_crop(rgb, faces)
    if debug_path is not None:
        write_debug(rgb, faces, crop, debug_path)
    result = rgb.crop(crop).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if caption:
        result = caption_image(result, caption, font)
    # Keep partially written JPEGs out of the slideshow folder.
    dst = Path(dst)
    fd, tmp = tempfile.mkstemp(prefix='.processing_', suffix='.tmp', dir=dst.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            result.save(output, 'JPEG', quality=92, optimize=True)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return crop, faces

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
    p.add_argument('--detector-size', type=int, default=320, help='Maximum detection thumbnail dimension (default 320)')
    p.add_argument('--min-confidence', type=float, default=0.85, help='Minimum native face confidence, 0..1 (default 0.85)')
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
        crop, faces = process(args.input, dst, args.caption, args.detector_size, args.min_confidence, args.font, debug_path)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    print(f'Saved {dst}; detected {len(faces)} face(s); crop={crop}')
    for i, (x, y, w, h, confidence) in enumerate(faces, 1):
        print(f'  face {i}: x={x:.0f}, y={y:.0f}, w={w:.0f}, h={h:.0f}, confidence={confidence:.3f}')
    if debug_path:
        print(f'Debug image: {debug_path}')

if __name__ == '__main__':
    main()
