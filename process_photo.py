#!/usr/bin/env python3
"""Offline square slideshow image processor."""
import argparse
from datetime import datetime
from pathlib import Path
import os
import tempfile
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFilter

SIZE = 720

def detect_faces(rgb, model):
    if not model:
        return []
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError('Face detection requires opencv-python') from exc
    if not Path(model).is_file():
        raise FileNotFoundError(f'Face model not found: {model}')
    # YuNet's standard input size is 320x320. Detect at that size and
    # transform face coordinates back into the original image space.
    w, h = rgb.size
    input_w = input_h = 320
    arr = np.asarray(rgb.resize((input_w, input_h), Image.Resampling.BILINEAR))
    detector = cv2.FaceDetectorYN.create(str(model), '', (input_w, input_h), score_threshold=0.9)
    try:
        _, faces = detector.detect(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    except cv2.error as exc:
        raise RuntimeError(
            f'YuNet model incompatible with OpenCV {cv2.__version__}; '
            'try the original 2022mar YuNet ONNX model or upgrade OpenCV. '
            f'Model: {model}'
        ) from exc
    if faces is None:
        return []
    return [(float(x) * w / input_w, float(y) * h / input_h,
             float(fw) * w / input_w, float(fh) * h / input_h, float(row[14]))
            for row in faces for x, y, fw, fh in [row[:4]]]

def choose_crop(image, faces):
    w, h = image.size
    side = min(w, h)
    if w == h:
        return (0, 0, w, h)
    # Score candidate square windows. Prioritize keeping all faces, then their centers.
    # Add a mild central bias for aesthetic composition.
    max_offset = max(w, h) - side
    candidates = np.unique(np.linspace(0, max_offset, min(301, max_offset + 1)).astype(int))
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
        # Edge/detail saliency: prefer windows with visual information, but
        # gently discourage extreme off-center crops.
        gray = np.asarray(image.convert('L').resize((max(1, round(w * min(1, 512 / max(w,h)))), max(1, round(h * min(1, 512 / max(w,h)))))), dtype=np.float32)
        gy, gx = np.gradient(gray)
        detail = np.hypot(gx, gy)
        scale_x = gray.shape[1] / w
        scale_y = gray.shape[0] / h
        scores = []
        for offset in candidates:
            l, t = (int(offset), 0) if w > h else (0, int(offset))
            patch = detail[round(t * scale_y):max(round((t + side) * scale_y), round(t * scale_y)+1), round(l * scale_x):max(round((l + side) * scale_x), round(l * scale_x)+1)]
            scores.append(float(patch.mean()) - 0.12 * float(detail.mean()) * abs(offset - max_offset / 2) / max(1, max_offset))
        offset = int(candidates[int(np.argmax(scores))])
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


def process(src, dst, caption='', face_model=None, font=None, debug_path=None):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert('RGB')
    faces = detect_faces(rgb, face_model)
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
    p.add_argument('--face-model', type=Path, help='Path to local YuNet ONNX model')
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
        crop, faces = process(args.input, dst, args.caption, args.face_model, args.font, debug_path)
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
