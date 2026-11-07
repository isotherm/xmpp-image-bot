#!/usr/bin/env python3
"""Offline square slideshow image processor."""
import argparse
from pathlib import Path
import textwrap
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
    arr = np.asarray(rgb)
    h, w = arr.shape[:2]
    detector = cv2.FaceDetectorYN.create(str(model), '', (w, h), score_threshold=0.65)
    _, faces = detector.detect(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    for face in faces:
        print(
            f"x={face[0]:.0f}, y={face[1]:.0f}, "
            f"w={face[2]:.0f}, h={face[3]:.0f}, "
            f"confidence={face[14]:.3f}"
        )
    return [tuple(map(float, row[:4])) for row in faces] if faces is not None else []

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
            for x, y, fw, fh in faces:
                # Give full-face preservation more weight than centering.
                overlap_w = max(0, min(left + side, x + fw) - max(left, x))
                overlap_h = max(0, min(top + side, y + fh) - max(top, y))
                visible = (overlap_w * overlap_h) / max(1, fw * fh)
                cx, cy = x + fw / 2, y + fh / 2
                center_inside = left <= cx <= left + side and top <= cy <= top + side
                score += 10 * visible**2 + (2 if center_inside else 0)
                # Prefer some breathing room around faces.
                margin = min(cx - left, left + side - cx, cy - top, top + side - cy)
                score += max(0, min(margin / side, 0.25))
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

def process(src, dst, caption='', face_model=None, font=None):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert('RGB')
    faces = detect_faces(rgb, face_model)
    crop = choose_crop(rgb, faces)
    result = rgb.crop(crop).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    if caption:
        result = caption_image(result, caption, font)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    result.save(dst, 'JPEG', quality=92, optimize=True)
    return crop, len(faces)

def main():
    p = argparse.ArgumentParser(description='Offline smart-crop to 640x640 JPEG for slideshow.digital')
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--caption', default='')
    p.add_argument('--face-model', type=Path, help='Path to local YuNet ONNX model')
    p.add_argument('--font', type=Path)
    args = p.parse_args()
    crop, count = process(args.input, args.output, args.caption, args.face_model, args.font)
    print(f'Saved {args.output}; detected {count} face(s); crop={crop}')

if __name__ == '__main__':
    main()
