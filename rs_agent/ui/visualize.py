"""
결과 시각화 유틸리티

실제 모델 출력이 없는 mock 환경에서도
시각적으로 의미있는 결과 이미지를 생성합니다.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import List, Dict, Tuple, Optional
import random

# ── 색상 팔레트 ─────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "vehicle":   (255, 80,  80),
    "ship":      (80,  160, 255),
    "aircraft":  (80,  220, 80),
    "building":  (255, 180, 0),
    "road":      (180, 100, 255),
    "person":    (255, 100, 200),
    "default":   (255, 220, 50),
}

SEG_PALETTE = [
    (68,  1,   84),   # 0: 배경
    (59,  82,  139),  # 1: 수계
    (33,  145, 140),  # 2: 녹지
    (94,  201, 98),   # 3: 농경지
    (253, 231, 37),   # 4: 도시
    (200, 100, 50),   # 5: 나지
    (150, 150, 150),  # 6: 도로
    (255, 255, 255),  # 7: 구름
]


def _load_pil(image) -> Image.Image:
    """입력을 PIL Image로 변환 (numpy, path, PIL 모두 허용)."""
    if image is None:
        return Image.new("RGB", (512, 512), (50, 50, 60))
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).clip(0, 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image)
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    return image.convert("RGB")


def draw_detections(
    image,
    categories: List[str],
    n_objects: Optional[int] = None,
    rng_seed: int = 42,
) -> Tuple[Image.Image, List[Dict]]:
    """
    탐지 결과 시각화: 이미지 위에 bounding box와 레이블을 그립니다.
    Returns: (result_image, detections_list)
    """
    img = _load_pil(image).resize((512, 512)).convert("RGB")
    draw = ImageDraw.Draw(img)
    rng = random.Random(rng_seed)
    w, h = img.size

    if n_objects is None:
        n_objects = rng.randint(3, 10)

    detections = []
    for i in range(n_objects):
        cat = rng.choice(categories)
        color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["default"])

        bw = rng.randint(30, 100)
        bh = rng.randint(20, 70)
        x1 = rng.randint(10, w - bw - 10)
        y1 = rng.randint(10, h - bh - 10)
        x2, y2 = x1 + bw, y1 + bh
        conf = round(rng.uniform(0.65, 0.99), 2)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = f"{cat} {conf:.2f}"
        draw.rectangle([x1, y1 - 16, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 15), label, fill=(255, 255, 255))

        detections.append({
            "category": cat, "confidence": conf,
            "bbox": [x1, y1, x2, y2],
        })

    return img, detections


def draw_segmentation(image, n_classes: int = 5) -> Tuple[Image.Image, Image.Image]:
    """
    시맨틱 분할 시각화: 컬러 마스크와 오버레이 이미지를 반환합니다.
    Returns: (overlay_image, colormap_image)
    """
    img = _load_pil(image).resize((512, 512)).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    # 간단한 Voronoi-like 분할 (seed point 기반)
    rng = np.random.default_rng(99)
    n = min(n_classes, len(SEG_PALETTE))
    seeds = rng.integers(0, [h, w], size=(n, 2))
    labels = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    dists = np.stack([
        (yy - s[0]) ** 2 + (xx - s[1]) ** 2 for s in seeds
    ])
    labels = np.argmin(dists, axis=0).astype(np.uint8)

    # 컬러맵 이미지
    colormap = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(SEG_PALETTE[:n]):
        colormap[labels == i] = color
    colormap_img = Image.fromarray(colormap)

    # 원본과 오버레이 합성
    overlay = Image.blend(img, colormap_img, alpha=0.45)
    return overlay, colormap_img


def draw_change_map(
    before: Image.Image,
    after: Image.Image,
) -> Tuple[Image.Image, Image.Image, Dict]:
    """
    변화 탐지 시각화: before/after 비교와 변화 맵을 반환합니다.
    Returns: (side_by_side, change_map, statistics)
    """
    sz = (256, 256)
    b = _load_pil(before).resize(sz).convert("RGB")
    a = _load_pil(after).resize(sz).convert("RGB")

    b_arr = np.array(b, dtype=float)
    a_arr = np.array(a, dtype=float)

    diff = np.abs(a_arr - b_arr).mean(axis=2)
    max_diff = diff.max()
    diff_norm = (diff / max_diff * 255).astype(np.uint8) if max_diff > 0 else np.zeros_like(diff, dtype=np.uint8)

    # 임계값으로 변화 마스크 생성
    threshold = np.percentile(diff_norm, 75)
    change_mask = diff_norm > threshold

    # 변화 맵: 빨간색으로 강조
    change_map_arr = np.zeros((*diff_norm.shape, 3), dtype=np.uint8)
    change_map_arr[~change_mask] = [30, 30, 30]
    change_map_arr[change_mask] = [220, 60, 60]
    change_map_img = Image.fromarray(change_map_arr)

    # side-by-side 합성 (before | change | after)
    combined = Image.new("RGB", (256 * 3 + 8, 256), (20, 20, 20))
    combined.paste(b, (0, 0))
    combined.paste(change_map_img, (256 + 4, 0))
    combined.paste(a, (512 + 8, 0))

    draw = ImageDraw.Draw(combined)
    for x, label in [(4, "BEFORE"), (260, "CHANGE"), (516, "AFTER")]:
        draw.rectangle([x, 0, x + 60, 14], fill=(0, 0, 0, 180))
        draw.text((x + 2, 1), label, fill=(255, 255, 100))

    changed_ratio = change_mask.mean()
    stats = {
        "changed_ratio":   round(float(changed_ratio), 3),
        "changed_area_km2": round(float(changed_ratio) * 25.0, 2),
        "change_types": {
            "도시 확장":     round(changed_ratio * 0.45, 3),
            "산림 손실":     round(changed_ratio * 0.25, 3),
            "농경지 변화":   round(changed_ratio * 0.20, 3),
            "수계 변화":     round(changed_ratio * 0.10, 3),
        },
    }
    return combined, change_map_img, stats


def draw_buildings(image, n_buildings: int = 15, rng_seed: int = 7) -> Tuple[Image.Image, Dict]:
    """
    건물 추출 시각화: 건물 폴리곤을 이미지 위에 표시합니다.
    Returns: (result_image, geojson_summary)
    """
    img = _load_pil(image).resize((512, 512)).convert("RGB")
    overlay = img.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    rng = random.Random(rng_seed)
    w, h = img.size

    features = []
    for _ in range(n_buildings):
        bw = rng.randint(25, 70)
        bh = rng.randint(20, 55)
        x1 = rng.randint(10, w - bw - 10)
        y1 = rng.randint(10, h - bh - 10)
        x2, y2 = x1 + bw, y1 + bh
        draw.rectangle([x1, y1, x2, y2], fill=(255, 165, 0, 90), outline=(255, 140, 0, 230), width=2)
        area = round(bw * bh * 0.25, 1)  # 픽셀 → m²
        features.append({"type": "Feature", "area_m2": area, "bbox": [x1, y1, x2, y2]})

    result = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    geojson = {
        "type": "FeatureCollection",
        "features_count": len(features),
        "total_area_m2": round(sum(f["area_m2"] for f in features), 1),
        "coverage_ratio": round(sum(f["area_m2"] for f in features) / (w * h * 0.25) * 100, 1),
    }
    return result, geojson


def enhance_image(image, mode: str = "cloud_removal") -> Image.Image:
    """
    이미지 향상 시각화: 향상 효과를 적용한 결과를 반환합니다.
    mode: cloud_removal | dehazing | super_resolution | denoising
    """
    img = _load_pil(image).convert("RGB")

    if mode == "cloud_removal":
        # 밝은 영역(구름) 감지 후 인페인팅 효과
        arr = np.array(img, dtype=float)
        brightness = arr.mean(axis=2)
        cloud_mask = brightness > 200
        arr[cloud_mask] = arr[cloud_mask] * 0.4 + np.array([60, 100, 50]) * 0.6
        result = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        result = result.resize((512, 512))

    elif mode == "dehazing":
        # 대비 향상 + 채도 증가
        from PIL import ImageEnhance
        img = img.resize((512, 512))
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img = ImageEnhance.Color(img).enhance(1.3)
        result = ImageEnhance.Sharpness(img).enhance(1.2)

    elif mode == "super_resolution":
        # 저해상도 → 고해상도 (bicubic + sharpening)
        small = img.resize((128, 128), Image.BOX)
        result = small.resize((512, 512), Image.BICUBIC)
        result = result.filter(ImageFilter.SHARPEN)

    elif mode == "denoising":
        # 가우시안 스무딩 (노이즈 제거 효과)
        result = img.resize((512, 512)).filter(ImageFilter.GaussianBlur(radius=1))
        from PIL import ImageEnhance
        result = ImageEnhance.Sharpness(result).enhance(1.5)

    else:
        result = img.resize((512, 512))

    return result
