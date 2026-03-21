"""
core/coordinate_transform.py — ROI 좌표 변환 시스템 (핵심 모듈)

═══════════════════════════════════════════════════════════════
설계 원칙
═══════════════════════════════════════════════════════════════
1. 원본 프레임 좌표계가 단일 진실 공급원(Single Source of Truth)이다.
2. ROI는 항상 원본 해상도 기준으로 저장한다.
3. 화면 표시 좌표는 매 렌더링마다 변환 행렬로 새로 계산한다.
4. 마우스 입력(화면 좌표)은 반드시 원본 좌표로 역변환 후 저장한다.
5. 레터박싱/필러박싱, 줌, 패닝을 모두 하나의 아핀 변환으로 처리한다.

───────────────────────────────────────────────────────────────
좌표 변환 공식  (keep_aspect_ratio=True 기준)
───────────────────────────────────────────────────────────────
  uniform_scale = min(display_w / src_w,  display_h / src_h)
  render_w      = src_w * uniform_scale
  render_h      = src_h * uniform_scale
  offset_x      = (display_w - render_w) / 2   ← 레터/필러 패딩
  offset_y      = (display_h - render_h) / 2

  원본 → 화면:  xd = xo * uniform_scale + offset_x
  화면 → 원본:  xo = (xd - offset_x) / uniform_scale

keep_aspect_ratio=False 시:
  scale_x = display_w / src_w
  scale_y = display_h / src_h
  offset_x = offset_y = 0

줌이 적용되면 uniform_scale *= zoom_factor 이고
offset은 뷰포트 중심을 기준으로 재계산된다.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple


# 원본/화면 점과 사각형의 타입 별칭
Point  = Tuple[float, float]         # (x, y)
Rect   = Tuple[float, float, float, float]   # (x, y, w, h)
IPoint = Tuple[int, int]
IRect  = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# CoordinateTransform  ← 불변 값 객체
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoordinateTransform:
    """
    원본 프레임과 화면 사이의 완전한 좌표 변환 정보를 담는 불변 값 객체.

    EngineCore/VideoCanvas 가 레이아웃이 변경될 때마다 새 인스턴스를 생성한다.
    ROIManager, OverlayRenderer, 마우스 이벤트 핸들러가 이 객체를 공유한다.
    """

    # 원본 프레임 크기
    source_width:  int = 1
    source_height: int = 1

    # 캔버스(디스플레이 영역) 크기
    display_width:  int = 1
    display_height: int = 1

    # 비율 유지 여부
    keep_aspect_ratio: bool = True

    # 줌 팩터 (1.0 = 원본 크기 기준 배율)
    zoom_factor: float = 1.0

    # 팬 오프셋 (소스 좌표계 기준, 추후 확장용)
    pan_x: float = 0.0
    pan_y: float = 0.0

    # ── 파생 필드 (init 후 계산) ────────────────────────────────────────
    # frozen dataclass이므로 __post_init__ 에서 object.__setattr__ 사용
    scale_x:       float = field(default=1.0, init=False)
    scale_y:       float = field(default=1.0, init=False)
    uniform_scale: float = field(default=1.0, init=False)
    offset_x:      float = field(default=0.0, init=False)
    offset_y:      float = field(default=0.0, init=False)
    # 실제 렌더링 영역 (레터박스 제외)
    render_width:  float = field(default=1.0, init=False)
    render_height: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        sw = max(self.source_width,  1)
        sh = max(self.source_height, 1)
        dw = max(self.display_width,  1)
        dh = max(self.display_height, 1)
        z  = max(self.zoom_factor, 0.01)

        if self.keep_aspect_ratio:
            base_scale = min(dw / sw, dh / sh)
            us = base_scale * z
            rw = sw * us
            rh = sh * us
            ox = (dw - rw) / 2.0
            oy = (dh - rh) / 2.0
            sx = us
            sy = us
        else:
            sx = (dw / sw) * z
            sy = (dh / sh) * z
            us = math.sqrt(sx * sy)   # 기하 평균
            rw = sw * sx
            rh = sh * sy
            ox = (dw - rw) / 2.0
            oy = (dh - rh) / 2.0

        # frozen 우회
        object.__setattr__(self, "scale_x",       sx)
        object.__setattr__(self, "scale_y",        sy)
        object.__setattr__(self, "uniform_scale",  us)
        object.__setattr__(self, "offset_x",       ox)
        object.__setattr__(self, "offset_y",       oy)
        object.__setattr__(self, "render_width",   rw)
        object.__setattr__(self, "render_height",  rh)

    # ── 점 변환 ────────────────────────────────────────────────────────

    def original_to_display_point(self, x: float, y: float) -> Point:
        """원본 픽셀 좌표 → 화면 좌표."""
        return (
            x * self.scale_x + self.offset_x,
            y * self.scale_y + self.offset_y,
        )

    def display_to_original_point(self, xd: float, yd: float) -> Point:
        """화면 좌표 → 원본 픽셀 좌표."""
        sx = max(self.scale_x, 1e-9)
        sy = max(self.scale_y, 1e-9)
        return (
            (xd - self.offset_x) / sx,
            (yd - self.offset_y) / sy,
        )

    def original_to_display_point_int(self, x: float, y: float) -> IPoint:
        p = self.original_to_display_point(x, y)
        return (round(p[0]), round(p[1]))

    def display_to_original_point_int(self, xd: float, yd: float) -> IPoint:
        p = self.display_to_original_point(xd, yd)
        return (round(p[0]), round(p[1]))

    # ── 사각형 변환 ────────────────────────────────────────────────────

    def original_to_display_rect(self, x: float, y: float,
                                  w: float, h: float) -> Rect:
        """원본 (x,y,w,h) → 화면 (x,y,w,h)."""
        x1, y1 = self.original_to_display_point(x, y)
        x2, y2 = self.original_to_display_point(x + w, y + h)
        return (x1, y1, x2 - x1, y2 - y1)

    def display_to_original_rect(self, xd: float, yd: float,
                                  wd: float, hd: float) -> Rect:
        """화면 (x,y,w,h) → 원본 (x,y,w,h)."""
        x1, y1 = self.display_to_original_point(xd, yd)
        x2, y2 = self.display_to_original_point(xd + wd, yd + hd)
        return (x1, y1, x2 - x1, y2 - y1)

    def original_to_display_rect_int(self, x: float, y: float,
                                      w: float, h: float) -> IRect:
        r = self.original_to_display_rect(x, y, w, h)
        return (round(r[0]), round(r[1]), round(r[2]), round(r[3]))

    def display_to_original_rect_int(self, xd: float, yd: float,
                                      wd: float, hd: float) -> IRect:
        r = self.display_to_original_rect(xd, yd, wd, hd)
        return (round(r[0]), round(r[1]), round(r[2]), round(r[3]))

    # ── 클램핑 헬퍼 ─────────────────────────────────────────────────────

    def clamp_to_source(self, x: float, y: float,
                         w: float, h: float) -> IRect:
        """원본 좌표 기준으로 프레임 경계 내로 클램핑."""
        x1 = max(0.0, min(x, self.source_width  - 1))
        y1 = max(0.0, min(y, self.source_height - 1))
        x2 = max(x1 + 1, min(x + w, self.source_width))
        y2 = max(y1 + 1, min(y + h, self.source_height))
        return (round(x1), round(y1), round(x2 - x1), round(y2 - y1))

    def is_point_in_render_area(self, xd: float, yd: float) -> bool:
        """화면 좌표가 실제 렌더링 영역(레터박스 제외) 안에 있는지 확인."""
        return (
            self.offset_x <= xd <= self.offset_x + self.render_width
            and self.offset_y <= yd <= self.offset_y + self.render_height
        )

    # ── 정규화 좌표 (0~1) ────────────────────────────────────────────────

    def to_normalized(self, x: float, y: float,
                       w: float, h: float) -> Tuple[float, float, float, float]:
        """원본 픽셀 좌표 → 정규화 좌표 (0~1)."""
        sw, sh = max(self.source_width, 1), max(self.source_height, 1)
        return (x / sw, y / sh, w / sw, h / sh)

    def from_normalized(self, nx: float, ny: float,
                         nw: float, nh: float) -> IRect:
        """정규화 좌표 (0~1) → 원본 픽셀 정수 좌표."""
        return (
            round(nx * self.source_width),
            round(ny * self.source_height),
            round(nw * self.source_width),
            round(nh * self.source_height),
        )

    def __repr__(self) -> str:
        return (
            f"CoordinateTransform("
            f"src={self.source_width}×{self.source_height} "
            f"disp={self.display_width}×{self.display_height} "
            f"scale={self.uniform_scale:.4f} "
            f"offset=({self.offset_x:.1f},{self.offset_y:.1f}) "
            f"zoom={self.zoom_factor:.2f})"
        )


# ---------------------------------------------------------------------------
# CoordinateTransformManager  ← 변환 객체 생성 및 캐시 관리
# ---------------------------------------------------------------------------

class CoordinateTransformManager:
    """
    현재 유효한 CoordinateTransform 을 보유하고, 레이아웃 변경 시
    새로운 인스턴스로 갱신한다.

    사용 패턴:
        mgr = CoordinateTransformManager()
        mgr.update(src_w=1920, src_h=1080, disp_w=800, disp_h=450)
        tf = mgr.transform   # 현재 변환 객체
        disp_pt = tf.original_to_display_point(960, 540)
    """

    def __init__(self, keep_aspect_ratio: bool = True):
        self._keep_aspect = keep_aspect_ratio
        self._zoom: float = 1.0
        self._transform = CoordinateTransform()   # 기본(1×1)

    def update(
        self,
        src_w: int,
        src_h: int,
        disp_w: int,
        disp_h: int,
        zoom_factor: Optional[float] = None,
    ) -> CoordinateTransform:
        """레이아웃 또는 소스 해상도가 바뀔 때마다 호출한다."""
        if zoom_factor is not None:
            self._zoom = max(0.1, zoom_factor)
        self._transform = CoordinateTransform(
            source_width=max(src_w, 1),
            source_height=max(src_h, 1),
            display_width=max(disp_w, 1),
            display_height=max(disp_h, 1),
            keep_aspect_ratio=self._keep_aspect,
            zoom_factor=self._zoom,
        )
        return self._transform

    def zoom_in(self, factor: float = 1.2) -> CoordinateTransform:
        tf = self._transform
        return self.update(tf.source_width, tf.source_height,
                           tf.display_width, tf.display_height,
                           zoom_factor=self._zoom * factor)

    def zoom_out(self, factor: float = 1.2) -> CoordinateTransform:
        return self.zoom_in(1.0 / factor)

    def zoom_reset(self) -> CoordinateTransform:
        tf = self._transform
        return self.update(tf.source_width, tf.source_height,
                           tf.display_width, tf.display_height,
                           zoom_factor=1.0)

    @property
    def transform(self) -> CoordinateTransform:
        return self._transform

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def keep_aspect_ratio(self) -> bool:
        return self._keep_aspect

    @keep_aspect_ratio.setter
    def keep_aspect_ratio(self, v: bool) -> None:
        self._keep_aspect = v
        tf = self._transform
        self.update(tf.source_width, tf.source_height,
                    tf.display_width, tf.display_height)
