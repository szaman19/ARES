"""Render the Home Inventory Ledger CONCEPT MOCKUP video (no real pipeline involved).

Frames are drawn with Pillow; voiceover uses macOS `say`; ffmpeg muxes the result.
All numbers are illustrative. Every frame carries a "CONCEPT MOCKUP" watermark.

Usage:  python poc/make_poc_video.py [out.mp4]
Needs:  pillow, and ffmpeg on PATH or the `imageio-ffmpeg` package.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 24
VOICE, RATE = "Samantha", 182

# ---------------------------------------------------------------- palette
BG = (14, 20, 36)
PANEL = (24, 32, 54)
CARD = (247, 248, 251)
INK = (20, 24, 36)
MUTED = (120, 128, 150)
WHITE = (240, 243, 250)
GREEN = (22, 163, 74)
AMBER = (217, 140, 20)
RED = (220, 50, 60)
BLUE = (59, 130, 246)
GREY = (150, 156, 170)

HN = "/System/Library/Fonts/HelveticaNeue.ttc"
_fc: dict = {}


def F(size: int, w: str = "reg") -> ImageFont.FreeTypeFont:
    idx = {"reg": 0, "bold": 1, "light": 7, "med": 10}[w]
    key = (size, idx)
    if key not in _fc:
        _fc[key] = ImageFont.truetype(HN, size, index=idx)
    return _fc[key]


def MONO(size: int) -> ImageFont.FreeTypeFont:
    key = ("mono", size)
    if key not in _fc:
        _fc[key] = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", size, index=0)
    return _fc[key]


# ---------------------------------------------------------------- helpers
def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def ease(x: float) -> float:
    x = clamp(x)
    return x * x * (3 - 2 * x)


def prog(t: float, start: float, dur: float = 0.5) -> float:
    return ease((t - start) / dur)


def lerp(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(lerp(a, b, k)) for a, b in zip(c1, c2))


def layer(base: Image.Image, alpha: float, draw_fn) -> None:
    """Draw onto a transparent layer and composite it with opacity `alpha`."""
    if alpha <= 0.01:
        return
    lay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(lay), lay)
    if alpha < 0.99:
        a = lay.getchannel("A").point(lambda v: int(v * alpha))
        lay.putalpha(a)
    base.alpha_composite(lay)


def text_c(d: ImageDraw.ImageDraw, cx: float, y: float, s: str, font, fill) -> None:
    w = d.textlength(s, font=font)
    d.text((cx - w / 2, y), s, font=font, fill=fill)


def badge(d, x, y, label, color, font=None):
    font = font or F(15, "bold")
    w = d.textlength(label, font=font) + 20
    d.rounded_rectangle((x, y, x + w, y + 26), 13, fill=color)
    d.text((x + 10, y + 4), label, font=font, fill=WHITE)
    return w


def flag_dot(d, cx, cy, r=10):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=RED)
    text_c(d, cx, cy - r + 1, "!", F(16, "bold"), WHITE)


def money(v: float) -> str:
    return f"${v:,.0f}"


def chrome(img: Image.Image, scene_title: str) -> None:
    d = ImageDraw.Draw(img)
    d.text((40, 26), "Home Inventory Ledger", font=F(20, "bold"), fill=WHITE)
    if scene_title:
        d.text((40 + d.textlength("Home Inventory Ledger", font=F(20, "bold")) + 18, 28),
               "·  " + scene_title, font=F(18, "light"), fill=GREY)
    wm = "CONCEPT MOCKUP · illustrative numbers"
    tw = d.textlength(wm, font=F(14, "med"))
    d.rounded_rectangle((W - tw - 60, 24, W - 36, 50), 13, outline=AMBER, width=2)
    d.text((W - tw - 48, 29), wm, font=F(14, "med"), fill=AMBER)


def caption(img: Image.Image, s: str, alpha: float = 1.0) -> None:
    def fn(d, _):
        w = d.textlength(s, font=F(22, "med"))
        d.rounded_rectangle((W / 2 - w / 2 - 22, H - 78, W / 2 + w / 2 + 22, H - 34), 22,
                            fill=(0, 0, 0, 170))
        d.text((W / 2 - w / 2, H - 70), s, font=F(22, "med"), fill=WHITE)
    layer(img, alpha, fn)


def new_frame() -> Image.Image:
    return Image.new("RGBA", (W, H), BG + (255,))


# ---------------------------------------------------------------- the desk "camera" scene
def build_desk() -> tuple[Image.Image, dict]:
    """Wide illustrated desk the virtual camera pans across. Returns image + item boxes."""
    DW, DH = 2000, 900
    im = Image.new("RGB", (DW, DH), (186, 196, 206))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, DW, 560), fill=(205, 212, 220))          # wall
    d.rectangle((0, 560, DW, DH), fill=(140, 104, 72))          # desk top
    d.rectangle((0, 560, DW, 575), fill=(110, 80, 55))
    boxes = {}
    # guitar leaning on wall (left)
    d.ellipse((90, 330, 330, 610), fill=(170, 110, 50), outline=(90, 55, 25), width=4)
    d.ellipse((120, 230, 300, 400), fill=(170, 110, 50), outline=(90, 55, 25), width=4)
    d.ellipse((180, 380, 240, 440), fill=(40, 25, 15))
    d.rectangle((198, 20, 222, 300), fill=(80, 50, 25))
    boxes["guitar"] = (80, 15, 340, 615)
    # monitor
    d.rectangle((520, 120, 1060, 440), fill=(25, 25, 30))
    d.rectangle((535, 135, 1045, 425), fill=(40, 70, 120))
    d.rectangle((760, 440, 820, 540), fill=(60, 60, 66))
    d.rectangle((700, 540, 880, 565), fill=(60, 60, 66))
    boxes["monitor"] = (510, 110, 1070, 570)
    # laptop
    d.polygon([(420, 820), (840, 820), (800, 700), (460, 700)], fill=(170, 172, 178))
    d.polygon([(470, 700), (790, 700), (770, 600), (490, 600)], fill=(120, 122, 130))
    boxes["laptop"] = (410, 590, 850, 830)
    # keyboard
    d.rounded_rectangle((900, 700, 1240, 780), 10, fill=(50, 52, 60))
    for r in range(3):
        for c in range(12):
            d.rectangle((915 + c * 27, 712 + r * 21, 935 + c * 27, 728 + r * 21), fill=(90, 92, 102))
    boxes["keyboard"] = (895, 695, 1245, 785)
    # headphones
    d.arc((1290, 470, 1470, 650), 180, 360, fill=(30, 30, 34), width=16)
    d.rounded_rectangle((1280, 560, 1330, 650), 18, fill=(30, 30, 34))
    d.rounded_rectangle((1430, 560, 1480, 650), 18, fill=(30, 30, 34))
    boxes["headphones"] = (1270, 460, 1490, 660)
    # phone
    d.rounded_rectangle((1300, 720, 1370, 850), 12, fill=(20, 20, 24))
    d.rounded_rectangle((1307, 730, 1363, 840), 8, fill=(60, 90, 140))
    boxes["phone"] = (1292, 712, 1378, 858)
    # camera + lens
    d.rounded_rectangle((1560, 600, 1800, 760), 18, fill=(28, 28, 32))
    d.rectangle((1600, 570, 1680, 605), fill=(28, 28, 32))
    d.ellipse((1620, 615, 1760, 755), fill=(15, 15, 18), outline=(70, 70, 76), width=6)
    d.ellipse((1655, 650, 1725, 720), fill=(40, 60, 90))
    boxes["camera"] = (1550, 560, 1810, 770)
    return im, boxes


DESK, DESK_BOXES = build_desk()

DETECT = [  # (key, label, appear_at_fraction_of_pan)
    ("guitar", "acoustic guitar", 0.04),
    ("monitor", "monitor · Dell 27\"", 0.18),
    ("laptop", "laptop · Apple (model ?)", 0.26),
    ("keyboard", "mech. keyboard", 0.40),
    ("headphones", "headphones · Bose", 0.58),
    ("phone", "phone · iPhone", 0.64),
    ("camera", "camera · Sony α7 IV", 0.82),
]
SAYS = [  # (fraction, narration bubble)
    (0.20, "“that's my MacBook, got it last year”"),
    (0.55, "“and my Sony headphones”"),
    (0.80, "“the camera's an a7 IV with the 24-70”"),
]


def camera_view(p: float, t: float, vw: int, vh: int) -> tuple[Image.Image, float, float, float]:
    """Crop of the desk at pan progress p (0..1) with a little hand shake."""
    scale = 0.72
    cw, ch = vw / scale, vh / scale
    x0 = lerp(0, DESK.width - cw, ease(p)) + 6 * math.sin(t * 3.1)
    y0 = (DESK.height - ch) / 2 + 5 * math.sin(t * 2.3)
    crop = DESK.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch))).resize((vw, vh))
    return crop, x0, y0, scale


# ---------------------------------------------------------------- ledger data (illustrative)
LIMIT = 8000
ROWS = [  # name, detail, status, price or (low, high), flag
    ["Camera", "Sony α7 IV · ILCE-7M4", "CERTIFIED", 2498, None],
    ["Lens", "Sony FE 24-70 GM II · SEL2470GM2", "CERTIFIED", 2298, None],
    ["Monitor", "Dell 27\" · U2723QE", "CERTIFIED", 579, None],
    ["Laptop", "Apple MacBook Pro · model not visible", "UNCERTIFIED", (900, 2600), None],
    ["Guitar", "acoustic · brand not visible", "UNCERTIFIED", (300, 1500), None],
    ["Phone", "Apple iPhone · generation unclear", "UNCERTIFIED", (500, 1200), None],
    ["Headphones", "Bose QuietComfort", "UNCERTIFIED", (200, 430), "narration said “Sony” · logo reads Bose"],
    ["Keyboard", "mechanical · unbranded", "UNCERTIFIED", (80, 220), None],
]


def totals(rows):
    cert = sum(r[3] for r in rows if r[2] == "CERTIFIED")
    lo = cert + sum(r[3][0] for r in rows if r[2] != "CERTIFIED")
    hi = cert + sum(r[3][1] for r in rows if r[2] != "CERTIFIED")
    return cert, lo, hi


def draw_ledger(img, t, rows, row_alpha_fn, highlight=None, flip_k=0.0, x=40, y=250, w=700):
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((x, y, x + w, y + 46 * len(rows) + 50), 16, fill=CARD)
    d.text((x + 20, y + 14), "ITEM", font=F(13, "bold"), fill=MUTED)
    d.text((x + w - 330, y + 14), "STATUS", font=F(13, "bold"), fill=MUTED)
    d.text((x + w - 150, y + 14), "REPLACEMENT", font=F(13, "bold"), fill=MUTED)
    for i, r in enumerate(rows):
        a = row_alpha_fn(i)
        if a <= 0:
            continue
        ry = y + 42 + i * 46

        def fn(dd, _, r=r, ry=ry, i=i):
            if highlight == i:
                dd.rounded_rectangle((x + 8, ry - 2, x + w - 8, ry + 42), 10,
                                     fill=mix(CARD, (255, 236, 179), 0.9))
            dd.text((x + 20, ry + 2), r[0], font=F(17, "bold"), fill=INK)
            dd.text((x + 20, ry + 23), r[1], font=F(13), fill=MUTED)
            st = r[2]
            col = GREEN if st == "CERTIFIED" else AMBER
            if highlight == i and flip_k > 0:
                col = mix(AMBER, GREEN, flip_k)
                st = "CERTIFIED" if flip_k > 0.5 else "UNCERTIFIED"
            badge(dd, x + w - 330, ry + 8, st, col, F(12, "bold"))
            price = money(r[3]) if isinstance(r[3], (int, float)) else f"{money(r[3][0])}–{money(r[3][1])}"
            dd.text((x + w - 150, ry + 10), price, font=F(16, "med"), fill=INK)
            if r[4]:
                flag_dot(dd, x + w - 330 + 124, ry + 21)
        layer(img, a, fn)


def verdict_card(img, x, y, w, verdict, color, conf, cert, lo, hi, hinges, alpha=1.0):
    def fn(d, _):
        d.rounded_rectangle((x, y, x + w, y + 400), 18, fill=CARD)
        d.text((x + 24, y + 20), "COVERAGE VERDICT", font=F(14, "bold"), fill=MUTED)
        d.text((x + 24, y + 46), verdict, font=F(34, "bold"), fill=color)
        d.text((x + 24, y + 92), f"Decision-1 confidence {conf:.2f}", font=F(16), fill=MUTED)
        # range bar
        bx0, bx1, by = x + 24, x + w - 24, y + 170
        vmin, vmax = 4000, 12000
        X = lambda v: bx0 + (v - vmin) / (vmax - vmin) * (bx1 - bx0)
        d.rounded_rectangle((bx0, by, bx1, by + 14), 7, fill=(225, 228, 236))
        d.rounded_rectangle((X(vmin), by, X(cert), by + 14), 7, fill=GREEN)
        d.rounded_rectangle((X(lo), by, X(hi), by + 14), 7, fill=AMBER)
        d.line((X(LIMIT), by - 16, X(LIMIT), by + 30), fill=INK, width=3)
        text_c(d, X(LIMIT), by - 38, f"limit {money(LIMIT)}", F(14, "bold"), INK)
        d.text((x + 24, y + 210), "Certified", font=F(14), fill=MUTED)
        d.text((x + 24, y + 230), money(cert), font=F(26, "bold"), fill=GREEN)
        d.text((x + 200, y + 210), "Estimated total", font=F(14), fill=MUTED)
        d.text((x + 200, y + 230), f"{money(lo)}–{money(hi)}", font=F(26, "bold"), fill=AMBER)
        d.text((x + 24, y + 290), "Verdict hinges on", font=F(14), fill=MUTED)
        hx = x + 24
        for h in hinges:
            hx += badge(d, hx, y + 314, h, BLUE, F(14, "bold")) + 8
        d.text((x + 24, y + 360), "Informational only · not a coverage determination", font=F(12), fill=MUTED)
    layer(img, alpha, fn)


# ---------------------------------------------------------------- scenes
def s_title(t, dur):
    img = new_frame()
    a = prog(t, 0.2, 0.8)
    def fn(d, _):
        text_c(d, W / 2, 230, "Home Inventory Ledger", F(64, "bold"), WHITE)
        text_c(d, W / 2, 320, "Film your things for 60 seconds.", F(30, "light"), GREY)
        text_c(d, W / 2, 362, "Know today whether your insurance covers them.", F(30, "light"), GREY)
    layer(img, a, fn)
    def fn2(d, _):
        text_c(d, W / 2, 470, "Gemini 4 Carbon  ·  Gemini 4 Hydrogen  ·  Gemini Decision-1  ·  shop.dev",
               F(18, "med"), BLUE)
    layer(img, prog(t, 1.4, 0.8), fn2)
    chrome(img, "")
    return img


def s_hook(t, dur):
    img = new_frame()
    def fn(d, _):
        text_c(d, W / 2, 210, "After the 2025 L.A. fires, families had to", F(36, "light"), WHITE)
        text_c(d, W / 2, 258, "rebuild their inventories from memory.", F(36, "light"), WHITE)
    layer(img, prog(t, 0.2, 0.8), fn)
    def fn2(d, _):
        text_c(d, W / 2, 360, "The ones who had filmed their homes beforehand had proof.", F(32, "bold"), AMBER)
    layer(img, prog(t, 3.5, 0.8), fn2)
    def fn3(d, _):
        text_c(d, W / 2, 470, "This is a before-the-loss product: the video is your proof.", F(20), GREY)
    layer(img, prog(t, 6.0, 0.8), fn3)
    chrome(img, "why")
    return img


def s_policy(t, dur):
    img = new_frame()
    d = ImageDraw.Draw(img)
    # phone frame with a document
    px, py = 170, 110
    d.rounded_rectangle((px, py, px + 300, py + 560), 36, fill=(10, 10, 14))
    d.rounded_rectangle((px + 14, py + 14, px + 286, py + 546), 26, fill=(60, 64, 76))
    tilt = lerp(-6, 0, prog(t, 0.3, 1.2))
    doc = Image.new("RGBA", (220, 300), (250, 248, 240, 255))
    dd = ImageDraw.Draw(doc)
    dd.text((14, 12), "SAMPLE POLICY", font=F(14, "bold"), fill=INK)
    dd.text((14, 32), "Declarations page", font=F(12), fill=MUTED)
    for i in range(9):
        dd.rectangle((14, 62 + i * 22, 14 + (170 if i % 3 else 120), 70 + i * 22), fill=(205, 205, 210))
    dd.rectangle((10, 150, 210, 178), outline=BLUE, width=3)
    dd.text((16, 156), "Coverage C   $8,000", font=F(13, "bold"), fill=INK)
    doc = doc.rotate(tilt, expand=True, fillcolor=(0, 0, 0, 0))
    img.alpha_composite(doc, (int(px + 150 - doc.width / 2), int(py + 280 - doc.height / 2)))
    # scanning line
    sk = prog(t, 1.2, 1.6)
    if 0 < sk < 1:
        yy = py + 130 + sk * 300
        d.line((px + 30, yy, px + 270, yy), fill=BLUE, width=3)
    # extraction card
    def fn(dd, _):
        x, y = 560, 210
        dd.rounded_rectangle((x, y, x + 560, y + 280), 18, fill=CARD)
        dd.text((x + 26, y + 22), "EXTRACTED BY GEMINI 4 CARBON", font=F(14, "bold"), fill=MUTED)
        dd.text((x + 26, y + 60), "Personal property (Coverage C)", font=F(20), fill=INK)
        dd.text((x + 26, y + 90), "$8,000", font=F(54, "bold"), fill=INK)
        dd.text((x + 26, y + 170), "Deductible  $500", font=F(20), fill=INK)
        badge(dd, x + 26, y + 220, "CERTIFIED · policy_limit_parse", GREEN, F(13, "bold"))
    layer(img, prog(t, 2.6, 0.7), fn)
    chrome(img, "1 · policy")
    caption(img, "Snap the declarations page (sample, never a real policy)", prog(t, 0.4, 0.5))
    return img


def s_capture(t, dur):
    img = new_frame()
    vx, vy, vw, vh = 60, 80, 820, 560
    p = clamp((t - 0.5) / (dur - 2.0))
    view, x0, y0, sc = camera_view(p, t, vw, vh)
    img.paste(view, (vx, vy))
    d = ImageDraw.Draw(img)
    d.rectangle((vx, vy, vx + vw, vy + vh), outline=(0, 0, 0), width=4)
    rx = vx + vw - 150
    d.rounded_rectangle((rx - 8, vy + 10, rx + 136, vy + 40), 8, fill=(0, 0, 0))
    if int(t * 2) % 2 == 0:
        d.ellipse((rx, vy + 17, rx + 16, vy + 33), fill=RED)
    d.text((rx + 24, vy + 13), f"REC 00:{int(t):02d}", font=MONO(18), fill=WHITE)
    # detections
    found = []
    for key, label, at in DETECT:
        if p < at:
            continue
        found.append(label)
        bx0, by0, bx1, by1 = DESK_BOXES[key]
        X0, Y0 = vx + (bx0 - x0) * sc, vy + (by0 - y0) * sc
        X1, Y1 = vx + (bx1 - x0) * sc, vy + (by1 - y0) * sc
        if X1 < vx or X0 > vx + vw:
            continue
        X0, X1 = max(X0, vx + 2), min(X1, vx + vw - 2)
        if X1 - X0 < 12:
            continue
        d.rectangle((X0, Y0, X1, Y1), outline=(80, 220, 140), width=3)
        tw = d.textlength(label, font=F(14, "bold"))
        d.rectangle((X0, Y0 - 22, X0 + tw + 12, Y0), fill=(80, 220, 140))
        d.text((X0 + 6, Y0 - 20), label, font=F(14, "bold"), fill=INK)
    # narration bubble
    for at, s in SAYS:
        if at <= p < at + 0.17:
            def fn(dd, _, s=s):
                tw = dd.textlength(s, font=F(20, "med"))
                dd.rounded_rectangle((vx + 20, vy + vh - 66, vx + 50 + tw, vy + vh - 20), 20,
                                     fill=(255, 255, 255, 230))
                dd.text((vx + 36, vy + vh - 56), s, font=F(20, "med"), fill=INK)
            layer(img, 1.0, fn)
    # right panel: parallel workers + stream
    x = 910
    d.text((x, 90), "HYDROGEN WORKERS (parallel)", font=F(14, "bold"), fill=GREY)
    for i in range(4):
        k = clamp((p * 4) - i * 0.6)
        d.rounded_rectangle((x, 120 + i * 34, x + 320, 144 + i * 34), 8, fill=PANEL)
        d.rounded_rectangle((x, 120 + i * 34, x + 320 * k, 144 + i * 34), 8, fill=BLUE)
        d.text((x + 10, 123 + i * 34), f"segment {i + 1}", font=F(14, "med"), fill=WHITE)
    d.text((x, 280), "ITEMS FOUND", font=F(14, "bold"), fill=GREY)
    for i, lab in enumerate(found):
        d.text((x, 308 + i * 30), "+ " + lab, font=F(18), fill=WHITE)
    chrome(img, "2 · capture")
    caption(img, "Film your desk and just talk", prog(t, 0.3, 0.5) * (1 - prog(t, 4.0, 0.5)))
    return img


def s_verdict(t, dur):
    img = new_frame()
    cert, lo, hi = totals(ROWS)
    verdict_card(img, 770, 90, 470, "Likely underinsured", AMBER, 0.66, cert, lo, hi,
                 ["Laptop", "Guitar", "Phone"], alpha=prog(t, 0.2, 0.6))
    draw_ledger(img, t, ROWS, lambda i: prog(t, 2.0 + i * 0.35, 0.4), y=90, w=700)
    # contradiction callout
    def fn(d, _):
        x, y = 770, 520
        d.rounded_rectangle((x, y, x + 470, y + 110), 16, fill=(60, 20, 26))
        flag_dot(d, x + 30, y + 25)
        d.text((x + 48, y + 16), "NARRATION CONFLICT", font=F(15, "bold"), fill=(255, 140, 150))
        d.text((x + 20, y + 44), "You said “Sony headphones” at 0:14,", font=F(17), fill=WHITE)
        d.text((x + 20, y + 70), "but the logo in frame reads Bose. Label value kept.", font=F(17), fill=WHITE)
    layer(img, prog(t, dur - 5.0, 0.6), fn)
    chrome(img, "3 · verdict")
    return img


def s_ask(t, dur):
    img = new_frame()
    t_flip = dur * 0.58
    k = prog(t, t_flip, 0.8)
    rows = [r[:] for r in ROWS]
    after = [r[:] for r in ROWS]
    after[3] = ["Laptop", "MacBook Pro 14\" · A2992 · listing matched", "CERTIFIED", 1999, None]
    rows_show = after if k > 0.5 else rows
    if k > 0.5:
        rows_show = [r[:] for r in after]
    else:
        rows_show = [r[:] for r in rows]
    cert0, lo0, hi0 = totals(ROWS)
    cert1, lo1, hi1 = totals(after)
    cert, lo, hi = (lerp(cert0, cert1, k), lerp(lo0, lo1, k), lerp(hi0, hi1, k))
    vtxt, vcol = ("Underinsured", RED) if k > 0.5 else ("Likely underinsured", AMBER)
    conf = lerp(0.66, 0.91, k)
    verdict_card(img, 770, 90, 470, vtxt, vcol, conf, cert, lo, hi, ["Guitar", "Phone", "Headphones"]
                 if k > 0.5 else ["Laptop", "Guitar", "Phone"])
    draw_ledger(img, t, rows_show, lambda i: 1.0, highlight=3, flip_k=k, y=90, w=700)
    # ask card / label close-up overlay
    def ask(d, _):
        x, y = 770, 520
        d.rounded_rectangle((x, y, x + 470, y + 110), 16, fill=(20, 40, 80))
        d.text((x + 20, y + 14), "BEST NEXT SHOT  (highest value of information)", font=F(13, "bold"),
               fill=(150, 190, 255))
        d.text((x + 20, y + 40), "Show me the label under the laptop", font=F(22, "bold"), fill=WHITE)
        d.text((x + 20, y + 74), "worth ±$850 of uncertainty in your verdict", font=F(16), fill=(190, 205, 230))
    layer(img, prog(t, 0.5, 0.5) * (1 - prog(t, t_flip + 1.2, 0.5)), ask)

    def closeup(d, lay):
        x, y = 160, 160
        d.rounded_rectangle((x, y, x + 460, y + 300), 20, fill=(150, 152, 160))
        d.rounded_rectangle((x + 60, y + 90, x + 400, y + 220), 8, fill=(210, 212, 218))
        d.text((x + 80, y + 104), "Designed by Apple in California", font=F(13), fill=(70, 70, 76))
        d.text((x + 80, y + 132), "Model A2992", font=MONO(26), fill=(30, 30, 34))
        d.text((x + 80, y + 172), "EMC 8102   Serial ••••••", font=MONO(14), fill=(70, 70, 76))
        d.rectangle((x + 74, y + 126, x + 290, y + 166), outline=(80, 220, 140), width=3)
        d.text((x + 20, y + 16), "follow-up clip · 0:05", font=F(15, "bold"), fill=WHITE)
    layer(img, prog(t, t_flip - 3.4, 0.4) * (1 - prog(t, t_flip + 0.6, 0.4)), closeup)

    def listing(d, _):
        x, y = 770, 520
        d.rounded_rectangle((x, y, x + 470, y + 110), 16, fill=(16, 60, 36))
        d.text((x + 20, y + 14), "CERTIFIED  ·  catalog_exact_match passed", font=F(13, "bold"),
               fill=(140, 240, 170))
        d.text((x + 20, y + 40), "MacBook Pro 14\" (A2992)  $1,999", font=F(22, "bold"), fill=WHITE)
        d.text((x + 20, y + 74), "shop.dev listing · retrieved 2 min ago", font=F(16), fill=(190, 230, 200))
    layer(img, prog(t, t_flip + 1.4, 0.5), listing)
    chrome(img, "4 · ask & refilm")
    return img


def s_proof(t, dur):
    img = new_frame()
    d = ImageDraw.Draw(img)
    text_c(d, W / 2, 100, "Proof: unseen clips, honest baseline", F(36, "bold"), WHITE)
    text_c(d, W / 2, 152, "Test clips filmed by people outside the team, labeled before any run",
           F(18), GREY)
    cols = ["System", "Invented prices", "Duplicate rows", "Contradictions caught"]
    rows = [["B0 · single Gemini call", "—", "—", "—"],
            ["B1 · single call + top-1 search", "—", "—", "—"],
            ["Ledger (no asks)", "—", "—", "—"],
            ["Ledger (+1 ask round)", "—", "—", "—"]]
    x0, y0, cw = 150, 220, [380, 220, 200, 200]
    def fn(dd, _):
        dd.rounded_rectangle((x0 - 20, y0 - 20, x0 + sum(cw) + 20, y0 + 60 * 5), 16, fill=CARD)
        cx = x0
        for c, w in zip(cols, cw):
            dd.text((cx, y0), c, font=F(16, "bold"), fill=MUTED)
            cx += w
        for i, r in enumerate(rows):
            cx = x0
            for j, (v, w) in enumerate(zip(r, cw)):
                dd.text((cx, y0 + 56 + i * 60), v, font=F(20, "bold" if j == 0 and i >= 2 else "reg"),
                        fill=INK)
                cx += w
    layer(img, prog(t, 0.4, 0.6), fn)
    def fn2(dd, _):
        s = "Results pending: filled in from the real eval run, never from this mockup"
        text_c(dd, W / 2, 560, s, F(20, "med"), AMBER)
    layer(img, prog(t, 2.5, 0.6), fn2)
    chrome(img, "5 · proof")
    return img


def s_close(t, dur):
    img = new_frame()
    def fn(d, _):
        x, y = 140, 130
        d.rounded_rectangle((x, y, x + 520, y + 420), 18, fill=CARD)
        d.text((x + 24, y + 22), "REBUILD MY HOME  ·  replacement cart via shop.dev", font=F(14, "bold"), fill=MUTED)
        items = [("Sony α7 IV", 2498), ("Sony FE 24-70 GM II", 2298), ("MacBook Pro 14\"", 1999),
                 ("Dell 27\" U2723QE", 579)]
        for i, (n, pr) in enumerate(items):
            d.text((x + 24, y + 70 + i * 52), n, font=F(20), fill=INK)
            pw = d.textlength(money(pr), font=F(20, "bold"))
            d.text((x + 496 - pw, y + 70 + i * 52), money(pr), font=F(20, "bold"), fill=INK)
        d.line((x + 24, y + 290, x + 496, y + 290), fill=(220, 222, 230), width=2)
        d.rounded_rectangle((x + 24, y + 320, x + 496, y + 380), 14, fill=(90, 49, 244))
        text_c(d, x + 260, y + 338, "Review & check out (you confirm)", F(20, "bold"), WHITE)
    layer(img, prog(t, 0.3, 0.6), fn)
    def fn2(d, _):
        d.text((740, 220), "Home Inventory Ledger", font=F(40, "bold"), fill=WHITE)
        d.text((740, 280), "Every price certified or labeled an estimate.", font=F(22), fill=GREY)
        d.text((740, 314), "Never an invented number.", font=F(22), fill=GREY)
        d.text((740, 380), "Film it before. Know today.", font=F(28, "bold"), fill=AMBER)
    layer(img, prog(t, 2.0, 0.8), fn2)
    chrome(img, "6 · after a loss")
    return img


SCENES = [
    (s_title, 6.0, "Home Inventory Ledger. Film your things for sixty seconds, and find out today "
                   "whether your insurance actually covers them."),
    (s_hook, 8.5, "After the L.A. fires, families had to rebuild their inventories from memory. "
                  "The ones who had filmed their homes beforehand had proof."),
    (s_policy, 7.0, "Start by snapping your policy's declarations page. Gemini Carbon reads your "
                    "personal property limit: eight thousand dollars."),
    (s_capture, 12.0, "Then film your desk and just talk. Parallel Gemini Hydrogen workers itemize "
                      "every segment of the video, and line up what you say with what the camera sees."),
    (s_verdict, 12.0, "You get a verdict, not just a list. Every price is either certified against a real "
                      "Shopify listing, or clearly marked as an estimate. And it caught that you called "
                      "your Bose headphones Sony."),
    (s_ask, 13.0, "Gemini Decision One works out which items the verdict hinges on, and asks for the one "
                  "shot worth the most. Film the label. It matches a live listing, the price is certified, "
                  "and the verdict firms up."),
    (s_proof, 9.0, "We test on clips filmed by people we've never met, against a single Gemini call, "
                   "measuring invented prices, double counts, and contradictions caught."),
    (s_close, 8.0, "And if the worst happens, the same ledger becomes your replacement cart. "
                   "Home Inventory Ledger. Film it before. Know today."),
]


# ---------------------------------------------------------------- audio + mux
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # type: ignore
    return imageio_ffmpeg.get_ffmpeg_exe()


def tts(text: str, path: Path) -> float:
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(path),
                    "--data-format=LEI16@22050", text], check=True)
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def main(out: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="poc_"))
    lead = 0.4
    durations, clips = [], []
    for i, (_, min_d, vo) in enumerate(SCENES):
        p = tmp / f"vo{i}.wav"
        vlen = tts(vo, p)
        durations.append(max(min_d, vlen + lead + 0.9))
        clips.append(p)

    # one continuous narration track, each VO starting `lead` s into its scene
    rate = 22050
    with wave.open(str(tmp / "narration.wav"), "wb") as outw:
        outw.setnchannels(1)
        outw.setsampwidth(2)
        outw.setframerate(rate)
        for p, dur in zip(clips, durations):
            with wave.open(str(p)) as w:
                data = w.readframes(w.getnframes())
            total = int(dur * rate)
            pre = int(lead * rate)
            body = data[: (total - pre) * 2]
            outw.writeframes(b"\x00\x00" * pre + body + b"\x00\x00" * (total - pre - len(body) // 2))

    total_frames = sum(int(round(d * FPS)) for d in durations)
    print(f"{len(SCENES)} scenes, {sum(durations):.1f}s, {total_frames} frames")
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(tmp / "narration.wav"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert proc.stdin
    for (fn, _, _), dur in zip(SCENES, durations):
        n = int(round(dur * FPS))
        for f in range(n):
            t = f / FPS
            frame = fn(t, dur)
            # fade in/out at scene edges
            edge = min(1.0, t / 0.3, (dur - t) / 0.3)
            if edge < 1.0:
                frame = Image.blend(Image.new("RGBA", (W, H), BG + (255,)), frame, max(0.0, edge))
            proc.stdin.write(frame.convert("RGB").tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "poc_concept.mp4"))
