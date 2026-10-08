"""Renders resources/sc_remask.gif (and .mp4): an animation of the Mask, Sample,
Revise sampler with SC-ReMask, styled after the architecture figure.

Toy simulation with the paper's settings: linear schedule (kappa_t = t),
K = 8 steps, eta_rescale = eta_cap = 0.5, t_switch = 0. At step k each masked
suffix token is sampled with probability dt / (1 - t_k) and each generated
suffix token is remasked with probability sigma(t_k); the prompt and EOS stay
pinned.

Usage: python resources/make_sc_remask_animation.py  (needs Pillow, matplotlib, gifski, ffmpeg)
"""

import io
import math
import os
import random
import shutil
import subprocess
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = "/usr/share/fonts/truetype/noto"

# Logical canvas (rendered at SS x and downsampled for anti-aliasing)
W, H, SS = 1280, 620, 3
FPS = 20

# Palette sampled from the architecture figure
BLACK = (20, 20, 20)
GRAY = (110, 110, 110)
LIGHT_GRAY = (217, 217, 217)
PURPLE = (139, 93, 173)
GREEN = (195, 226, 191)
GREEN_DARK = (76, 140, 70)
ORANGE = (255, 189, 89)
ORANGE_LIGHT = (255, 228, 186)
ORANGE_DARK = (204, 122, 0)
BLUE_LIGHT = (197, 219, 253)
BG = (255, 255, 255)

# Sampler settings (paper defaults)
K = 8
N_PROMPT, N_GEN = 3, 12
ETA_RESCALE, ETA_CAP, T_SWITCH = 0.5, 0.5, 0.0

# Layout (logical px)
BOX_W, BOX_H, GAP = 44, 30, 6
GRID_X, GRID_Y, ROW_PITCH = 96, 132, 40
N_TOK = N_PROMPT + N_GEN + 1


def font(size, bold=False):
    name = "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), int(size * SS))


def sigma_at(k):
    t, t_next = k / K, (k + 1) / K
    sigma_max = 1.0 if t == 0 else min(1.0, (1.0 - t_next) / t)
    sigma = ETA_RESCALE * min(ETA_CAP, sigma_max)
    return 0.0 if t < T_SWITCH else sigma


def simulate(seed):
    """Returns per-step states; each token is (kind, label, event)."""
    rng = random.Random(seed)
    gen_count = [0] * N_GEN
    state = [("mask", "M", None)] * N_GEN
    rows, stats = [list(state)], []
    for k in range(K):
        p_unmask = 1.0 / (K - k)
        sigma = sigma_at(k)
        new, n_sample, n_remask = [], 0, 0
        for i, (kind, label, _) in enumerate(state):
            if kind == "mask":
                if rng.random() < p_unmask:
                    gen_count[i] += 1
                    label = f"S{N_PROMPT + 1 + i}" + "′" * (gen_count[i] - 1)
                    new.append(("tok", label, "sampled"))
                    n_sample += 1
                else:
                    new.append(("mask", "M", None))
            elif rng.random() < sigma:
                new.append(("mask", "M", "remasked"))
                n_remask += 1
            else:
                new.append((kind, label, None))
        state = new
        rows.append(list(state))
        stats.append((n_sample, n_remask, sigma))
    return rows, stats


def pick_seed():
    """A run that shows several remasks, including late ones that get re-sampled."""
    for seed in range(10_000):
        rows, stats = simulate(seed)
        remasks = [s[1] for s in stats]
        regenerated = sum("′" in lab for _, lab, _ in rows[-1])
        if 4 <= sum(remasks) <= 6 and regenerated >= 3 and sum(remasks[3:]) >= 2 and stats[0][0] >= 2:
            return seed
    raise RuntimeError("no suitable seed")


def dashed_rect(d, box, color, width, dash=5, gap=4):
    x0, y0, x1, y1 = box
    dash, gap = dash * SS, gap * SS
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = math.hypot(bx - ax, by - ay)
        n = 0.0
        while n < length:
            e = min(n + dash, length)
            d.line(
                (ax + (bx - ax) * n / length, ay + (by - ay) * n / length,
                 ax + (bx - ax) * e / length, ay + (by - ay) * e / length),
                fill=color, width=width,
            )
            n += dash + gap


def blend(c1, c2, a):
    return tuple(int(round(x + (y - x) * a)) for x, y in zip(c1, c2))


def s(v):
    return int(round(v * SS))


def draw_token(d, x, y, kind, label, event, alpha=1.0):
    box = (s(x), s(y), s(x + BOX_W), s(y + BOX_H))
    fill, outline, text_color, dashed = BG, BLACK, BLACK, False
    if kind == "prompt":
        outline = text_color = PURPLE
    elif kind == "eos":
        fill = LIGHT_GRAY
    elif event == "sampled":
        fill = GREEN
    elif event == "remasked":
        fill, dashed = ORANGE_LIGHT, True
    fill, outline, text_color = (blend(BG, c, alpha) for c in (fill, outline, text_color))
    d.rectangle(box, fill=fill)
    if dashed:
        dashed_rect(d, box, outline, s(1.6))
    else:
        d.rectangle(box, outline=outline, width=s(1.6))
    f = font(11 if kind == "eos" else 13, bold=True)
    d.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, font=f, fill=text_color, anchor="mm")


def token_x(j):
    return GRID_X + j * (BOX_W + GAP)


def draw_row(d, row, y, alpha=1.0):
    for j in range(N_PROMPT):
        draw_token(d, token_x(j), y, "prompt", f"S{j + 1}", None, alpha)
    for i, (kind, label, event) in enumerate(row):
        draw_token(d, token_x(N_PROMPT + i), y, kind, label, event, alpha)
    draw_token(d, token_x(N_TOK - 1), y, "eos", "EOS", None, alpha)


def mathtext(line, fontsize, color="#141414"):
    plt.rcParams["mathtext.fontset"] = "cm"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, line, fontsize=fontsize, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=72 * SS, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGBA")


def render_math():
    """Equations rendered once with matplotlib mathtext."""
    lines = [
        r"$\sigma_{\max}(t_k)=\min\left(1,\ \dfrac{1-\kappa_{t_{k+1}}}{\kappa_{t_k}}\right)$",
        r"$\sigma(t_k)=\eta_{\mathrm{rescale}}\,\min\left(\eta_{\mathrm{cap}},\ \sigma_{\max}(t_k)\right)$",
        r"$r_{\mathrm{rm}}(t_k)=-\log\left(1-\sigma(t_k)\right)/\Delta t$",
    ]
    images = [mathtext(line, 15) for line in lines]
    extra = {
        "cap": mathtext(r"$\eta_{\mathrm{rescale}}\,\eta_{\mathrm{cap}}$", 13, "#6e6e6e"),
        "params": mathtext(
            rf"$\eta_{{\mathrm{{rescale}}}}=\eta_{{\mathrm{{cap}}}}={ETA_RESCALE},\ \ "
            rf"t_{{\mathrm{{switch}}}}={T_SWITCH:g},\ \ K={K}$", 14, "#6e6e6e"),
        "tk": mathtext(r"$t_k$", 12, "#6e6e6e"),
    }
    return images, extra


MATH, MATH_EXTRA = None, None


def draw_static(img, d):
    d.text((s(40), s(26)), "SC-ReMask: Schedule-Constrained CTMC Remasking", font=font(25, True), fill=BLACK)
    d.text(
        (s(40), s(62)),
        "Mask, Sample, Revise: generated tokens can return to [M] and be re-sampled in later steps",
        font=font(15), fill=GRAY,
    )
    # Column group labels
    y = GRID_Y - 24
    def group(j0, j1, text, color):
        x0, x1 = token_x(j0), token_x(j1) + BOX_W
        d.line((s(x0), s(y + 16), s(x1), s(y + 16)), fill=color, width=s(1.2))
        d.text((s((x0 + x1) / 2), s(y + 6)), text, font=font(12, True), fill=color, anchor="mm")
    group(0, N_PROMPT - 1, "prompt (pinned)", PURPLE)
    group(N_PROMPT, N_TOK - 2, "target suffix (infilled in parallel)", BLACK)
    group(N_TOK - 1, N_TOK - 1, "pinned", GRAY)

    # Right panel: schedule plot frame and equations
    px0, py0, px1, py1 = 930, 132, 1240, 330
    d.text((s(px0), s(py0 - 26)), "Remask probability per step", font=font(13, True), fill=BLACK)
    d.line((s(px0), s(py1), s(px1), s(py1)), fill=BLACK, width=s(1.2))
    d.line((s(px0), s(py0), s(px0), s(py1)), fill=BLACK, width=s(1.2))
    for v in (0.0, 0.1, 0.2, 0.3):
        yy = py1 - (py1 - py0) * v / 0.3
        d.line((s(px0 - 4), s(yy), s(px0), s(yy)), fill=BLACK, width=s(1))
        d.text((s(px0 - 8), s(yy)), f"{v:.1f}", font=font(10), fill=GRAY, anchor="rm")
    cap_y = py1 - (py1 - py0) * (ETA_RESCALE * ETA_CAP) / 0.3
    dashed_rect(d, (s(px0), s(cap_y), s(px1), s(cap_y)), GRAY, s(1), dash=4, gap=4)
    cap = MATH_EXTRA["cap"]
    img.paste(cap, (s(px1) - cap.width, s(cap_y - 4) - cap.height), cap)
    tk = MATH_EXTRA["tk"]
    img.paste(tk, (s(px1 + 8), s(py1) - tk.height // 2), tk)
    for k in (0, K // 2, K):
        x = px0 + (px1 - px0) * k / K
        d.text((s(x), s(py1 + 16)), f"{k / K:g}", font=font(10), fill=GRAY, anchor="mm")

    y = 372
    for m in MATH:
        img.paste(m, (s(930), s(y)), m)
        y += m.height / SS + 12
    params = MATH_EXTRA["params"]
    img.paste(params, (s(930), s(y + 2)), params)
    d.text((s(930), s(y + 26)), "Remasking is added to the tau-leaping hazard;", font=font(11), fill=GRAY)
    d.text((s(930), s(y + 42)), "prompt and EOS are never remasked.", font=font(11), fill=GRAY)

    # Legend
    ly = 566
    items = [
        ("prompt", "S", None, "prompt token"),
        ("mask", "M", None, "masked"),
        ("tok", "S", "sampled", "sampled this step"),
        ("mask", "M", "remasked", "remasked (SC-ReMask)"),
        ("tok", "S′", None, "re-sampled after remask"),
    ]
    x = 96
    for kind, label, event, text in items:
        draw_token(d, x, ly, kind, label, event)
        d.text((s(x + BOX_W + 8), s(ly + BOX_H / 2)), text, font=font(12), fill=BLACK, anchor="lm")
        x += BOX_W + 8 + d.textlength(text, font=font(12)) / SS + 26


def draw_schedule(d, active_k):
    px0, py0, px1, py1 = 930, 132, 1240, 330
    bw = (px1 - px0) / K
    for k in range(K):
        sg = sigma_at(k)
        x0 = px0 + k * bw + 3
        x1 = px0 + (k + 1) * bw - 3
        yt = py1 - (py1 - py0) * sg / 0.3
        color = ORANGE if k == active_k else (ORANGE_LIGHT if active_k is None or k < active_k else (240, 240, 240))
        if sg > 0:
            d.rectangle((s(x0), s(yt), s(x1), s(py1 - 1)), fill=color, outline=BLACK if k == active_k else None,
                        width=s(1.2))
        if k == active_k:
            d.text((s((x0 + x1) / 2), s(yt - 10)), f"{sg:.2f}", font=font(11, True), fill=ORANGE_DARK,
                   anchor="mm")


def render_frame(rows, stats, n_rows, new_row_alpha=None, active_k=None, status=None, highlight=None):
    img = Image.new("RGB", (s(W), s(H)), BG)
    d = ImageDraw.Draw(img)
    draw_static(img, d)
    draw_schedule(d, active_k)

    for r in range(n_rows):
        y = GRID_Y + r * ROW_PITCH
        alpha = new_row_alpha if (new_row_alpha is not None and r == n_rows - 1) else 1.0
        d.text((s(GRID_X - 12), s(y + BOX_H / 2)), f"k={r}", font=font(12, True),
               fill=blend(BG, GRAY, alpha), anchor="rm")
        draw_row(d, rows[r], y if alpha == 1.0 else y - (1 - alpha) * 12, alpha)

    # Pulse the tokens that are about to change in the current row
    if highlight:
        r, positions, a = highlight
        y = GRID_Y + r * ROW_PITCH
        for i, kind in positions:
            x = token_x(N_PROMPT + i)
            color = GREEN_DARK if kind == "sample" else ORANGE_DARK
            pad = 3
            d.rectangle((s(x - pad), s(y - pad), s(x + BOX_W + pad), s(y + BOX_H + pad)),
                        outline=blend(BG, color, a), width=s(2.4))

    if status:
        d.text((s(96), s(GRID_Y + (K + 1) * ROW_PITCH + 10)), status, font=font(14, True), fill=BLACK)

    return img.resize((W, H), Image.LANCZOS)


def main():
    global MATH
    global MATH_EXTRA
    MATH, MATH_EXTRA = render_math()
    seed = pick_seed()
    rows, stats = simulate(seed)
    print(f"seed={seed} stats={stats}")

    frames = []
    hold = lambda img, n: frames.extend([img] * n)

    intro = "x₀: prompt codes + [M] everywhere else; EOS pinned at the end"
    hold(render_frame(rows, stats, 1, status=intro), 30)

    for k in range(K):
        n_sample, n_remask, sigma = stats[k]
        t = k / K
        status = (f"Step {k + 1}/{K}   t = {t:.3f}   σ(t) = {sigma:.2f}   "
                  f"sampled {n_sample}   remasked {n_remask}")
        changes = [(i, "sample" if ev == "sampled" else "remask")
                   for i, (_, _, ev) in enumerate(rows[k + 1]) if ev]
        for f in range(10):
            a = 0.5 + 0.5 * math.sin(math.pi * f / 5)
            frames.append(render_frame(rows, stats, k + 1, active_k=k, status=status, highlight=(k, changes, a)))
        for f in range(1, 9):
            frames.append(render_frame(rows, stats, k + 2, new_row_alpha=f / 8, active_k=k, status=status))
        hold(render_frame(rows, stats, k + 2, active_k=k, status=status), 22 if n_remask else 14)

    outro = "All suffix tokens decoded: the codes are decoded by NeuCodec into a 24 kHz waveform"
    hold(render_frame(rows, stats, K + 1, status=outro), 70)

    tmp = tempfile.mkdtemp()
    try:
        for i, fr in enumerate(frames):
            fr.save(os.path.join(tmp, f"{i:04d}.png"))
        pngs = sorted(os.path.join(tmp, p) for p in os.listdir(tmp))
        gif = os.path.join(HERE, "sc_remask.gif")
        subprocess.run(["gifski", "--fps", str(FPS), "--quality", "95", "-o", gif, *pngs], check=True,
                       stdout=subprocess.DEVNULL)
        mp4 = os.path.join(HERE, "sc_remask.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(tmp, "%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", mp4],
            check=True,
        )
        frames[-1].save(os.path.join(HERE, "sc_remask_final.png"))
    finally:
        shutil.rmtree(tmp)
    print(f"{len(frames)} frames, {len(frames) / FPS:.1f}s -> {gif}, {mp4}")


if __name__ == "__main__":
    main()
