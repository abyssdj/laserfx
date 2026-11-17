# ======================================================================== RAPID WIDE-COVERAGE BLACKOUT CHASING EFFECTS

def sc_blackout_zone_chase(c: Ctx) -> list[Path]:
    """Rapid 16th-note blackout chase jumping across 6 wide full-span spatial zones."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    sub = step_f - step
    # Hard blackout gap on the trailing 38% of every 16th-note step
    if sub > 0.62:
        return []
    pump = beat_pump(c)
    zone = step % 6
    col_a = c.palette(zone, v=1.0, white_boost=0.65 * pump)
    col_b = c.palette(zone + 1, v=1.0, white_boost=0.45 * pump)

    if zone == 0:    # Wide Upper Sky Deck
        return [
            line((-LIM, 0.78), (LIM, 0.62), col_a, dwell=0),
            line((LIM, 0.36), (-LIM, 0.48), col_b, dwell=0),
        ]
    elif zone == 1:  # Wide Lower Floor Deck
        return [
            line((-LIM, -0.78), (LIM, -0.62), col_a, dwell=0),
            line((LIM, -0.36), (-LIM, -0.48), col_b, dwell=0),
        ]
    elif zone == 2:  # Wide Left Wing Wall
        return [
            line((-0.82, -LIM), (-0.66, LIM), col_a, dwell=0),
            line((-0.38, LIM), (-0.50, -LIM), col_b, dwell=0),
        ]
    elif zone == 3:  # Wide Right Wing Wall
        return [
            line((0.82, -LIM), (0.66, LIM), col_a, dwell=0),
            line((0.38, LIM), (0.50, -LIM), col_b, dwell=0),
        ]
    elif zone == 4:  # Full-Span Diagonal Slash A
        return [
            line((-LIM, -0.84), (LIM, 0.84), col_a, dwell=0),
            line((-0.72, -LIM), (LIM, 0.56), col_b, dwell=0),
        ]
    else:            # Full-Span Diagonal Slash B
        return [
            line((-LIM, 0.84), (LIM, -0.84), col_a, dwell=0),
            line((-0.72, LIM), (LIM, -0.56), col_b, dwell=0),
        ]


def sc_quadrant_blackout_relay(c: Ctx) -> list[Path]:
    """High-speed 4-quadrant wide blackout relay (TL -> BR -> TR -> BL) with hard off-time."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    sub = step_f - step
    if sub > 0.58:
        return []
    pump = beat_pump(c)
    # Cross-corner hop order for maximum spatial contrast
    order = (0, 3, 1, 2)
    q = order[step % 4]
    sx = -1.0 if (q in (0, 2)) else 1.0
    sy = 1.0 if (q in (0, 1)) else -1.0
    col_a = c.palette(step, v=1.0, white_boost=0.70 * pump)
    col_b = c.palette(step + 2, v=1.0, white_boost=0.45 * pump)
    return [
        line((0.0, sy * 0.48), (sx * LIM, sy * 0.48), col_a, dwell=0),
        line((sx * 0.48, 0.0), (sx * 0.48, sy * LIM), col_b, dwell=0),
        line((0.0, 0.0), (sx * LIM, sy * LIM), col_a, dwell=0),
    ]


def sc_shutter_grid_chase(c: Ctx) -> list[Path]:
    """Rapid wide-coverage shutter chase stepping across 5 horizontal & 5 vertical lanes."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    sub = step_f - step
    if sub > 0.60:
        return []
    pump = beat_pump(c)
    lane = step % 5
    pos = -0.82 + (lane / 4.0) * 1.64
    col_a = c.palette(lane, v=1.0, white_boost=0.65 * pump)
    col_b = c.palette(lane + 1, v=1.0, white_boost=0.40 * pump)
    if (step // 5) % 2 == 0:
        y2 = float(np.clip(pos + (0.24 if pos < 0 else -0.24), -LIM, LIM))
        return [
            line((-LIM, pos), (LIM, pos), col_a, dwell=0),
            line((LIM, y2), (-LIM, y2), col_b, dwell=0),
        ]
    else:
        x2 = float(np.clip(pos + (0.24 if pos < 0 else -0.24), -LIM, LIM))
        return [
            line((pos, -LIM), (pos, LIM), col_a, dwell=0),
            line((x2, LIM), (x2, -LIM), col_b, dwell=0),
        ]


def sc_pingpong_blackout_whip(c: Ctx) -> list[Path]:
    """Wide full-span blackout whip snapping between extreme edges at 16th-note rate."""
    step_f = c.bt * 4.0
    step = int(math.floor(step_f))
    sub = step_f - step
    if sub > 0.56:
        return []
    pump = beat_pump(c)
    sign = 1.0 if (step % 2 == 0) else -1.0
    col_a = c.palette(step % 3, v=1.0, white_boost=0.72 * pump)
    col_b = c.palette((step + 1) % 3, v=1.0, white_boost=0.45 * pump)
    if (step // 4) % 2 == 0:
        y_outer = sign * 0.84
        y_inner = sign * 0.46
        return [
            line((-LIM, y_outer), (LIM, y_outer * 0.92), col_a, dwell=0),
            line((LIM, y_inner), (-LIM, y_inner * 1.06), col_b, dwell=0),
        ]
    else:
        x_outer = sign * 0.84
        x_inner = sign * 0.46
        return [
            line((x_outer, -LIM), (x_outer * 0.92, LIM), col_a, dwell=0),
            line((x_inner, LIM), (x_inner * 1.06, -LIM), col_b, dwell=0),
        ]


def sc_build_blackout_chase(c: Ctx) -> list[Path]:
    """Escalating wide-coverage blackout chase accelerating from 8ths to 16ths during Builds."""
    b = c.f.build
    rate = 2.0 if b < 0.5 else 4.0
    step_f = c.rt * rate
    step = int(math.floor(step_f))
    sub = step_f - step
    if sub > 0.60:
        return []
    zone = step % 4
    col = c.palette(zone, v=0.85 + 0.15 * b, white_boost=0.75 * (b ** 2))
    if zone == 0:
        return [line((-LIM, 0.78), (LIM, 0.78), col, dwell=0)]
    elif zone == 1:
        return [line((-LIM, -0.78), (LIM, -0.78), col, dwell=0)]
    elif zone == 2:
        return [line((-0.80, -LIM), (-0.80, LIM), col, dwell=0)]
    else:
        return [line((0.80, -LIM), (0.80, LIM), col, dwell=0)]
