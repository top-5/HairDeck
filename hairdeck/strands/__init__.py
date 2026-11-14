# Copyright 2026 top-5 (seed). Licensed under the Apache License, Version 2.0 (see LICENSE).
# Added in the top-5/HairDeck fork: not part of the upstream HairDeck release.
"""Cards -> strands -> ribbons on CUDA (torch), with deterministic card edits.

A HairDeck card is (n, 8): xyz, width, thickness, root width direction (hairdeck.cards.mesh). Strands are spread across
each card's width with rotation-minimising width frames (all cards at once), a low-frequency lateral wobble that grows from
the root, a per-strand tip trim, and arc-length resampling of a Catmull-Rom upsampled centre line; ribbons are quads along
each strand. Card edits applied before: width scale, an offset along the card's outward surface normal (lift / volume),
and a curl (helix around the centre line, zero at the root).

Measured on seeker (RTX PRO 6000, 117 cards): 180k strands x 12 points 24 ms, ribbons 2 ms (the numpy reference took
745 + 379 ms); CPU vs CUDA strand clouds differ by the random-seed noise floor (0.88 vs 0.85 mm p50).

  strands(cards, per_card=..., points=..., width_scale=1, offset_m=0, curl_amp=0, curl_cycles=0, seed=0) -> (P [S,K,3], colour [S,K,3])
  ribbons(P, colour, width_m) -> (vertices [2SK,3], faces [.,3], colours uint8)
  write_npz / write_glb"""
from __future__ import annotations

import math

import numpy as np
import torch


def pack(cards, device="cuda"):
    """list of {"seq": (n, 8), "colors": (16, 3)} -> seq [C, L, 8] (padded by repeating the tip), colours [C, 16, 3] 0..1."""
    L = max(len(c["seq"]) for c in cards)
    seq = np.zeros((len(cards), L, 8)); col = np.zeros((len(cards), 16, 3))
    for i, c in enumerate(cards):
        s = np.asarray(c["seq"], np.float64)
        seq[i, :len(s)] = s; seq[i, len(s):] = s[-1]
        cc = np.asarray(c.get("colors") if c.get("colors") is not None else np.tile([0.5, 0.35, 0.2], (16, 1)), np.float64)
        col[i] = (cc / 255.0 if cc.max() > 1.5 else cc)[:16]
    return torch.as_tensor(seq, device=device), torch.as_tensor(col, device=device)


def frames(pos, root_dir):
    """Unit tangents and rotation-minimising width directions [C, L, 3] for every card (L sequential steps over C)."""
    tang = torch.gradient(pos, dim=1)[0]
    tang = tang / tang.norm(dim=2, keepdim=True).clamp_min(1e-12)
    n = torch.empty_like(tang)
    d = root_dir - tang[:, 0] * (root_dir * tang[:, 0]).sum(1, keepdim=True)
    n[:, 0] = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
    for i in range(1, pos.shape[1]):
        a, b = tang[:, i - 1], tang[:, i]
        ax = torch.cross(a, b, dim=1); s = ax.norm(dim=1, keepdim=True); c = (a * b).sum(1, keepdim=True).clamp(-1, 1)
        k = ax / s.clamp_min(1e-12); v = n[:, i - 1]
        rot = v * c + torch.cross(k, v, dim=1) * s + k * (k * v).sum(1, keepdim=True) * (1 - c)
        v = torch.where(s > 1e-9, rot, v)
        v = v - b * (v * b).sum(1, keepdim=True)
        n[:, i] = v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return tang, n


def catmull_rom(x, k):
    """[N, L, D] -> [N, (L-1)*k+1, D], uniform Catmull-Rom through the points."""
    p = torch.cat([2 * x[:, :1] - x[:, 1:2], x, 2 * x[:, -1:] - x[:, -2:-1]], 1)
    s = torch.linspace(0, 1, k + 1, device=x.device, dtype=x.dtype)[:-1]
    s2, s3 = s * s, s * s * s
    b = torch.stack([-0.5 * s3 + s2 - 0.5 * s, 1.5 * s3 - 2.5 * s2 + 1, -1.5 * s3 + 2 * s2 + 0.5 * s, 0.5 * s3 - 0.5 * s2], 1)
    segs = torch.stack([p[:, i:i + p.shape[1] - 3] for i in range(4)], 2)
    out = torch.einsum("kj,cljd->clkd", b, segs).reshape(x.shape[0], -1, x.shape[2])
    return torch.cat([out, x[:, -1:]], 1)


@torch.no_grad()
def strands(cards, *, per_card=256, points=16, width_scale=1.0, offset_m=0.0, curl_amp=0.0, curl_cycles=0.0, wobble=0.12,
            wobble_freq=(0.4, 1.3), trim=0.12, brightness=0.06, seed=0, upsample=4, device="cuda"):
    """Strands of every card at once. `width_scale` scales the card widths; `offset_m` moves every card point along its
    outward surface normal (away from the cards' centroid: lift / volume); `curl_amp` (fraction of the card width) and
    `curl_cycles` (turns along the card) add a helix around the centre line that is zero at the root."""
    g = torch.Generator(device=device).manual_seed(int(seed))
    seq, col = pack(cards, device)
    C, L, _ = seq.shape
    pos, width = seq[..., :3].clone(), seq[..., 3] * float(width_scale)
    tang, n = frames(pos, seq[:, 0, 5:8])
    bn = torch.cross(tang, n, dim=2)                                               # the card's surface normal
    out_sign = torch.sign(((pos - pos[:, :1].mean(0, keepdim=True).mean(1, keepdim=True)) * bn).sum(2, keepdim=True).mean(1, keepdim=True))
    bn = bn * torch.where(out_sign == 0, torch.ones_like(out_sign), out_sign)
    tt = torch.linspace(0, 1, L, device=device, dtype=seq.dtype)[None]
    if offset_m:
        pos = pos + bn * float(offset_m) * tt[..., None].clamp(0.15, 1.0)
    if curl_amp and curl_cycles:
        ang = 2 * math.pi * float(curl_cycles) * tt
        r = float(curl_amp) * 0.5 * width * tt.clamp(0, 1) ** 0.8
        pos = pos + (n * torch.cos(ang)[..., None] + bn * torch.sin(ang)[..., None]) * r[..., None]
        tang, n = frames(pos, n[:, 0])
    S = C * int(per_card)
    ci = torch.arange(C, device=device).repeat_interleave(int(per_card))
    rnd = lambda *shape: torch.rand(*shape, device=device, generator=g, dtype=seq.dtype)   # noqa: E731
    u = ((torch.arange(int(per_card), device=device).repeat(C) + 0.1 + 0.8 * rnd(S)) / int(per_card)) * 2 - 1
    ph, fr = 2 * math.pi * rnd(S, 1), wobble_freq[0] + (wobble_freq[1] - wobble_freq[0]) * rnd(S, 1)
    wob = wobble * torch.sin(2 * math.pi * fr * tt + ph) * tt ** 0.7
    curves = pos[ci] + ((u[:, None] + wob) * 0.5 * width[ci])[..., None] * n[ci]
    dense = catmull_rom(curves, int(upsample))
    arc = torch.cat([torch.zeros(S, 1, device=device, dtype=seq.dtype), (dense[:, 1:] - dense[:, :-1]).norm(dim=2).cumsum(1)], 1)
    keep = 1.0 - trim * rnd(S, 1)
    target = torch.linspace(0, 1, int(points), device=device, dtype=seq.dtype)[None] * keep * arc[:, -1:]
    j = torch.searchsorted(arc.contiguous(), target.contiguous()).clamp(1, arc.shape[1] - 1)
    a0, a1 = arc.gather(1, j - 1), arc.gather(1, j)
    w = ((target - a0) / (a1 - a0).clamp_min(1e-12))[..., None]
    P = dense.gather(1, (j - 1)[..., None].expand(-1, -1, 3)) * (1 - w) + dense.gather(1, j[..., None].expand(-1, -1, 3)) * w
    tc = torch.linspace(0, 1, int(points), device=device, dtype=seq.dtype) * (col.shape[1] - 1)
    i0 = tc.floor().long().clamp(0, col.shape[1] - 2); fw = (tc - i0)[None, :, None]
    base = col[:, i0] * (1 - fw) + col[:, i0 + 1] * fw
    colour = (base[ci] * (1 + brightness * torch.randn(S, 1, 1, device=device, generator=g, dtype=seq.dtype))).clamp(0, 1)
    return P, colour


@torch.no_grad()
def ribbons(P, colour, width_m, taper=0.85):
    """Quads along every strand (one winding; render double-sided). width axis = tangent x world z (x where degenerate)."""
    S, K, _ = P.shape
    tan = torch.gradient(P, dim=1)[0]; tan = tan / tan.norm(dim=2, keepdim=True).clamp_min(1e-12)
    ref = torch.zeros_like(tan); ref[..., 2] = 1
    side = torch.cross(tan, ref, dim=2)
    side = torch.where(side.norm(dim=2, keepdim=True) < 1e-3, torch.tensor([1.0, 0, 0], device=P.device, dtype=P.dtype), side)
    side = side / side.norm(dim=2, keepdim=True).clamp_min(1e-12)
    t = (1 - taper * torch.linspace(0, 1, K, device=P.device, dtype=P.dtype) ** 1.5)[None, :, None]
    V = torch.stack([P - side * width_m * 0.5 * t, P + side * width_m * 0.5 * t], 2).reshape(-1, 3)
    idx = torch.arange(S * K * 2, device=P.device).reshape(S, K, 2)
    p0, p1, q0, q1 = idx[:, :-1, 0], idx[:, :-1, 1], idx[:, 1:, 0], idx[:, 1:, 1]
    F = torch.stack([torch.stack([p0, p1, q0], -1), torch.stack([p1, q1, q0], -1)], 2).reshape(-1, 3)
    C = (colour * 255).round().to(torch.uint8).repeat_interleave(2, dim=1).reshape(-1, 3)
    return V, F, C


def write_npz(path, P, colour, **meta):
    np.savez_compressed(path, points=P.float().cpu().numpy(), colour=(colour * 255).round().to(torch.uint8).cpu().numpy(),
                        meta=np.array(repr(meta)))


def write_glb(path, P, colour, width_m, head=None):
    """Ribbons (double-sided material, vertex colours) + optional head proxy (V, F) as one GLB."""
    import trimesh

    V, F, C = ribbons(P, colour, width_m)
    hair = trimesh.Trimesh(V.cpu().numpy(), F.cpu().numpy(), vertex_colors=np.c_[C.cpu().numpy(), np.full(len(C), 255, np.uint8)], process=False)
    sc = trimesh.Scene()
    sc.add_geometry(hair, geom_name="hair_ribbons")
    if head is not None:
        sc.add_geometry(trimesh.Trimesh(*head, process=False), geom_name="head_proxy")
    sc.export(path)
    return {"strands": int(P.shape[0]), "points": int(P.shape[1]), "triangles": int(F.shape[0])}


__all__ = ["pack", "frames", "catmull_rom", "strands", "ribbons", "write_npz", "write_glb"]
