"""Hair cards, their colours, spring-bone chains and GLB files from character images with the released image model
(hairdeck.infer.image.Pipeline: preprocessing, DINOv3, image DiT, card VAE, colour model, chain DiT).

python image_to_hair.py IMAGE [IMAGE ...] --out OUT_DIR [--seed 0] [--n 1] [--steps 50] [--cfg 2.0] [--no-chains]
       [--no-colors] [--ckpt CKPT] [--chain_ckpt CKPT] [--color_ckpt CKPT]
Writes OUT_DIR/<image name>/ per image: input.png, prep.png (masks and crop box), model_input.png, and per variation
(seeds seed .. seed + n - 1) cards_seed<s>.json (with 16 colours along every card), tokens_seed<s>.npy,
chains_seed<s>.json, hair_seed<s>.glb (cards and head proxy), hair_seed<s>_nohead.glb, the *_chains.glb pair with the
chains drawn; info.json with timings and notes; OUT_DIR/summary.json. --no-colors: one hair colour (that of the input's
hair) instead of the colour model's. Exits with an error when an image fails.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hairdeck.infer.image import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('images', nargs='+')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n', type=int, default=1, help='variations per image (1..4), seeds seed, seed + 1, ...')
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--cfg', type=float, default=2.0)
    ap.add_argument('--no-chains', action='store_true')
    ap.add_argument('--no-colors', action='store_true', help='one hair colour instead of the colour model\'s')
    ap.add_argument('--ckpt', default=None, help='image DiT checkpoint (default: the released one under paths.ckpt())')
    ap.add_argument('--chain_ckpt', default=None, help='chain DiT checkpoint, one on the card VAE of --ckpt (default: the '
                    'released one under paths.ckpt())')
    ap.add_argument('--color_ckpt', default=None, help='colour model checkpoint, one on the card VAE of --ckpt (default: '
                    'the released one under paths.ckpt())')
    # top-5 fork: strands + ribbons from the cards on CUDA (hairdeck.strands); 0 = off (upstream behaviour)
    ap.add_argument('--strands-per-card', type=int, default=0, help='strands spread across each card (0: no strand output)')
    ap.add_argument('--strand-points', type=int, default=16, help='points per strand')
    ap.add_argument('--strand-width-mm', type=float, default=0.35, help='ribbon width of one strand in the GLB')
    ap.add_argument('--width-scale', type=float, default=1.0, help='scale every card width before the strands are spread')
    ap.add_argument('--offset-mm', type=float, default=0.0, help='move cards along their outward surface normal (lift / volume)')
    ap.add_argument('--curl-amp', type=float, default=0.0, help='curl radius as a fraction of the card width (0: none)')
    ap.add_argument('--curl-cycles', type=float, default=0.0, help='curl turns along the card length')
    a = ap.parse_args()
    names = [os.path.splitext(os.path.basename(f))[0] for f in a.images]
    keys = [os.path.normcase(n) for n in names]
    dup = sorted({n for n, key in zip(names, keys) if keys.count(key) > 1})
    if dup:  # each image writes OUT_DIR/<its name>/ and one entry of summary.json
        sys.exit(f'images with the same name {dup}: rename them or run them separately')
    P = Pipeline(ckpt=a.ckpt, chain_ckpt=a.chain_ckpt, chains=not a.no_chains, color_ckpt=a.color_ckpt,
                 colors=not a.no_colors)
    print(f'loaded in {P.load_sec} s' + ('' if P.chain is not None or a.no_chains else f'; chains off ({P.chain_error})')
          + ('' if P.color is not None or a.no_colors else f'; one hair colour ({P.color_error})'), flush=True)
    summary = {}
    for f in a.images:
        name = os.path.splitext(os.path.basename(f))[0]
        r = P.run(f, seed=a.seed, steps=a.steps, cfg=a.cfg, n_samples=a.n, out_dir=os.path.join(a.out, name),
                  chains=not a.no_chains, colors=not a.no_colors)
        summary[name] = dict(ok=r['ok'], message=r['message'], status=r['status'], notes=r['notes'],
                             n_cards=[s['n_cards'] for s in r['samples']], n_chains=[s['n_chains'] for s in r['samples']],
                             timings=r['timings'])
        print(name, 'ok' if r['ok'] else 'FAILED', summary[name]['n_cards'], r['message'], flush=True)
        if r['ok'] and a.strands_per_card > 0:
            summary[name]['strands'] = write_strands(os.path.join(a.out, name), [s['seed'] for s in r['samples']], a)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    if not all(s['ok'] for s in summary.values()):
        sys.exit(f'FAILED: {[n for n, s in summary.items() if not s["ok"]]}')


def write_strands(out_dir, seeds, a):
    """strands_seed<s>.npz + strands_seed<s>.glb next to cards_seed<s>.json (top-5 fork, hairdeck.strands on CUDA)."""
    import time

    import torch

    from hairdeck.strands import strands, write_glb, write_npz

    out = []
    for s in seeds:
        cards = json.load(open(os.path.join(out_dir, f'cards_seed{s}.json')))
        t = time.perf_counter()
        P, col = strands(cards, per_card=a.strands_per_card, points=a.strand_points, width_scale=a.width_scale,
                         offset_m=a.offset_mm / 1000.0, curl_amp=a.curl_amp, curl_cycles=a.curl_cycles, seed=s)
        torch.cuda.synchronize()
        gen_ms = 1000 * (time.perf_counter() - t)
        write_npz(os.path.join(out_dir, f'strands_seed{s}.npz'), P, col, per_card=a.strands_per_card, points=a.strand_points,
                  width_scale=a.width_scale, offset_mm=a.offset_mm, curl_amp=a.curl_amp, curl_cycles=a.curl_cycles)
        rep = write_glb(os.path.join(out_dir, f'strands_seed{s}.glb'), P, col, a.strand_width_mm / 1000.0)
        rep.update(seed=s, generate_ms=round(gen_ms, 1), total_ms=round(1000 * (time.perf_counter() - t), 1))
        out.append(rep)
        print(f'  strands seed {s}: {rep}', flush=True)
    return out


if __name__ == '__main__':
    main()
