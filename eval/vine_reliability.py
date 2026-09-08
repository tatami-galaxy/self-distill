"""Compare independent halves of cached Vine MC draws; no model loading.

uv run python -m eval.vine_reliability \
  --run-dir results/advantage_comparison/Qwen3-1.7B --k 16
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from eval.advantage_comparison import fingerprint, keyed_seed, prefix_key, read_json, write_json


def split_counts(cache, k, seed):
    """Use the first k canonical draws, split by sample index, preserving resume semantics."""
    if k < 2 or k % 2:
        raise ValueError('k must be even and at least 2')
    prefix = cache['prefix']
    expected_key = fingerprint([prefix['question_id'], prefix['student_prompt_ids'], prefix['prefix_ids']])
    if prefix['key'] != expected_key or len(cache['draws']) < k:
        raise ValueError('MC prefix identity mismatch or insufficient draws')
    draws = cache['draws'][:k]
    for i, draw in enumerate(draws):
        if (draw['sample_idx'] != i or draw['seed'] != keyed_seed(seed, 'mc', prefix['key'], i)
                or draw['reward'] not in (0, 1)):
            raise ValueError('MC draws have invalid ordering, seeds, or rewards')
    half = k // 2
    return {
        'a': {'successes': sum(d['reward'] for d in draws[:half]), 'samples': half},
        'b': {'successes': sum(d['reward'] for d in draws[half:]), 'samples': half},
        'full': {'successes': sum(d['reward'] for d in draws), 'samples': k},
        'draw_fingerprint': fingerprint(draws),
    }


def advantage_pair(before, after, reward):
    """Terminal reward is observed once and shared; interior values use separate draw halves."""
    return tuple((reward if after is None else after[h]['successes'] / after[h]['samples'])
                 - before[h]['successes'] / before[h]['samples'] for h in ('a', 'b'))


def load_pairs(root, k):
    if k < 2 or k % 2:
        raise ValueError('k must be positive, even, and at least 2')
    source = read_json(root / f'k-{k}/summary.json')
    manifest = read_json(root / 'manifest.json')
    rollouts = read_json(root / 'rollouts.json')
    if source['mc_samples'] != k or source['rollout_fingerprint'] != fingerprint(rollouts):
        raise ValueError('Summary does not match requested K or canonical rollouts')
    originals = {r['rollout_id']: r for r in rollouts['rows']}
    if len(originals) != source['num_rollouts']:
        raise ValueError('Rollout count mismatch')
    seed = manifest['config']['seed']
    caches, endpoint_keys, pairs, seen = {}, {}, [], set()

    def get_counts(key, original, end):
        endpoint = (original['rollout_id'], end)
        if endpoint not in endpoint_keys:
            endpoint_keys[endpoint] = prefix_key(original, end)
        if key != endpoint_keys[endpoint]:
            raise ValueError('Comparison prefix does not match its canonical rollout endpoint')
        if key not in caches:
            cache = read_json(root / 'mc' / f'{key}.json')
            if (cache['prefix']['key'] != key or cache['prefix']['final_answer'] != original['final_answer']):
                raise ValueError('MC cache provenance mismatch')
            caches[key] = split_counts(cache, k, seed) | {'question_id': original['question_id']}
        return caches[key]

    with (root / f'k-{k}/comparisons.jsonl').open() as file:
        for line in file:
            row = json.loads(line)
            original = originals[row['rollout_id']]
            start, end = row['start'], row['end']
            identity = (row['rollout_id'], row['selection'], start, end)
            if (identity in seen or row['selection'] not in ('uniform', 'steps')
                    or not 0 <= start < end <= len(original['completion_ids'])
                    or row['token_ids'] != original['completion_ids'][start:end]
                    or row['question_id'] != original['question_id'] or row['reward'] != original['reward']
                    or (row['selection'] == 'uniform' and end != start + 1)):
                raise ValueError('Duplicate or misaligned comparison')
            seen.add(identity)
            vine = row['vine']
            terminal = end == len(original['completion_ids'])
            if terminal != (vine['after_key'] is None):
                raise ValueError('Incorrect terminal prefix in comparison')
            before = get_counts(vine['before_key'], original, start)
            after = None if terminal else get_counts(vine['after_key'], original, end)
            if (vine['before_counts'] != before['full']
                    or vine['after_counts'] != (None if terminal else after['full'])):
                raise ValueError('Saved comparisons and current MC draws disagree')
            a, b = advantage_pair(before, after, original['reward'])
            if not np.isclose((a + b) / 2, vine['advantage'], atol=1e-12, rtol=0):
                raise ValueError('Split-half average does not recover saved full-K advantage')
            pairs.append({'question_id': row['question_id'], 'rollout_id': row['rollout_id'],
                          'selection': row['selection'], 'start': start, 'end': end,
                          'terminal': terminal, 'before_key': vine['before_key'],
                          'after_key': vine['after_key'], 'a': a, 'b': b, 'full': (a + b) / 2})
    expected = {r['selection']: r['num_observations'] for r in source['comparisons']
                if r['unit'] == 'segment_mean' or r['selection'] == 'uniform'}
    if dict(Counter(r['selection'] for r in pairs)) != expected or len(caches) != source['num_prefixes']:
        raise ValueError('Incomplete comparison or prefix coverage')
    prefixes = [{'question_id': c['question_id'],
                 'a': c['a']['successes'] / c['a']['samples'],
                 'b': c['b']['successes'] / c['b']['samples']} for c in caches.values()]
    metadata = {'run_dir': str(root.resolve()), 'student': manifest['config']['student'], 'k': k,
                'half_k': k // 2, 'sample_indices_a': list(range(k // 2)),
                'sample_indices_b': list(range(k // 2, k)),
                'rollout_fingerprint': source['rollout_fingerprint'],
                'mc_draws_fingerprint': fingerprint(sorted((key, c['draw_fingerprint']) for key, c in caches.items()))}
    return pairs, prefixes, metadata


def metrics(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not len(a):
        return {'num_observations': 0}
    varying = len(a) > 1 and np.ptp(a) > 0 and np.ptp(b) > 0
    nonzero = (a != 0) & (b != 0)
    return {'num_observations': len(a), 'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'pearson': float(np.corrcoef(a, b)[0, 1]) if varying else None,
            'spearman': float(spearmanr(a, b).statistic) if varying else None,
            'rmse': float(np.sqrt(np.mean((a - b) ** 2))),
            'zero_fraction_a': float(np.mean(a == 0)), 'zero_fraction_b': float(np.mean(b == 0)),
            'both_zero_fraction': float(np.mean((a == 0) & (b == 0))),
            'sign_agreement': float(np.mean(np.sign(a) == np.sign(b))),
            'num_both_nonzero': int(nonzero.sum()),
            'sign_agreement_both_nonzero': float(np.mean(np.sign(a[nonzero]) == np.sign(b[nonzero]))) if nonzero.any() else None}


def summarize(points, bootstrap_samples, seed):
    a = np.array([r['a'] for r in points])
    b = np.array([r['b'] for r in points])
    result = metrics(a, b)
    clusters = defaultdict(list)
    for i, row in enumerate(points):
        clusters[row['question_id']].append(i)
    result['num_questions'] = len(clusters)
    intervals = defaultdict(list)
    rng = np.random.default_rng(seed)
    groups = [np.array(clusters[q], dtype=int) for q in sorted(clusters)]
    if len(groups) >= 2:
        for _ in range(bootstrap_samples):
            indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            sample = metrics(a[indices], b[indices])
            for name in ('pearson', 'spearman', 'rmse', 'sign_agreement_both_nonzero'):
                if sample[name] is not None:
                    intervals[name].append(sample[name])
    result['question_bootstrap_ci95'] = {name: np.quantile(values, [.025, .975]).tolist() for name, values in intervals.items()}
    result['valid_bootstrap_samples'] = {name: len(values) for name, values in intervals.items()}
    return result


def analyze(pairs, prefixes, metadata, bootstrap_samples=1000, seed=42):
    groups = {'prefix_values': prefixes}
    for selection in sorted({r['selection'] for r in pairs}):
        selected = [r for r in pairs if r['selection'] == selection]
        for subset in ('all', 'nonterminal', 'terminal'):
            groups[f'{selection}_{subset}'] = [r for r in selected if subset == 'all' or r['terminal'] == (subset == 'terminal')]
    summary = metadata | {'bootstrap_samples': bootstrap_samples, 'bootstrap_seed': seed,
        'notes': [
            'Each prefix uses disjoint sample-index halves. Shared prefixes retain shared estimates across comparisons.',
            'Steps are counted once per segment, without broadcasting across its tokens.',
            'Terminal halves share the observed original reward; inspect nonterminal advantages for interior credit reliability.',
            'Bootstrap resamples whole questions, preserving repeated rollouts and shared-prefix dependence within questions.',
            'Bootstrap intervals condition on this cohort and draw split; they do not resample fresh MC completions.',
            'Exact zero ties inflate overall sign agreement. The both-nonzero metric has a selected denominator.',
            'This measures half-K reliability. Correlation with an average containing the same draws is not independent reliability.',
        ], 'groups': {}}
    for name, points in groups.items():
        print(f'Summarizing {name}: {len(points)} observations', flush=True)
        summary['groups'][name] = summarize(points, bootstrap_samples, seed)
    return summary, groups


def save_plot(groups, summary, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    names = [name for name in ('prefix_values', 'uniform_nonterminal', 'steps_nonterminal') if groups.get(name)]
    if not names:
        return
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.7), squeeze=False, constrained_layout=True)
    half = summary['half_k']
    for ax, name in zip(axes[0], names, strict=True):
        points = groups[name]
        lo, hi = (0, 1) if name == 'prefix_values' else (-1, 1)
        edges = np.arange(lo * half, hi * half + 2) / half - .5 / half
        counts, _, _ = np.histogram2d([r['a'] for r in points], [r['b'] for r in points], bins=(edges, edges))
        mesh = ax.pcolormesh(edges, edges, np.ma.masked_equal(counts.T, 0), cmap='viridis', norm=LogNorm(vmin=1, vmax=max(2, counts.max())))
        ax.plot([lo, hi], [lo, hi], '--', color='gray', linewidth=1)
        ax.set(xlabel=f'First {half} draws', ylabel=f'Next {half} draws', aspect='equal')
        metric = summary['groups'][name]
        corr = 'undefined' if metric['pearson'] is None else f"{metric['pearson']:.3f}"
        label = {'prefix_values': 'Prefix success probabilities', 'uniform_nonterminal': 'Individual-token advantages', 'steps_nonterminal': 'Step advantages'}[name]
        ax.set_title(f'{label}\nPearson r = {corr}; n = {len(points):,}')
        fig.colorbar(mesh, ax=ax, label='Observations (log scale)', shrink=.8)
    fig.suptitle(f"{summary['student']}: independent {half} + {half} MC draws\nTerminal transitions excluded from advantage panels", fontsize=13)
    for suffix in ('png', 'pdf'):
        fig.savefig(target / f'split_half.{suffix}', dpi=180)
    plt.close(fig)


def save_report(summary, target):
    def fmt(value):
        return 'undefined' if value is None else f'{value:.3f}'
    lines = [f"# Vine split-half reliability: {summary['student']}", '',
             f"K={summary['k']}: first {summary['half_k']} versus next {summary['half_k']} draws at each prefix.", '',
             '| Group | N | Pearson (95% question bootstrap CI) | Spearman | Both nonzero | Sign agreement, both nonzero |',
             '|---|---:|---|---:|---:|---:|']
    for name, group in summary['groups'].items():
        if not group['num_observations']:
            continue
        ci = group['question_bootstrap_ci95'].get('pearson')
        interval = '' if ci is None else f' [{ci[0]:.3f}, {ci[1]:.3f}]'
        both_nonzero = '—' if name == 'prefix_values' else f"{group['num_both_nonzero']:,}"
        sign_agreement = '—' if name == 'prefix_values' else fmt(group['sign_agreement_both_nonzero'])
        lines.append(f"| {name} | {group['num_observations']:,} | {fmt(group['pearson'])}{interval} | {fmt(group['spearman'])} | {both_nonzero} | {sign_agreement} |")
    lines += ['', '![Split-half density plots](split_half.png)', '', *('- ' + note for note in summary['notes']), '']
    (target / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--k', type=int, default=16, help='Even total draw count; requires an existing k-K aggregate')
    parser.add_argument('--bootstrap-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42, help='Question-bootstrap seed; does not change the MC split')
    args = parser.parse_args()
    if args.k < 2 or args.k % 2 or args.bootstrap_samples < 0:
        parser.error('Require even --k >= 2 and --bootstrap-samples >= 0')
    print('Validating cached draws and loading paired estimates', flush=True)
    pairs, prefixes, metadata = load_pairs(args.run_dir, args.k)
    summary, groups = analyze(pairs, prefixes, metadata, args.bootstrap_samples, args.seed)
    target = args.run_dir / f'k-{args.k}' / 'split_half'
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'summary.json', summary)
    with (target / 'paired_estimates.jsonl').open('w') as file:
        for row in pairs:
            file.write(json.dumps(row, allow_nan=False) + '\n')
    save_plot(groups, summary, target)
    save_report(summary, target)
    print(f'Saved split-half analysis to {target}', flush=True)


if __name__ == '__main__':
    main()
