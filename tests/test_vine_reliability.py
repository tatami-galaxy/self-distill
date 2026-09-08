"""Checks for independent cached MC splits and aligned reliability estimates."""
import tempfile
import unittest
from pathlib import Path

from eval import advantage_comparison as ac
from eval import vine_reliability as vr


def cache_for(original, end, rewards=(0, 0, 1, 1)):
    key = ac.prefix_key(original, end)
    return {'prefix': {'key': key, 'question_id': original['question_id'],
                      'student_prompt_ids': original['student_prompt_ids'],
                      'prefix_ids': original['completion_ids'][:end], 'final_answer': original['final_answer']},
            'draws': [{'sample_idx': i, 'seed': ac.keyed_seed(42, 'mc', key, i), 'reward': reward}
                      for i, reward in enumerate(rewards)]}


def original():
    return {'rollout_id': 'r', 'question_id': 'q', 'student_prompt_ids': [7],
            'completion_ids': [1, 2], 'final_answer': 'answer', 'reward': 1}


class SplitHalfTest(unittest.TestCase):
    def test_exact_disjoint_halves_ignore_later_resume_draws(self):
        cache = cache_for(original(), 0, (0, 0, 1, 1, 1, 1))
        counts = vr.split_counts(cache, 4, 42)
        self.assertEqual(counts['a'], {'successes': 0, 'samples': 2})
        self.assertEqual(counts['b'], {'successes': 2, 'samples': 2})
        self.assertEqual(counts['full'], {'successes': 2, 'samples': 4})
        cache['draws'] = cache['draws'][:4]
        self.assertEqual(vr.split_counts(cache, 4, 42), counts)

    def test_invalid_cache_is_rejected(self):
        for corruption in ('short', 'order', 'seed', 'reward', 'prefix'):
            with self.subTest(corruption=corruption):
                cache = cache_for(original(), 0)
                if corruption == 'short': cache['draws'].pop()
                if corruption == 'order': cache['draws'].reverse()
                if corruption == 'seed': cache['draws'][0]['seed'] += 1
                if corruption == 'reward': cache['draws'][0]['reward'] = .5
                if corruption == 'prefix': cache['prefix']['student_prompt_ids'] = [8]
                with self.assertRaises(ValueError): vr.split_counts(cache, 4, 42)

    def test_interior_halves_and_shared_terminal_reward(self):
        before = vr.split_counts(cache_for(original(), 0), 4, 42)
        after = vr.split_counts(cache_for(original(), 1, (1, 1, 0, 0)), 4, 42)
        self.assertEqual(vr.advantage_pair(before, after, 1), (1, -1))
        self.assertEqual(vr.advantage_pair(before, None, 1), (1, 0))
        self.assertEqual(vr.advantage_pair(before, None, 0), (0, -1))

    def test_metrics_distinguish_ties_from_nonzero_agreement(self):
        result = vr.metrics([0, 0, 1, -1], [0, 1, -1, -1])
        self.assertEqual(result['sign_agreement'], .5)
        self.assertEqual(result['num_both_nonzero'], 2)
        self.assertEqual(result['sign_agreement_both_nonzero'], .5)
        self.assertIsNone(vr.metrics([0, 0], [0, 0])['pearson'])
        self.assertIsNone(vr.metrics([0, 0], [0, 0])['sign_agreement_both_nonzero'])
        self.assertAlmostEqual(vr.metrics([0, 1, 2], [2, 1, 0])['pearson'], -1)

    def test_question_bootstrap_reproducible_and_clustered(self):
        points = [{'question_id': str(i // 3), 'a': i / 9, 'b': i / 9} for i in range(9)]
        result = vr.summarize(points, 20, 42)
        self.assertEqual(result, vr.summarize(points, 20, 42))
        self.assertEqual(result['num_questions'], 3)
        self.assertEqual(result['valid_bootstrap_samples']['pearson'], 20)
        self.assertAlmostEqual(result['question_bootstrap_ci95']['pearson'][0], 1)

    def test_load_validates_saved_full_k_and_preserves_pair_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = original()
            rollouts = {'rows': [row]}
            ac.write_json(root / 'rollouts.json', rollouts)
            ac.write_json(root / 'manifest.json', {'config': {'seed': 42, 'student': 'fake'}})
            ac.write_json(root / 'k-4/summary.json', {'mc_samples': 4, 'rollout_fingerprint': ac.fingerprint(rollouts),
                'num_rollouts': 1, 'num_prefixes': 2,
                'comparisons': [{'selection': 'uniform', 'unit': 'token', 'num_observations': 2}]})
            for end in (0, 1):
                cache = cache_for(row, end)
                ac.write_json(root / 'mc' / (cache['prefix']['key'] + '.json'), cache)
            comparisons = []
            for start in (0, 1):
                comparisons.append({'rollout_id': 'r', 'question_id': 'q', 'selection': 'uniform',
                    'start': start, 'end': start + 1, 'token_ids': row['completion_ids'][start:start + 1], 'reward': 1,
                    'vine': {'before_key': ac.prefix_key(row, start),
                             'after_key': ac.prefix_key(row, start + 1) if start == 0 else None,
                             'before_counts': {'successes': 2, 'samples': 4},
                             'after_counts': {'successes': 2, 'samples': 4} if start == 0 else None,
                             'advantage': 0 if start == 0 else .5}})
            import json
            (root / 'k-4/comparisons.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in comparisons))
            pairs, prefixes, metadata = vr.load_pairs(root, 4)
            self.assertEqual([(r['a'], r['b'], r['terminal']) for r in pairs], [(0, 0, False), (1, 0, True)])
            self.assertEqual(len(prefixes), 2)
            self.assertEqual(metadata['sample_indices_b'], [2, 3])
            cache = cache_for(row, 0)
            cache['draws'][0]['reward'] = 1
            ac.write_json(root / 'mc' / (cache['prefix']['key'] + '.json'), cache)
            with self.assertRaisesRegex(ValueError, 'current MC draws disagree'):
                vr.load_pairs(root, 4)


if __name__ == '__main__':
    unittest.main()
