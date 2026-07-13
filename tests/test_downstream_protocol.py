import unittest

from scripts import downstream_protocol as protocol


class DownstreamProtocolTest(unittest.TestCase):
    def test_source_directory_is_resolved_once_and_reused_for_every_task(self):
        calls = []

        def moving_latest(requested):
            calls.append(requested)
            step = 11500 if len(calls) == 1 else 12000
            return {
                'checkpoint_path': (
                    'gs://bucket/run/checkpoints/' + f'{step:012d}'),
                'step': step,
            }

        pinned = protocol.pin_source_once('gs://bucket/run', moving_latest)
        by_task = protocol.pinned_sources_for_tasks(
            ('sst2', 'rte', 'wic', 'boolq', 'mnli'), pinned)

        self.assertEqual(1, len(calls))
        self.assertEqual(11500, pinned.step)
        self.assertTrue(all(source is pinned for source in by_task.values()))
        self.assertTrue(all(source.step == 11500 for source in by_task.values()))

    def test_default_epoch_schedules_match_real_row_counts(self):
        cases = {
            'sst2': (67349, 1053, 3159),
            'rte': (2490, 39, 195),
            'wic': (5428, 85, 425),
            'boolq': (9427, 148, 740),
            'mnli': (392702, 6136, 18408),
        }
        for task, (rows, steps_per_epoch, total_steps) in cases.items():
            with self.subTest(task=task):
                schedule = protocol.calculate_schedule(task, rows, 64, {})
                self.assertEqual(steps_per_epoch, schedule.steps_per_epoch)
                self.assertEqual(total_steps, schedule.total_steps)
                self.assertEqual(rows * schedule.requested_epochs,
                                 schedule.expected_examples_seen)
                self.assertEqual('task_default_epochs',
                                 schedule.schedule_source)

    def test_explicit_total_steps_remains_supported(self):
        schedule = protocol.calculate_schedule(
            'sst2', 67349, 64,
            {'total_steps': 800, 'eval_interval': 100})
        self.assertEqual(800, schedule.total_steps)
        self.assertIsNone(schedule.requested_epochs)
        self.assertEqual('explicit_total_steps', schedule.schedule_source)

    def test_explicit_epochs_and_total_steps_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            protocol.calculate_schedule(
                'rte', 2490, 64,
                {'num_epochs': 5, 'total_steps': 195})
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            protocol.calculate_schedule(
                'rte', 2490, 64,
                {'num_epochs': 5, 'max_steps': 195})

    def test_reported_accuracy_is_best_seen_not_final(self):
        schedule = protocol.calculate_schedule('rte', 2490, 64, {})
        source = protocol.PinnedSource(
            requested='gs://bucket/run',
            resolved='gs://bucket/run/checkpoints/000000011500',
            step=11500)
        result = protocol.build_result(
            task='rte', source=source, schedule=schedule,
            initial_acc=0.51, best_seen_acc=0.73, best_seen_step=120,
            final_acc=0.69, final_step=schedule.total_steps,
            eval_total=277)
        self.assertEqual(0.73, result['reported_acc'])
        self.assertNotEqual(result['final_acc'], result['reported_acc'])
        self.assertIn('0.730000', protocol.format_summary([result]))

    def test_task_source_mismatch_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, 'pinned suite source'):
            protocol.verify_task_source(
                'gs://bucket/run/checkpoints/000000012000', 12000,
                'gs://bucket/run/checkpoints/000000011500', 11500)

    def test_final_step_is_always_evaluated_once(self):
        schedule = protocol.calculate_schedule(
            'sst2', 67349, 64, {'epochs': 1, 'eval_interval': 100})
        steps = protocol.evaluation_steps(schedule)
        self.assertEqual(schedule.total_steps, steps[-1])
        self.assertEqual(1, steps.count(schedule.total_steps))
        self.assertIn('final', protocol.evaluation_reasons(
            schedule.total_steps, schedule))

    def test_small_task_eval_intervals_are_valid_and_capped(self):
        rte = protocol.calculate_schedule('rte', 2490, 64, {})
        wic = protocol.calculate_schedule('wic', 5428, 64, {})
        tiny = protocol.calculate_schedule(
            'rte', 10, 64, {'total_steps': 5})
        self.assertEqual(10, rte.eval_interval)
        self.assertEqual(20, wic.eval_interval)
        self.assertEqual(5, tiny.eval_interval)
        tiny_steps = protocol.evaluation_steps(tiny)
        self.assertEqual(5, tiny_steps[-1])
        self.assertEqual(len(tiny_steps), len(set(tiny_steps)))


if __name__ == '__main__':
    unittest.main()
