"""Check that GPU measurements are attributed and missing counters stay missing."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from profile_preprocessing import parse_pmon, summarize


class ProfileTests(unittest.TestCase):
    def test_pmon_header_units_and_missing_metrics(self):
        _, header = parse_pmon('# gpu pid type sm mem enc dec fb command', None)
        _, header2 = parse_pmon('# Idx # C/G % % % % MB name', header)
        self.assertEqual(header2, header)
        row, _ = parse_pmon('0 123 C 42 7 - - 2048 python', header)
        self.assertEqual(row['pid'], 123)
        self.assertEqual(row['sm_percent'], 42)
        self.assertEqual(row['vram_mib'], 2048)
        self.assertIsNone(row['encoder_percent'])
        row, _ = parse_pmon('0 456 C - - - - 4096 python', header)
        self.assertIsNone(row['sm_percent'])
        self.assertEqual(row['vram_mib'], 4096)

    def test_summary_does_not_assign_device_activity_to_process(self):
        sample = dict(processes=[dict(pid=123, created_at=1, name='python', cpu_percent_one_core=250,
                         rss_mib=1000, gpu=[dict(gpu_index=0, sm_percent=None, vram_mib=2000)])],
                      gpus=[dict(index=0, utilization_percent=90, used_mib=10000, total_mib=48000)])
        report = summarize(iter([sample, sample]))
        process = report['processes'][0]
        self.assertEqual(process['cpu_percent_one_core']['mean'], 250)
        self.assertEqual(process['gpu'][0]['vram_mib']['peak'], 2000)
        self.assertIsNone(process['gpu'][0]['sm_percent']['mean'])
        self.assertEqual(report['gpus'][0]['utilization_percent']['mean'], 90)


if __name__ == '__main__':
    unittest.main()
