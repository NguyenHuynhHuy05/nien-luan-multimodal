"""Tests use temporary image fixtures, never research data."""
import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from audit_image_hashes import collect_manifest, compare_manifest, cross_split_duplicates


class ImageHashAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.images = self.root / 'data', self.root / 'Images'
        self.data.mkdir()
        (self.images / 'class').mkdir(parents=True)
        for i, split in enumerate(('train', 'val', 'test')):
            Image.new('RGB', (8, 8), (i * 80, 20, 40)).save(self.images / 'class' / f'{split}.png')
            with (self.data / f'{split}_processed.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['split', 'image_path'])
                writer.writeheader()
                writer.writerow({'split': split, 'image_path': f'class/{split}.png'})

    def cli(self, output, *more):
        return subprocess.run([sys.executable, str(Path(__file__).with_name('audit_image_hashes.py')),
                               '--data-dir', str(self.data), '--image-root', str(self.images),
                               '--output', str(output), *more], capture_output=True, text=True)

    def test_hash_and_all_cross_split_pairs(self):
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            with self.subTest(pair=(a, b)):
                for i, split in enumerate(('train', 'val', 'test')):
                    Image.new('RGB', (8, 8), (i * 80, 20, 40)).save(self.images / 'class' / f'{split}.png')
                (self.images / 'class' / f'{b}.png').write_bytes((self.images / 'class' / f'{a}.png').read_bytes())
                groups = cross_split_duplicates(collect_manifest(self.data, self.images))
                digest = hashlib.sha256((self.images / 'class' / f'{a}.png').read_bytes()).hexdigest()
                self.assertEqual({row['split'] for row in groups[digest]}, {a, b})

    def test_cli_stops_reports_paths_saves_manifest_and_preserves_inputs(self):
        (self.images / 'class/val.png').write_bytes((self.images / 'class/train.png').read_bytes())
        before = {path: path.read_bytes() for folder in (self.data, self.images)
                  for path in folder.rglob('*') if path.is_file()}
        output = self.root / 'manifest.csv'
        result = self.cli(output)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn('[train] class/train.png', result.stderr)
        self.assertIn('[val] class/val.png', result.stderr)
        with output.open(encoding='utf-8-sig', newline='') as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 3)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_cli_compare_and_no_overwrite(self):
        baseline = self.root / 'baseline.csv'
        self.assertEqual(self.cli(baseline).returncode, 0)
        original = baseline.read_bytes()
        self.assertEqual(self.cli(baseline).returncode, 1)
        self.assertEqual(baseline.read_bytes(), original)
        self.assertEqual(self.cli(self.root / 'same.csv', '--compare', str(baseline)).returncode, 0)
        Image.new('RGB', (8, 8), (255, 0, 0)).save(self.images / 'class/test.png')
        result = self.cli(self.root / 'changed.csv', '--compare', str(baseline))
        self.assertEqual(result.returncode, 1)
        self.assertIn('"changed": [["test", "class/test.png"]]', result.stdout)

    def test_corrupt_image_cli_stops_with_path_and_no_manifest(self):
        (self.images / 'class/val.png').write_bytes(b'not an image')
        output = self.root / 'invalid.csv'
        result = self.cli(output)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Cannot read/decode image [val]', result.stderr)
        self.assertIn('val.png', result.stderr)
        self.assertFalse(output.exists())

    def test_unreadable_image_reports_path(self):
        with patch('audit_image_hashes.sha256_file', side_effect=PermissionError('access denied')):
            with self.assertRaisesRegex(ValueError, r'Cannot read/decode image \[train\].*train.png'):
                collect_manifest(self.data, self.images)

    def test_missing_image_cli_stops_with_path_and_no_manifest(self):
        (self.images / 'class/test.png').unlink()
        output = self.root / 'missing.csv'
        result = self.cli(output)
        self.assertEqual(result.returncode, 1)
        self.assertIn('Image missing:', result.stderr)
        self.assertIn('test.png', result.stderr)
        self.assertFalse(output.exists())

    def test_added_removed_and_changed_records(self):
        old = collect_manifest(self.data, self.images)
        current = [dict(row) for row in old]
        current.pop()
        current[0]['sha256'] = '0' * 64
        current.append({'split': 'test', 'image_path': 'class/new.png', 'sha256': '1' * 64})
        diff = compare_manifest(current, old)
        self.assertEqual(diff['added'], [('test', 'class/new.png')])
        self.assertEqual(diff['removed'], [('test', 'class/test.png')])
        self.assertEqual(diff['changed'], [('train', 'class/train.png')])

    def test_missing_and_escaping_images_stop(self):
        (self.images / 'class/train.png').unlink()
        with self.assertRaises(FileNotFoundError):
            collect_manifest(self.data, self.images)
        (self.data / 'train_processed.csv').write_text('split,image_path\ntrain,../outside.png\n')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            collect_manifest(self.data, self.images)


if __name__ == '__main__':
    unittest.main()
