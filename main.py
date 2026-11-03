# -*- coding: utf-8 -*-
"""Fresh RetinexNet training, dated artifacts, and testing of selected models."""
from __future__ import print_function

import argparse
import csv
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import math
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from PIL import Image

from image_metrics import save_image_metrics
from run_storage import (create_run, load_run, local_timestamp, publish_latest,
                         write_json)

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
LOSS_FIELDS = ['phase', 'epoch', 'batch', 'total_batches', 'global_step',
               'loss', 'learning_rate', 'elapsed_seconds']
EPOCH_FIELDS = ['phase', 'epoch', 'training_loss', 'is_best']


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def build_parser(default_phase='train'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['train', 'test'], default=default_phase)
    parser.add_argument('--epoch', type=positive_int, default=100,
                        help='epochs for EACH phase; every training run starts fresh')
    parser.add_argument('--batch_size', type=positive_int, default=16)
    parser.add_argument('--patch_size', type=positive_int, default=48)
    parser.add_argument('--start_lr', type=float, default=0.001)
    parser.add_argument('--eval_every_epoch', type=positive_int, default=20,
                        help='interval for training sample images and last-model snapshots; lowest mean training loss is selected every epoch')
    parser.add_argument('--data_root', default=None,
                        help='training data root; default: project data directory, or saved data root for testing')
    parser.add_argument('--output_root', default=str(PROJECT_DIR / 'runs'),
                        help='parent of dated training runs and latest.json')
    parser.add_argument('--run_dir', default=None,
                        help='test a completed run (absolute or project-relative path) instead of latest.json')
    parser.add_argument('--test_dir', default=None,
                        help='test inputs; default: data_root/eval/low for both training and test-only runs')
    parser.add_argument('--reference_dir', default=None,
                        help='same-named reference images for normalized MAE/MSE; default: high beside the test directory')
    parser.add_argument('--save_dir', default=None,
                        help='optional parent for dated test-only outputs')
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=None,
                        help='training default: 0 (CPU); test-only default: saved configuration')
    parser.add_argument('--gpu_idx', default=None)
    parser.add_argument('--gpu_mem', type=float, default=None)
    parser.add_argument('--decom', type=int, choices=[0, 1], default=None)
    return parser


def project_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_DIR / path).resolve()


def apply_runtime_defaults(args, saved=None):
    saved = saved or {}
    for name, fallback in [('data_root', str(PROJECT_DIR / 'data')),
                           ('use_gpu', 0), ('gpu_idx', '0'), ('gpu_mem', 0.5), ('decom', 0)]:
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, fallback))
    if not math.isfinite(args.gpu_mem) or not 0 < args.gpu_mem <= 1:
        raise ValueError('--gpu_mem must be greater than 0 and at most 1')
    if not math.isfinite(args.start_lr) or args.start_lr <= 0:
        raise ValueError('--start_lr must be positive and finite')


def image_paths(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError('Image directory does not exist: {}'.format(directory))
    return sorted(path for path in directory.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def paired_paths(directory):
    low = image_paths(directory / 'low')
    high = image_paths(directory / 'high')
    if [path.name for path in low] != [path.name for path in high]:
        raise ValueError('Low/high image filenames must match: {}'.format(directory))
    return list(zip(low, high))


def load_rgb(path):
    with Image.open(str(path)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def load_pairs(pairs, patch_size=None):
    lows, highs = [], []
    for low_path, high_path in pairs:
        low, high = load_rgb(low_path), load_rgb(high_path)
        if low.shape != high.shape:
            raise ValueError('Low/high image dimensions differ: {}'.format(low_path.name))
        if patch_size and min(low.shape[:2]) < patch_size:
            raise ValueError('Image smaller than --patch_size: {}'.format(low_path))
        lows.append(low)
        highs.append(high)
    return lows, highs


class TeeLogger(object):
    def __init__(self, console, log_file):
        self.console, self.log_file = console, log_file

    def write(self, message):
        self.console.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def capture_console(path):
    with Path(path).open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(TeeLogger(sys.stdout, log)), redirect_stderr(TeeLogger(sys.stderr, log)):
            yield


@contextmanager
def model_session(args):
    # Set visibility BEFORE importing TensorFlow or constructing a new graph.
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx) if args.use_gpu else '-1'
    import tensorflow as tf
    from model import lowlight_enhance
    if args.use_gpu and not tf.test.is_gpu_available(cuda_only=True):
        raise RuntimeError('This TensorFlow environment cannot use a CUDA GPU. Run with --use_gpu=0.')
    with tf.Graph().as_default():
        if args.use_gpu:
            options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
            config = tf.ConfigProto(gpu_options=options)
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})
        print('[*] Device: {}'.format('GPU' if args.use_gpu else 'CPU'))
        with tf.Session(config=config) as session:
            yield tf, lowlight_enhance(session)


def run_inference(model, inputs, checkpoint_dir, output_dir, decom, reference_dir=None):
    inputs = [Path(path) for path in inputs]
    if not inputs:
        raise ValueError('Testing requires at least one input image')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [load_rgb(path) for path in inputs]
    model.test(images, [], [str(path) for path in inputs],
               save_dir=str(output_dir), decom_flag=decom,
               checkpoint_dir=str(checkpoint_dir))
    reference_dir = Path(reference_dir) if reference_dir is not None else inputs[0].parent.parent / 'high'
    metrics = save_image_metrics(inputs, output_dir, reference_dir)
    print('[*] Image metrics: {}'.format(metrics['csv_path']))
    print('[*] Metrics summary: {}'.format(metrics['summary_path']))
    if metrics['evaluated_count']:
        print('[*] Normalized [0, 1] MAE={:.8f}, MSE={:.8f} ({} / {} images)'.format(
            metrics['mean_mae'], metrics['mean_mse'], metrics['evaluated_count'], len(inputs)))
    if metrics['status'] != 'complete':
        print('[!] Metrics {}: see per-image status/error in metrics.csv'.format(metrics['status']))
    return {'input_count': len(inputs), 'output_count': len(image_paths(output_dir)),
            'output_dir': str(output_dir), 'metrics': metrics, 'completed_at': local_timestamp()}


def run_training(args):
    if args.run_dir or args.save_dir:
        raise ValueError('--run_dir and --save_dir apply to test-only runs')
    apply_runtime_defaults(args)
    data_root = project_path(args.data_root)
    test_dir = project_path(args.test_dir or data_root / 'eval' / 'low')
    reference_dir = project_path(args.reference_dir or test_dir.parent / 'high')
    training_pairs = paired_paths(data_root / 'our485')
    if len(training_pairs) < args.batch_size:
        raise ValueError('Need at least {} paired training images; found {}'.format(args.batch_size, len(training_pairs)))
    test_inputs = image_paths(test_dir)
    if not test_inputs:
        raise ValueError('No test images found: {}'.format(test_dir))

    output_root = project_path(args.output_root)
    run_dir = create_run(output_root)
    paths = {'run_dir': str(run_dir),
             'checkpoint_dir': str(run_dir / 'checkpoints' / 'best'),
             'last_checkpoint_dir': str(run_dir / 'checkpoints' / 'last'),
             'sample_dir': str(run_dir / 'samples'),
             'train_log': str(run_dir / 'train.log'),
             'loss_csv': str(run_dir / 'loss.csv'),
             'epoch_csv': str(run_dir / 'epochs.csv'),
             'test_dir': str(test_dir),
             'reference_dir': str(reference_dir),
             'test_results': str(run_dir / 'test_results')}
    for name in ['checkpoint_dir', 'last_checkpoint_dir', 'sample_dir']:
        Path(paths[name]).mkdir(parents=True, exist_ok=True)
    configuration = {'schema_version': 2, 'run_id': run_dir.name,
                     'status': 'training', 'training_complete': False,
                     'started_at': local_timestamp(), 'arguments': vars(args).copy(),
                     'paths': paths, 'checkpoints': {}, 'last_checkpoints': {},
                     'selection': {'Decom': {'metric': 'training_loss', 'mode': 'min'},
                                   'Relight': {'metric': 'training_loss', 'mode': 'min'}},
                     'data': {'training_pairs': len(training_pairs),
                              'test_images': len(test_inputs)}}
    config_file = run_dir / 'config.json'
    write_json(config_file, configuration)

    with capture_console(paths['train_log']):
        try:
            print('[*] Run: {}'.format(run_dir))
            print('[*] Fresh training: {} epochs for Decom and {} for Relight'.format(args.epoch, args.epoch))
            print('[*] Log: {}'.format(paths['train_log']))
            train_low, train_high = load_pairs(training_pairs, args.patch_size)
            sample_low = train_low[:3]
            learning_rates = args.start_lr * np.ones(args.epoch, dtype=np.float32)
            learning_rates[20:] *= 0.1
            with open(paths['loss_csv'], 'w', newline='', encoding='utf-8', buffering=1) as loss_file, \
                    open(paths['epoch_csv'], 'w', newline='', encoding='utf-8', buffering=1) as epoch_file:
                loss_writer = csv.DictWriter(loss_file, fieldnames=LOSS_FIELDS)
                epoch_writer = csv.DictWriter(epoch_file, fieldnames=EPOCH_FIELDS)
                loss_writer.writeheader()
                epoch_writer.writeheader()
                epoch_losses = defaultdict(list)

                def record_loss(record):
                    if not math.isfinite(record['loss']):
                        raise FloatingPointError('Non-finite training loss for {} epoch {}'.format(record['phase'], record['epoch']))
                    loss_writer.writerow(record)
                    loss_file.flush()
                    epoch_losses[(record['phase'], record['epoch'])].append(record['loss'])

                with model_session(args) as (tf, model):
                    def training_score(phase, epoch):
                        batches = epoch_losses.pop((phase, epoch))
                        score = math.fsum(batches) / len(batches)
                        selected = configuration['selection'][phase]
                        previous = selected.get('value')
                        is_best = previous is None or score < previous
                        if is_best:
                            selected.update({'epoch': epoch, 'value': score})
                            write_json(config_file, configuration)
                        epoch_writer.writerow({'phase': phase, 'epoch': epoch, 'training_loss': score,
                                               'is_best': int(is_best)})
                        epoch_file.flush()
                        print('[*] Training {} epoch {}: {}={:.6f}{}'.format(
                            phase, epoch, selected['metric'], score, ' [best]' if is_best else ''))
                        return score

                    for phase in ['Decom', 'Relight']:
                        best_dir = Path(paths['checkpoint_dir']) / phase
                        last_dir = Path(paths['last_checkpoint_dir']) / phase
                        final_checkpoint = model.train(
                            train_low, train_high, sample_low,
                            batch_size=args.batch_size, patch_size=args.patch_size,
                            epoch=args.epoch, lr=learning_rates, sample_dir=paths['sample_dir'],
                            ckpt_dir=str(last_dir), eval_every_epoch=args.eval_every_epoch,
                            train_phase=phase, resume=False, loss_callback=record_loss,
                            best_ckpt_dir=str(best_dir), epoch_callback=training_score,
                            best_mode=configuration['selection'][phase]['mode'])
                        configuration['last_checkpoints'][phase] = str(Path(final_checkpoint).resolve())
                        selected_checkpoint = tf.train.latest_checkpoint(str(best_dir))
                        if selected_checkpoint is None:
                            raise RuntimeError('Best checkpoint missing for {}'.format(phase))
                        configuration['checkpoints'][phase] = str(Path(selected_checkpoint).resolve())
                        # Relight must be trained with the selected BEST Decom, not the last epoch.
                        loaded, _ = model.load(model.saver_Decom if phase == 'Decom' else model.saver_Relight, str(best_dir))
                        if not loaded:
                            raise RuntimeError('Cannot restore selected {} model'.format(phase))
                        write_json(config_file, configuration)

                    configuration.update({'training_complete': True, 'status': 'trained',
                                          'training_completed_at': local_timestamp()})
                    write_json(config_file, configuration)
                    publish_latest(output_root, run_dir)
                    print('[*] Training complete. Testing the selected best Decom/Relight models...')
                    configuration['test'] = run_inference(
                        model, test_inputs, paths['checkpoint_dir'], paths['test_results'], args.decom,
                        reference_dir=reference_dir)
            configuration.update({'status': 'completed', 'completed_at': local_timestamp()})
            write_json(config_file, configuration)
            print('[*] Finished. Best models: {}'.format(paths['checkpoint_dir']))
            print('[*] Test results: {}'.format(paths['test_results']))
        except BaseException as error:
            configuration.update({'status': 'test_failed' if configuration['training_complete'] else 'failed',
                                  'error': {'type': type(error).__name__, 'message': str(error)},
                                  'failed_at': local_timestamp()})
            write_json(config_file, configuration)
            traceback.print_exc()
            raise
    return run_dir


def run_test(args):
    output_root = project_path(args.output_root)
    explicit_run = project_path(args.run_dir) if args.run_dir else None
    run_dir, configuration = load_run(output_root, explicit_run)
    apply_runtime_defaults(args, configuration.get('arguments'))
    test_dir = project_path(args.test_dir or project_path(args.data_root) / 'eval' / 'low')
    reference_dir = project_path(args.reference_dir or test_dir.parent / 'high')
    inputs = image_paths(test_dir)
    if not inputs:
        raise ValueError('No test images found: {}'.format(test_dir))
    test_root = project_path(args.save_dir) if args.save_dir else run_dir / 'tests'
    test_run = create_run(test_root)
    report = {'source_run': str(run_dir), 'started_at': local_timestamp(),
              'test_dir': str(test_dir), 'reference_dir': str(reference_dir),
              'checkpoints': configuration['checkpoints'],
              'selection': configuration.get('selection'), 'arguments': vars(args).copy()}
    with capture_console(test_run / 'test.log'):
        try:
            print('[*] Testing latest completed training run: {}'.format(run_dir))
            print('[*] Selected models: {}'.format(configuration['selection']))
            with model_session(args) as (_, model):
                report.update(run_inference(model, inputs, configuration['paths']['checkpoint_dir'],
                                            test_run / 'results', args.decom, reference_dir=reference_dir))
            report['status'] = 'completed'
            print('[*] Test results: {}'.format(report['output_dir']))
        except BaseException as error:
            report.update({'status': 'failed', 'error': str(error)})
            traceback.print_exc()
            raise
        finally:
            write_json(test_run / 'test_config.json', report)
    return test_run


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.phase == 'train':
        return run_training(args)
    return run_test(args)


if __name__ == '__main__':
    main()
