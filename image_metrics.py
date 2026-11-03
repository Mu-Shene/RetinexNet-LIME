"""Normalized pixel errors for saved enhancement images and paired references."""

import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

from run_storage import write_json


METRIC_FIELDS = ['input_image', 'output_image', 'reference_image', 'width',
                 'height', 'pixel_values', 'mae', 'mse', 'status', 'error']


def _load_normalized_rgb(path):
    with Image.open(str(path)) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float64) / 255.0


def save_image_metrics(inputs, output_dir, reference_dir):
    """Save per-image MAE/MSE and summaries in the normalized RGB [0, 1] scale.

    Measurements use the saved, quantized enhancement files, not raw network
    predictions. References match input filenames exactly. Missing, unreadable,
    or differently sized images are recorded without preventing other pairs
    from being measured; their empty metrics never contribute to averages.
    The final CSV row, TEST_SET_MEAN, averages the successfully measured images.
    """
    inputs = [Path(path) for path in inputs]
    if not inputs:
        raise ValueError('Image metrics require at least one input image.')
    output_dir = Path(output_dir).resolve()
    reference_dir = Path(reference_dir).resolve() if reference_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'metrics.csv'
    summary_path = output_dir / 'metrics_summary.json'
    rows = []
    valid_metrics = []
    absolute_sums = []
    squared_sums = []
    pixel_values = 0

    for input_path in inputs:
        output_path = output_dir / (input_path.stem + '_S' + (input_path.suffix or '.png'))
        reference_path = reference_dir / input_path.name if reference_dir is not None else None
        row = {'input_image': str(input_path), 'output_image': str(output_path),
               'reference_image': str(reference_path) if reference_path is not None else '',
               'width': '', 'height': '', 'pixel_values': '',
               'mae': '', 'mse': '', 'status': '', 'error': ''}
        rows.append(row)
        if not output_path.is_file():
            row.update(status='missing_output', error='Enhanced image does not exist.')
            continue
        try:
            enhanced = _load_normalized_rgb(output_path)
        except (OSError, ValueError) as error:
            row.update(status='unreadable_output', error=str(error))
            continue
        row.update(width=int(enhanced.shape[1]), height=int(enhanced.shape[0]),
                   pixel_values=int(enhanced.size))
        if reference_path is None or not reference_path.is_file():
            row.update(status='missing_reference', error='Matching reference image does not exist.')
            continue
        try:
            reference = _load_normalized_rgb(reference_path)
        except (OSError, ValueError) as error:
            row.update(status='unreadable_reference', error=str(error))
            continue
        if enhanced.shape != reference.shape:
            row.update(status='size_mismatch', error='Enhanced RGB shape {} differs from reference {}.'.format(
                enhanced.shape, reference.shape))
            continue

        difference = enhanced - reference
        absolute_sum = float(np.sum(np.abs(difference), dtype=np.float64))
        squared_sum = float(np.sum(np.square(difference), dtype=np.float64))
        count = int(difference.size)
        mae = absolute_sum / count
        mse = squared_sum / count
        if not math.isfinite(mae) or not math.isfinite(mse):
            row.update(status='invalid_metric', error='Pixel error is not finite.')
            continue
        row.update(mae=mae, mse=mse, status='ok')
        valid_metrics.append((mae, mse))
        absolute_sums.append(absolute_sum)
        squared_sums.append(squared_sum)
        pixel_values += count

    evaluated_count = len(valid_metrics)
    failed_count = len(rows) - evaluated_count
    summary = {
        'status': 'complete' if not failed_count else ('partial' if evaluated_count else 'unavailable'),
        'normalization_range': [0.0, 1.0],
        'normalization': 'RGB channel values divided by 255',
        'basis': 'saved_enhanced_images',
        'aggregation': 'arithmetic mean of per-image metrics; pixel-weighted metrics weight RGB-value counts',
        'reference_matching': 'exact input filename',
        'input_count': len(rows),
        'evaluated_count': evaluated_count,
        'failed_count': failed_count,
        'evaluated_pixel_values': pixel_values,
        'mean_mae': math.fsum(pair[0] for pair in valid_metrics) / evaluated_count if evaluated_count else None,
        'mean_mse': math.fsum(pair[1] for pair in valid_metrics) / evaluated_count if evaluated_count else None,
        'pixel_weighted_mae': math.fsum(absolute_sums) / pixel_values if pixel_values else None,
        'pixel_weighted_mse': math.fsum(squared_sums) / pixel_values if pixel_values else None,
        'reference_dir': str(reference_dir) if reference_dir is not None else None,
        'csv_path': str(csv_path),
        'summary_path': str(summary_path),
    }
    with csv_path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            'input_image': 'TEST_SET_MEAN',
            'mae': summary['mean_mae'],
            'mse': summary['mean_mse'],
            'status': 'summary',
            'error': 'Evaluated images: {}/{}; metrics status: {}'.format(
                evaluated_count, len(rows), summary['status']),
        })
    write_json(summary_path, summary)
    return summary
