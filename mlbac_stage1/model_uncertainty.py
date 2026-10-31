"""Simulator configuration and retrospective errors; never controller inputs."""
import numpy as np
from .config import safety_margin_settings


def nonnegative_scalar(value, name):
    """Reject nonfinite values, booleans and arrays rather than silently broadcast."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite nonnegative scalar') from exc
    if isinstance(value, (bool, np.bool_)) or array.ndim != 0 or not np.isfinite(array) or array < 0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return float(array)


COMPARISON_FIELDS = (
    'predicted_temperature_c', 'actual_temperature_c', 'temperature_error_c',
    'predicted_battery_j', 'actual_battery_j', 'battery_error_j',
    'predicted_energy_j', 'actual_energy_j', 'energy_error_j',
    'temperature_adverse_error_c', 'battery_adverse_error_j',
    'temperature_adverse_error_within_configured_margin',
    'battery_adverse_error_within_configured_margin',
    'temperature_adverse_error_within_effective_margin',
    'battery_adverse_error_within_effective_margin',
    'temperature_absolute_error_within_configured_margin',
    'battery_absolute_error_within_configured_margin',
)


def prediction_error_report(predicted_report, post_state, metrics, params):
    """Compare the executed action's final-gate prediction to its actual endpoint.

    Errors are actual minus predicted. Coverage comparisons use no outward
    tolerance. Configured margins can be evaluated even when filtering is off.
    """
    margins = safety_margin_settings(params)
    result = {key: None for key in COMPARISON_FIELDS}
    result['safety_margins'] = margins
    if post_state is None:
        return result
    prediction = predicted_report['predicted_state']
    predicted_t = np.asarray(prediction['T'], dtype=float)
    predicted_b = np.asarray(prediction['B'], dtype=float)
    predicted_e = np.asarray(predicted_report['predicted_energy_j'], dtype=float)
    actual_t = np.asarray(post_state['T'], dtype=float)
    actual_b = np.asarray(post_state['B'], dtype=float)
    actual_e = np.asarray(metrics['energy'], dtype=float)
    error_t, error_b = actual_t - predicted_t, actual_b - predicted_b
    adverse_t, adverse_b = np.maximum(0., error_t), np.maximum(0., -error_b)
    result.update(
        predicted_temperature_c=predicted_t.copy(), actual_temperature_c=actual_t.copy(),
        temperature_error_c=error_t, predicted_battery_j=predicted_b.copy(),
        actual_battery_j=actual_b.copy(), battery_error_j=error_b,
        predicted_energy_j=predicted_e.copy(), actual_energy_j=actual_e.copy(),
        energy_error_j=actual_e-predicted_e,
        temperature_adverse_error_c=adverse_t, battery_adverse_error_j=adverse_b,
        temperature_adverse_error_within_configured_margin=(adverse_t <= margins['temperature_margin_c']),
        battery_adverse_error_within_configured_margin=(adverse_b <= margins['battery_margin_j']),
        temperature_adverse_error_within_effective_margin=(adverse_t <= margins['effective_temperature_margin_c']),
        battery_adverse_error_within_effective_margin=(adverse_b <= margins['effective_battery_margin_j']),
        temperature_absolute_error_within_configured_margin=(np.abs(error_t) <= margins['temperature_margin_c']),
        battery_absolute_error_within_configured_margin=(np.abs(error_b) <= margins['battery_margin_j']),
    )
    return result
