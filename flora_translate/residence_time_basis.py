"""Shared residence-time basis helpers for gas-liquid flow designs."""

from __future__ import annotations


P_STP_BAR = 1.01325
T_STP_K = 273.15

INLET_STP_BASIS = "inlet_stp"
IN_CHANNEL_BASIS = "in_channel"
LIQUID_ONLY_BASIS = "liquid_only"
UNKNOWN_BASIS = "unknown"


def normalize_residence_time_basis(value: object) -> str:
    """Normalize free-text residence-time basis labels."""

    text = str(value or "").strip().lower()
    if not text:
        return UNKNOWN_BASIS
    normalized = (
        text.replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    if any(token in normalized for token in ("inlet", "stp", "mfc", "metered")):
        return INLET_STP_BASIS
    if any(token in normalized for token in ("channel", "actual", "pressure_corrected", "reactor_total")):
        return IN_CHANNEL_BASIS
    if "liquid" in normalized:
        return LIQUID_ONLY_BASIS
    return UNKNOWN_BASIS


def residence_time_basis_label(basis: object) -> str:
    """Human-readable basis label for JSON output and prompts."""

    normalized = normalize_residence_time_basis(basis)
    if normalized == INLET_STP_BASIS:
        return "inlet/STP apparent residence time"
    if normalized == IN_CHANNEL_BASIS:
        return "in-channel pressure-corrected total residence time"
    if normalized == LIQUID_ONLY_BASIS:
        return "liquid-only reactor volume / liquid flow"
    return "unknown"


def actual_gas_flow_from_stp(
    gas_stp_mL_min: float,
    temperature_C: float,
    pressure_gauge_bar: float,
    *,
    min_abs_bar: float = 6.0,
) -> float:
    """Convert an MFC/STP gas flow to reactor actual volume flow."""

    if gas_stp_mL_min <= 0:
        return 0.0
    t_k = float(temperature_C) + 273.15
    p_abs = max(float(pressure_gauge_bar or 0.0) + P_STP_BAR, min_abs_bar)
    return float(gas_stp_mL_min) * (t_k / T_STP_K) * (P_STP_BAR / p_abs)


def stp_gas_flow_from_actual(
    gas_actual_mL_min: float,
    temperature_C: float,
    pressure_gauge_bar: float,
    *,
    min_abs_bar: float = 6.0,
) -> float:
    """Convert reactor actual gas volume flow to an MFC/STP setpoint."""

    if gas_actual_mL_min <= 0:
        return 0.0
    t_k = float(temperature_C) + 273.15
    p_abs = max(float(pressure_gauge_bar or 0.0) + P_STP_BAR, min_abs_bar)
    return float(gas_actual_mL_min) * (T_STP_K / t_k) * (p_abs / P_STP_BAR)
