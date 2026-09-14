"""OpenDrift backward hindcast ensemble for spill attribution.

WHAT THIS DOES
--------------
Given a detection (a spill polygon centroid + detection time), run a
**backward** particle ensemble to estimate *where the oil came from*.
Combine the particles' final positions into a 2-D kernel-density origin
ellipse, then score nearby vessels by proximity.

This is the core of the attribution pipeline — the physical inverse of
the drift. OpenDrift handles the heavy lifting; we wrap the I/O and the
statistics.

FORCING
-------
* Surface currents: CMEMS GLORYS12 (already verified live in this env).
* Surface wind: ERA5 hourly 10m u/v (requires accepted CDS licence).

FALLBACKS
---------
* If ERA5 is unavailable (no licence / network down), use a constant
  synthetic wind reader so the pipeline still produces an answer — it
  just won't be physics-faithful. The runner flags the run with
  ``wind_source="synthetic_constant"`` so the UI can show the caveat.
* If CMEMS is unavailable, the runner still runs on wind-only forcing
  and flags ``current_source="synthetic_constant"``.

WMC (WELL-MIXED CRITERION)
--------------------------
The runner emits the divergence of the diffusion tensor ``∇·K`` per
ensemble so downstream code can apply the spec-mandated correction.
We do not apply the correction here — that lives in
``engine.wmc_correction`` and is plugged into the FastAPI layer.

CONFIG KNOBS
------------
Edit ``DetectionConfig`` in the FastAPI request body. Sensible defaults
already match the Wakashio case (24–48 h backtrack, 16 members).
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

# OpenDrift prints a LOT; trim it.
warnings.filterwarnings("ignore")
os.environ.setdefault("OPENDRIFT_DISABLE_LOGGING", "1")


# Lazy-imported so a missing OpenDrift doesn't break module load.
def _opendrift():
    from opendrift.models.openoil import OpenOil
    from opendrift.readers import reader_netCDF_CF_generic

    return reader_netCDF_CF_generic, OpenOil


@dataclass
class EnsembleConfig:
    """Backward hindcast knobs."""

    n_members: int = 16
    duration_h: float = 48.0
    time_step_min: float = 15.0
    oil_type: str = "GENERIC MEDIUM CRUDE"
    wind_drift_factor: float = 0.03  # ~3% of wind on surface slick
    current_uncertainty: float = 0.10  # m/s
    diffusion: float = 50.0  # m²/s
    seed: int = 20200725


@dataclass
class OriginEllipse:
    """Confidence ellipse of the backtracked origin cluster."""

    center_lon: float
    center_lat: float
    semi_major_km: float
    semi_minor_km: float
    orientation_deg: float  # 0=N, clockwise
    n_particles: int
    p50_radius_km: float  # 50% quantile distance from centre
    p95_radius_km: float  # 95% quantile distance from centre

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VesselCandidate:
    """One vessel scored against the origin ellipse."""

    mmsi: str
    name: str
    flag: str | None
    distance_to_origin_km: float
    inside_p95: bool
    inside_p50: bool
    age_h: float | None
    wind_flag: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionResult:
    """Full output of one attribution run."""

    run_utc: str
    detection_time: str
    detection_centroid: tuple[float, float]  # (lon, lat)
    detection_area_km2: float
    wind_source: str  # "era5" | "synthetic_constant"
    current_source: str  # "cmems" | "synthetic_constant"
    wmc_divergence_max: float  # max |∇·K| seen across the run
    config: dict[str, Any]
    origin: OriginEllipse
    suspect_vessels: list[VesselCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _ellipse_from_xy(x_km: np.ndarray, y_km: np.ndarray) -> OriginEllipse:
    """Build an OriginEllipse from particle x/y offsets (km) via PCA."""
    if len(x_km) < 4:
        # Fallback to a tiny circle so downstream code never crashes.
        return OriginEllipse(
            center_lon=0.0,
            center_lat=0.0,
            semi_major_km=0.0,
            semi_minor_km=0.0,
            orientation_deg=0.0,
            n_particles=int(len(x_km)),
            p50_radius_km=0.0,
            p95_radius_km=0.0,
        )
    cx, cy = float(np.mean(x_km)), float(np.mean(y_km))
    X = np.stack([x_km - cx, y_km - cy], axis=0)  # 2 x N
    cov = np.cov(X)
    if cov.shape != (2, 2):
        return OriginEllipse(cx, cy, 0.0, 0.0, 0.0, len(x_km), 0.0, 0.0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sqrt of eigenvalues -> 1-sigma semi-axes. We report 2-sigma (~95%) for `semi_*`.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    semi_major = float(2.0 * math.sqrt(max(eigvals[0], 0.0)))
    semi_minor = float(2.0 * math.sqrt(max(eigvals[1], 0.0)))
    # Orientation: angle of the major axis from north, clockwise.
    major_vec = eigvecs[:, 0]
    angle_math = math.degrees(math.atan2(major_vec[0], major_vec[1]))
    orientation = (90.0 - angle_math) % 180.0

    # Distance quantiles.
    r = np.sqrt(X[0] ** 2 + X[1] ** 2)
    p50 = float(np.percentile(r, 50))
    p95 = float(np.percentile(r, 95))
    return OriginEllipse(
        center_lon=cx,
        center_lat=cy,
        semi_major_km=semi_major,
        semi_minor_km=semi_minor,
        orientation_deg=orientation,
        n_particles=int(len(x_km)),
        p50_radius_km=p50,
        p95_radius_km=p95,
    )


def _make_constant_reader(
    name: str,
    fields: dict[str, float],
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> Any:
    """Build an in-memory OpenDrift constant reader from a fields dict.

    OpenDrift's constant reader defaults ``start_time``/``end_time`` to None;
    we set them to the simulation window so OpenDrift accepts queries at
    any time inside it.
    """
    from opendrift.readers.reader_constant import Reader as ConstantReader

    r = ConstantReader(
        {
            "sea_water_speed": fields.get("sea_water_speed", 0.0),
            "x_sea_water_velocity": fields.get("x_sea_water_velocity", 0.0),
            "y_sea_water_velocity": fields.get("y_sea_water_velocity", 0.0),
            "x_wind": fields.get("x_wind", 0.0),
            "y_wind": fields.get("y_wind", 0.0),
        },
        name=name,
    )
    if start_time is not None:
        r.start_time = start_time
    if end_time is not None:
        r.end_time = end_time
    return r


def run_backward_attribution(
    *,
    detection_time: datetime,
    detection_lon: float,
    detection_lat: float,
    detection_area_km2: float,
    wind_reader: Any,  # opendrift reader (or None for synthetic)
    current_reader: Any | None,  # opendrift reader (or None for synthetic)
    vessels: list[dict[str, Any]] | None = None,
    cfg: EnsembleConfig | None = None,
    synthetic_wind_ms: float = 5.0,
    synthetic_wind_dir_from_deg: float = 110.0,
    synthetic_current_ms: float = 0.10,
    synthetic_current_dir_deg: float = 220.0,
) -> AttributionResult:
    """Run a TRUE backward OpenDrift integration and return an AttributionResult.

    Method
    ------
    Seed ``n_members`` particles across the *detection footprint* at the
    acquisition instant, then integrate the OpenOil SDE **backwards in time**
    for ``duration_h``. The terminal positions are the candidate origins; we
    fit a 2-sigma PCA ellipse to them and rank vessels by distance to that
    ellipse.

    Two things this replaced, and why:

    1. *Release-in-the-past forward filtering.* The previous version seeded a
       ~50 km box at T-duration, ran forward, and hard-filtered to particles
       landing within ~10 km of the detection. That keeps roughly
       pi*10^2 / 100^2 ~ 3% of the ensemble — with 64 members you get ~2
       particles, so the "ellipse" was fitted to noise and n_particles came
       back as 3. Backward integration uses 100% of the ensemble.

    2. *Reading ``o.history``.* OpenDrift 1.14 exposes trajectories on
       ``o.result`` (an xarray Dataset) with dims ``(trajectory, time)`` —
       note the axis order, it is the transpose of the old ``o.history``
       convention. Slicing ``lon[-1]`` therefore grabs the last *particle*
       across all times, not the last *timestep*. Always use ``lon[:, -1]``.

    Backward mode requires BOTH ``time_step`` and ``duration`` to be negative;
    a positive duration silently runs forward.

    Landmask: OpenOil auto-loads the GSHHG global landmask. Oil slicks are
    frequently detected against a coastline (Wakashio grounded on a reef), so
    half the ensemble would strand within the first step and the origin would
    collapse back onto the detection. We disable the auto-landmask for the
    hindcast and treat the whole AOI as open water.
    """
    cfg = cfg or EnsembleConfig()
    _, OpenOil = _opendrift()

    # The reader may be ERA5 or GFS; it carries its own provenance so the
    # result never claims "era5" for a GFS-driven run.
    wind_src = (
        getattr(wind_reader, "sentinel_source", "era5")
        if wind_reader is not None
        else "synthetic_constant"
    )
    cur_src = "cmems" if current_reader is not None else "synthetic_constant"

    o = OpenOil(loglevel=0)
    # See docstring: no GSHHG, treat AOI as open water so a coastal slick
    # doesn't strand the ensemble on step 1.
    o.set_config("general:use_auto_landmask", False)
    o.set_config("environment:fallback:land_binary_mask", 0)

    # OpenOil will refuse to start without wind AND current readers, so when
    # the caller didn't supply real ones we synthesise constant fields from
    # the wind/current parameters above. The synthetic values are flagged in
    # the result so the UI can show the caveat.
    rad_wind = math.radians(synthetic_wind_dir_from_deg + 180.0)  # meteorological → toward
    u_wind = synthetic_wind_ms * math.sin(rad_wind)
    v_wind = synthetic_wind_ms * math.cos(rad_wind)
    rad_cur = math.radians(synthetic_current_dir_deg)
    u_cur = synthetic_current_ms * math.sin(rad_cur)
    v_cur = synthetic_current_ms * math.cos(rad_cur)

    sim_start = (detection_time - timedelta(hours=cfg.duration_h)).replace(tzinfo=None)
    sim_end = detection_time.replace(tzinfo=None)
    for r in (wind_reader, current_reader):
        if r is not None:
            o.add_reader(r)
    if wind_reader is None:
        o.add_reader(
            _make_constant_reader(
                "synthetic_wind",
                {
                    "x_wind": u_wind,
                    "y_wind": v_wind,
                },
                start_time=sim_start,
                end_time=sim_end,
            )
        )
    if current_reader is None:
        o.add_reader(
            _make_constant_reader(
                "synthetic_current",
                {
                    "x_sea_water_velocity": u_cur,
                    "y_sea_water_velocity": v_cur,
                },
                start_time=sim_start,
                end_time=sim_end,
            )
        )

    # WMC divergence sampler — track max |∇·K| over the run.
    # With a spatially constant K, ∇·K == 0 and the Well-Mixed Criterion
    # correction is a genuine no-op; we report that rather than pretending a
    # correction was applied. Once a diffusivity *field* is supplied, this
    # sampler is the hook that turns into a real correction term.
    wmc_max = 0.0

    # ── Seed across the observed slick footprint, at the acquisition time ──
    # A spill of area A is roughly a disc of radius sqrt(A/pi); seeding into
    # that disc is the honest initial condition. Anything narrower invents
    # precision the SAR polygon does not have.
    km_per_deg_lat = 110.57
    km_per_deg_lon = 111.32 * max(math.cos(math.radians(detection_lat)), 1e-3)
    footprint_radius_km = max(0.25, math.sqrt(detection_area_km2 / math.pi))

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_members
    # Uniform over the disc: sqrt(U) keeps the density even instead of
    # clustering at the centre (which a plain uniform radius would do).
    r_km = footprint_radius_km * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    seed_lon = detection_lon + (r_km * np.cos(theta)) / km_per_deg_lon
    seed_lat = detection_lat + (r_km * np.sin(theta)) / km_per_deg_lat

    det_naive = detection_time.replace(tzinfo=None)
    o.seed_elements(
        lon=seed_lon.tolist(),
        lat=seed_lat.tolist(),
        time=det_naive,
        oil_type=cfg.oil_type,
        wind_drift_factor=cfg.wind_drift_factor,
    )

    # ── Backward integration ──────────────────────────────────────────────
    # BOTH duration and time_step must be negative or OpenDrift silently
    # integrates forward and the "origin" lands downwind of the slick.
    o.run(
        duration=timedelta(hours=-cfg.duration_h),
        time_step=-timedelta(minutes=cfg.time_step_min),
        time_step_output=timedelta(hours=1),
        outfile=None,
    )

    # o.result dims are (trajectory, time) — see docstring. [:, -1] is the
    # furthest-back state of every particle.
    hist_lon = np.asarray(o.result["lon"].values)
    hist_lat = np.asarray(o.result["lat"].values)
    origin_lon = hist_lon[:, -1]
    origin_lat = hist_lat[:, -1]
    valid = np.isfinite(origin_lon) & np.isfinite(origin_lat)
    origin_lon, origin_lat = origin_lon[valid], origin_lat[valid]
    n_kept = int(valid.sum())

    # Local-km offsets relative to the DETECTION, so the ellipse centre can be
    # reported in absolute lon/lat at the end.
    x_km = (origin_lon - detection_lon) * km_per_deg_lon
    y_km = (origin_lat - detection_lat) * km_per_deg_lat

    if n_kept > 0:
        cluster_dx_km = float(np.mean(x_km))
        cluster_dy_km = float(np.mean(y_km))
        cluster_d_km = _haversine_km(
            detection_lon, detection_lat, float(np.mean(origin_lon)), float(np.mean(origin_lat))
        )
    else:
        cluster_dx_km = cluster_dy_km = cluster_d_km = 0.0

    notes: list[str] = []
    notes.append(
        "WMC: diffusivity K is spatially constant, so ∇·K = 0 and the "
        "Well-Mixed Criterion correction is a no-op for this run "
        f"(wmc_divergence_max={wmc_max})."
    )
    if n_kept == 0:
        notes.append("Backward run produced no surviving particles — origin indeterminate.")
    elif (
        cluster_d_km < 0.5 and wind_src == "synthetic_constant" and cur_src == "synthetic_constant"
    ):
        notes.append("Cluster collapsed onto detection: no real forcing available.")
    if n_kept < n:
        notes.append(f"{n - n_kept}/{n} particles deactivated during the backtrack.")

    origin = _ellipse_from_xy(x_km, y_km)
    # Translate the local-km offsets back to absolute lon/lat.
    origin = OriginEllipse(
        center_lon=detection_lon + cluster_dx_km / km_per_deg_lon,
        center_lat=detection_lat + cluster_dy_km / km_per_deg_lat,
        semi_major_km=origin.semi_major_km,
        semi_minor_km=origin.semi_minor_km,
        orientation_deg=origin.orientation_deg,
        n_particles=n_kept,
        p50_radius_km=origin.p50_radius_km,
        p95_radius_km=origin.p95_radius_km,
    )

    # Score vessels.
    suspects: list[VesselCandidate] = []
    for v in vessels or []:
        vlon = v.get("longitude")
        vlat = v.get("latitude")
        if vlon is None or vlat is None:
            continue
        d = _haversine_km(vlon, vlat, origin.center_lon, origin.center_lat)
        suspects.append(
            VesselCandidate(
                mmsi=v.get("mmsi", ""),
                name=v.get("name", ""),
                flag=v.get("flag"),
                distance_to_origin_km=round(d, 2),
                inside_p50=d <= origin.p50_radius_km,
                inside_p95=d <= origin.p95_radius_km,
                age_h=v.get("age_h"),
                wind_flag="OK" if wind_src == "era5" else "LOW_CONFIDENCE_NO_WIND",
                provenance=v.get("provenance", "live_terrestrial"),
            )
        )
    suspects.sort(key=lambda c: c.distance_to_origin_km)

    return AttributionResult(
        run_utc=datetime.now(UTC).isoformat(),
        detection_time=detection_time.isoformat(),
        detection_centroid=(detection_lon, detection_lat),
        detection_area_km2=detection_area_km2,
        wind_source=wind_src,
        current_source=cur_src,
        wmc_divergence_max=wmc_max,
        config=asdict(cfg),
        origin=origin,
        suspect_vessels=suspects[:25],  # top 25
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────
# Forward forecast (shoreline impact)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ForecastConeStep:
    """One time slice of the forward forecast cone."""

    hours_ahead: float
    valid_time: str
    center_lon: float
    center_lat: float
    p50_radius_km: float
    p95_radius_km: float
    n_active: int
    n_stranded: int


@dataclass
class ForecastResult:
    """Output of one forward drift forecast."""

    run_utc: str
    origin_time: str
    origin_lon: float
    origin_lat: float
    duration_h: float
    wind_source: str
    current_source: str
    config: dict[str, Any]
    cone: list[ForecastConeStep] = field(default_factory=list)
    final_center: tuple[float, float] = (0.0, 0.0)
    stranded_fraction: float = 0.0
    first_stranding_h: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_forward_forecast(
    *,
    origin_time: datetime,
    origin_lon: float,
    origin_lat: float,
    seed_radius_km: float,
    duration_h: float = 48.0,
    wind_reader: Any = None,
    current_reader: Any = None,
    cfg: EnsembleConfig | None = None,
    synthetic_wind_ms: float = 5.0,
    synthetic_wind_dir_from_deg: float = 110.0,
    synthetic_current_ms: float = 0.10,
    synthetic_current_dir_deg: float = 220.0,
    use_landmask: bool = True,
) -> ForecastResult:
    """Project the slick FORWARD from a known origin and report shoreline risk.

    This is the mirror of `run_backward_attribution`: same SDE, positive time.
    It answers "where does it go, and does it hit land?" — the question that
    actually drives a response.

    Landmask handling is deliberately the OPPOSITE of the hindcast. The
    backward run disables GSHHG because a slick detected against a coast would
    strand the whole ensemble on step 1 and collapse the origin onto the
    detection. Here stranding is the *signal* — it is how shoreline impact is
    measured — so the landmask is ON by default.

    `stranded_fraction` is the share of the ensemble that beached. This
    requires `general:coastline_action: 'stranding'`: with `'previous'`
    OpenDrift quietly nudges a beached element back to its last ocean position
    and leaves its status ACTIVE, so `stranded_fraction` reads 0% even when
    the oil is demonstrably ashore. Verified: GSHHG itself correctly reports
    land at 57.705,-20.425, so a 0% reading is an action-setting bug, not a
    missing coastline.
    """
    cfg = cfg or EnsembleConfig()
    cfg.duration_h = duration_h
    _, OpenOil = _opendrift()

    wind_src = (
        getattr(wind_reader, "sentinel_source", "era5")
        if wind_reader is not None
        else "synthetic_constant"
    )
    cur_src = "cmems" if current_reader is not None else "synthetic_constant"

    o = OpenOil(loglevel=0)
    if use_landmask:
        o.set_config("general:use_auto_landmask", True)
        o.set_config("general:coastline_action", "stranding")
    else:
        o.set_config("general:use_auto_landmask", False)
        o.set_config("environment:fallback:land_binary_mask", 0)

    rad_wind = math.radians(synthetic_wind_dir_from_deg + 180.0)
    u_wind = synthetic_wind_ms * math.sin(rad_wind)
    v_wind = synthetic_wind_ms * math.cos(rad_wind)
    rad_cur = math.radians(synthetic_current_dir_deg)
    u_cur = synthetic_current_ms * math.sin(rad_cur)
    v_cur = synthetic_current_ms * math.cos(rad_cur)

    t0 = origin_time.replace(tzinfo=None)
    t1 = t0 + timedelta(hours=duration_h)
    for r in (wind_reader, current_reader):
        if r is not None:
            o.add_reader(r)
    if wind_reader is None:
        o.add_reader(
            _make_constant_reader(
                "synthetic_wind",
                {
                    "x_wind": u_wind,
                    "y_wind": v_wind,
                },
                start_time=t0,
                end_time=t1,
            )
        )
    if current_reader is None:
        o.add_reader(
            _make_constant_reader(
                "synthetic_current",
                {
                    "x_sea_water_velocity": u_cur,
                    "y_sea_water_velocity": v_cur,
                },
                start_time=t0,
                end_time=t1,
            )
        )

    km_per_deg_lat = 110.57
    km_per_deg_lon = 111.32 * max(math.cos(math.radians(origin_lat)), 1e-3)

    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_members
    r_km = seed_radius_km * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    theta = rng.uniform(0.0, 2.0 * math.pi, size=n)
    seed_lon = origin_lon + (r_km * np.cos(theta)) / km_per_deg_lon
    seed_lat = origin_lat + (r_km * np.sin(theta)) / km_per_deg_lat

    o.seed_elements(
        lon=seed_lon.tolist(),
        lat=seed_lat.tolist(),
        time=t0,
        oil_type=cfg.oil_type,
        wind_drift_factor=cfg.wind_drift_factor,
    )

    # Forward: positive duration AND positive time_step (cf. the hindcast).
    o.run(
        duration=timedelta(hours=duration_h),
        time_step=timedelta(minutes=cfg.time_step_min),
        time_step_output=timedelta(hours=1),
        outfile=None,
    )

    lon = np.asarray(o.result["lon"].values)  # (trajectory, time)
    lat = np.asarray(o.result["lat"].values)
    status = np.asarray(o.result["status"].values)
    times = np.asarray(o.result["time"].values)

    # Status 1 == stranded in OpenOil's default status list.
    STRANDED = 1
    cone: list[ForecastConeStep] = []
    first_stranding_h: float | None = None
    n_total = lon.shape[0]
    # Cumulative stranding. Once an element beaches, OpenDrift deactivates it
    # and blanks its later status/position — so counting status at the FINAL
    # step reports ~0% even when nearly the whole slick is ashore (measured:
    # 62 of 64 beached, final-step fraction read 0.0). Shoreline impact is
    # "did it ever hit land", so accumulate over the whole trajectory.
    ever_stranded = np.any(status == STRANDED, axis=1)

    # When everything beaches early, deactivated positions go NaN and a naive
    # "skip steps with no finite positions" collapses the cone to ~1 point —
    # which looks like a broken run rather than "the oil is ashore and has
    # stopped moving". Keep emitting steps, freezing the centre at its last
    # known position, so the track visibly stops where the oil stopped.
    last_cx, last_cy = origin_lon, origin_lat
    for ti in range(lon.shape[1]):
        L, A = lon[:, ti], lat[:, ti]
        n_str = int(np.nansum(np.any(status[:, : ti + 1] == STRANDED, axis=1)))
        ok = np.isfinite(L) & np.isfinite(A)
        if ok.any():
            cx, cy = float(np.nanmean(L[ok])), float(np.nanmean(A[ok]))
            last_cx, last_cy = cx, cy
            dx = (L[ok] - cx) * km_per_deg_lon
            dy = (A[ok] - cy) * km_per_deg_lat
            rad = np.sqrt(dx * dx + dy * dy)
        else:
            cx, cy = last_cx, last_cy
            rad = np.zeros(0)
        hours = float((times[ti] - times[0]) / np.timedelta64(1, "h"))
        if n_str > 0 and first_stranding_h is None:
            first_stranding_h = round(hours, 2)
        cone.append(
            ForecastConeStep(
                hours_ahead=round(hours, 2),
                valid_time=str(times[ti])[:19],
                center_lon=cx,
                center_lat=cy,
                p50_radius_km=round(float(np.percentile(rad, 50)), 3) if rad.size else 0.0,
                p95_radius_km=round(float(np.percentile(rad, 95)), 3) if rad.size else 0.0,
                n_active=int(ok.sum()),
                n_stranded=n_str,
            )
        )

    final_lon, final_lat = lon[:, -1], lat[:, -1]
    ok_f = np.isfinite(final_lon) & np.isfinite(final_lat)
    final_center = (
        float(np.nanmean(final_lon[ok_f])) if ok_f.any() else origin_lon,
        float(np.nanmean(final_lat[ok_f])) if ok_f.any() else origin_lat,
    )
    # Denominator is the whole ensemble, not just the survivors — otherwise an
    # oil slick that beaches 90% of its particles reports a low fraction
    # because most of them are no longer "active".
    stranded_fraction = float(np.nansum(ever_stranded)) / max(n_total, 1)
    n_survivors = int(ok_f.sum())

    notes: list[str] = []
    if not use_landmask:
        notes.append(
            "Landmask disabled — stranded_fraction and first_stranding_h are "
            "meaningless for this run (shoreline impact not evaluated)."
        )
    if wind_src == "synthetic_constant":
        notes.append("Wind forcing is synthetic — forecast is illustrative only.")
    # OpenDrift ends a run once no elements are active, so a slick that beaches
    # completely terminates early. Report the REAL end hour, not the requested
    # one — "13/64 afloat at +48h" would be nonsense when the run stopped at +4h.
    end_h = cone[-1].hours_ahead if cone else 0.0
    ended_early = cone and end_h < duration_h - 0.5
    if use_landmask and ended_early:
        notes.append(
            f"Run ended at +{end_h:g} h (requested {duration_h:g} h): the ensemble "
            f"beached completely ({stranded_fraction * 100:.0f}%). There is nothing "
            "left afloat to forecast beyond that point."
        )
    elif use_landmask and n_survivors < 0.25 * max(n_total, 1):
        # Heavy beaching is itself the headline, but it also means the tail of
        # the cone is fitted to a handful of particles. Say so rather than
        # letting a 2-particle p95 look like a tight, confident forecast.
        notes.append(
            f"Only {n_survivors}/{n_total} particles remain afloat at +{end_h:g} h "
            f"({stranded_fraction * 100:.0f}% beached). Late-cone radii are "
            "computed from few particles and understate the true spread."
        )

    return ForecastResult(
        run_utc=datetime.now(UTC).isoformat(),
        origin_time=origin_time.isoformat(),
        origin_lon=origin_lon,
        origin_lat=origin_lat,
        duration_h=duration_h,
        wind_source=wind_src,
        current_source=cur_src,
        config=asdict(cfg),
        cone=cone,
        final_center=final_center,
        stranded_fraction=round(stranded_fraction, 4),
        first_stranding_h=first_stranding_h,
        notes=notes,
    )
