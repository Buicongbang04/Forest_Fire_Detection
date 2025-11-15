import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List

# ======================= utils =======================

def parse_acq_time(v) -> pd.Timedelta:
    try:
        iv = int(v)
        s = f"{iv:04d}"
        hh, mm = int(s[:-2]), int(s[-2:])
        return pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
    except Exception:
        return pd.NaT

def round_to_grid(x, step=0.25):
    return np.round(np.asarray(x, dtype=float) / step) * step

def rh_from_t_tdew_celsius(T_c: pd.Series, Td_c: pd.Series) -> pd.Series:
    a, b = 17.625, 243.04
    es = 6.1094 * np.exp((a * T_c) / (b + T_c))
    e  = 6.1094 * np.exp((a * Td_c) / (b + Td_c))
    RH = 100.0 * (e / es)
    return np.clip(RH, 0, 100)

# ======================= Tính FWI =======================

def ffmc_step(ffmc_prev, T, RH, W, R):
    mo = 147.2*(101.0-ffmc_prev)/(59.5+ffmc_prev)
    if R > 0.5:
        rf = R - 0.5
        if mo <= 150.0:
            mr = mo + 42.5*rf*np.exp(-100.0/(251.0-mo))*(1.0 - np.exp(-6.93/rf))
        else:
            mr = mo + 42.5*rf*np.exp(-100.0/(251.0-mo))*(1.0 - np.exp(-6.93/rf)) \
                 + 0.0015*(mo-150.0)**2 * (rf**0.5)
        mo = min(mr, 250.0)
    Ed = 0.942*(RH**0.679) + 11*np.exp((RH-100.0)/10.0) + 0.18*(21.1 - T)*(1 - np.exp(-0.115*RH))
    Ew = 0.618*(RH**0.753) + 10*np.exp((RH-100.0)/10.0) + 0.18*(21.1 - T)*(1 - np.exp(-0.115*RH))
    if mo > Ed:
        ko = 0.424*(1 - (RH/100.0)**1.7) + 0.0694*(W**0.5)*(1 - (RH/100.0)**8)
        kd = ko*0.581*np.exp(0.0365*T)
        m  = Ed + (mo-Ed)*(10**(-kd))
    elif mo < Ew:
        k1 = 0.424*(1 - ((100.0-RH)/100.0)**1.7) + 0.0694*(W**0.5)*(1 - ((100.0-RH)/100.0)**8)
        kw = k1*0.581*np.exp(0.0365*T)
        m  = Ew - (Ew-mo)*(10**(-kw))
    else:
        m = mo
    return float(np.clip(59.5*(250.0-m)/(147.2+m), 0, 101))

def dmc_step(dmc_prev, T, RH, R, month):
    if R > 1.5:
        Pe = 0.92*R - 1.27
        Mm1 = 20 + np.exp(5.6348 - dmc_prev/43.43)
        if dmc_prev <= 33:   b = 100.0/(0.5 + 0.3*dmc_prev)
        elif dmc_prev <= 65: b = 14.0 - 1.3*np.log(dmc_prev)
        else:                b = 6.2*np.log(dmc_prev) - 17.2
        Mr = Mm1 + 1000.0*Pe/(48.77 + b*Pe)
        dmc_prev = max(244.72 - 43.43*np.log(Mr - 20.0), 0.0)
    Le = {1:6.5,2:7.5,3:9.0,4:12.8,5:13.9,6:13.9,7:12.4,8:10.9,9:9.4,10:8.0,11:7.0,12:6.0}[int(month)]
    T_ = max(T, -1.1)
    K = 1.894*(T_+1.1)*(100.0-RH)*Le*1e-6
    return float(max(dmc_prev + 100.0*K, 0.0))

def dc_step(dc_prev, T, R, month):
    if R > 2.8:
        Pd = 0.83*R - 1.27
        Qm1 = 800.0*np.exp(-dc_prev/400.0)
        Qr  = Qm1 + 3.937*Pd
        dc_prev = max(400.0*np.log(800.0/Qr), 0.0)
    Lf = {1:-1.6,2:-1.6,3:-1.6,4:0.9,5:3.8,6:5.8,7:6.4,8:5.0,9:2.4,10:0.4,11:-1.6,12:-1.6}[int(month)]
    T_ = max(T, -2.8)
    V = max(0.36*(T_+2.8) + Lf, 0.0)
    return float(max(dc_prev + 0.5*V, 0.0))

def isi_from_ffmc_wind(ffmc, W):
    m = 147.2*(101.0-ffmc)/(59.5+ffmc)
    fU = np.exp(0.05039*W);  fF = 91.9*np.exp(-0.1386*m)*(1 + (m**5.31)/4.93e7)
    return float(0.208*fU*fF)

def bui_from_dmc_dc(dmc, dc):
    if dmc <= 0.4*dc:  return float(0.8*dmc*dc / (dmc + 0.4*dc + 1e-12))
    else:              return float(dmc - (1 - (0.8*dc)/(dmc + 0.4*dc + 1e-12)) * (0.92 + (0.0114*dmc)**1.7))

def fwi_from_isi_bui(isi, bui):
    if bui <= 80.0: D = 0.626*(bui**0.809) + 2.0
    else:           D = 1000.0/(25.0 + 108.64*np.exp(-0.023*bui))
    B = 0.1*isi*D
    return float(B if B <= 1.0 else np.exp(2.72*(0.434*np.log(B))**0.647))

# ======================= hàm chính =======================

def build_clean_one_year(year: int,
                        base_dir: Union[str, Path] = "data/data",
                        grid_step: float = 0.25) -> Path:
    base_dir = Path(base_dir)
    era5_path  = base_dir / "era5" / f"era5_{year}.csv"
    modis_path = base_dir / "nasa" / f"modis_{year}_Vietnam.csv"
    out_path   = base_dir / "clean" / f"clean_{year}.csv"

    era5 = pd.read_csv(era5_path)
    modis = pd.read_csv(modis_path)

    # chuẩn hóa thời gian
    era5 = era5.rename(columns={"latitude":"lat","longitude":"lon"})
    modis = modis.rename(columns={"latitude":"lat","longitude":"lon"})

    era5["valid_time"] = pd.to_datetime(era5["valid_time"], errors="coerce")
    era5["date"] = era5["valid_time"].dt.date

    modis["acq_date"]  = pd.to_datetime(modis["acq_date"], errors="coerce")
    modis["acq_delta"] = modis["acq_time"].apply(parse_acq_time)
    modis["event_time"] = modis["acq_date"] + modis["acq_delta"]
    modis["date"] = pd.to_datetime(modis["event_time"]).dt.date

    # chọn điểm chung
    era5["lat_grid"] = round_to_grid(era5["lat"], grid_step)
    era5["lon_grid"] = round_to_grid(era5["lon"], grid_step)
    modis["lat_grid"] = round_to_grid(modis["lat"], grid_step)
    modis["lon_grid"] = round_to_grid(modis["lon"], grid_step)

    # tính dữ liệu thời tiết
    era5["Temperature"] = era5["t2m"] - 273.15
    era5[" RH"] = rh_from_t_tdew_celsius(era5["t2m"] - 273.15, era5["d2m"] - 273.15)
    era5[" Ws"] = np.sqrt(era5["u10"]**2 + era5["v10"]**2) * 3.6
    era5["Rain "] = era5["tp"] * 1000.0
    era5["month"] = pd.to_datetime(era5["date"]).dt.month

    #
    join_cols = ["date","lat_grid","lon_grid"]
    era5_daily = (era5[join_cols + ["Temperature"," RH"," Ws","Rain ","month"]]
                .drop_duplicates(subset=join_cols + ["month"])  # safety
                .sort_values(join_cols))

    # Dữ liệu modis
    modis_presence = (modis.dropna(subset=["date"])
                        .drop_duplicates(subset=join_cols)[join_cols]
                        .assign(present=1))

    # dữ liệu era5
    base = era5_daily.merge(modis_presence, on=join_cols, how="left")
    base["present"] = base["present"].fillna(0).astype(int)
    base["Classes"] = np.where(base["present"] == 1, "fire", "not fire")
    base = base.drop(columns=["present"])

    # tính FWI
    init_FFMC, init_DMC, init_DC = 85.0, 6.0, 15.0
    records = []
    for (latg, long), grp in base.groupby(["lat_grid","lon_grid"], sort=False):
        ffmc, dmc, dc = init_FFMC, init_DMC, init_DC
        for r in grp.sort_values("date").itertuples(index=False):
            date, latg_v, long_v, T, RH, W, R, month, cls = r
            ffmc = ffmc_step(ffmc, T, RH, W, R)
            dmc  = dmc_step(dmc, T, RH, R, month)
            dc   = dc_step(dc, T, R, month)
            isi  = isi_from_ffmc_wind(ffmc, W)
            bui  = bui_from_dmc_dc(dmc, dc)
            fwi  = fwi_from_isi_bui(isi, bui)
            records.append({
                "date": date, "lat_grid": latg, "lon_grid": long,
                "Temperature": T, "RH": RH, "Ws": W, "Rain": R,
                "FFMC": ffmc, "DMC": dmc, "DC": dc, "ISI": isi, "BUI": bui, "FWI": fwi,
                "Classes": cls
            })
    out = pd.DataFrame.from_records(records)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols11 = ["Temperature","RH","Ws","Rain","FFMC","DMC","DC","ISI","BUI","FWI","Classes"]
    # out[cols11].to_csv(out_path, index=False) # dữ liệu không có timeline
    out.to_csv(out_path, index=False) # dữ liệu timeline 
    return out_path

def build_and_concat_years(year_start: int, year_end: int,
                        base_dir: Union[str, Path] = "data/data"):
    base_dir = Path(base_dir)
    all_paths: List[Path] = []
    for y in range(year_start, year_end + 1):
        p = build_clean_one_year(y, base_dir)
        all_paths.append(p)

    dfs = [pd.read_csv(p) for p in all_paths]
    big = pd.concat(dfs, axis=0, ignore_index=True)
    out_all = base_dir / "clean" / f"clean_{year_start}_{year_end}_timelines.csv"
    big.to_csv(out_all, index=False)
    return out_all

# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    BASE = "data/data"
    out_all = build_and_concat_years(2000, 2025, BASE)
    print(f"Wrote: {out_all}")
