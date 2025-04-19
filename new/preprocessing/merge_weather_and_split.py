#!/usr/bin/env python3
import os
import pandas as pd

# ─── CONFIG (no args!) ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(__file__)                   # new/preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR,'..'))
PD_DIR       = os.path.join(PROJECT_ROOT, 'processed-data')
JOURNEYS_CSV = os.path.join(PD_DIR, 'journey_with_station_meta.csv')
WEATHER_CSV  = os.path.join(PD_DIR, 'london_weather_2023_hourly.csv')
OUTDIR       = os.path.join(PD_DIR, 'clients')
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_and_build_features(journeys_csv: str,
                                 weather_csv: str) -> pd.DataFrame:
    # 1) Load journeys with station meta
    df = pd.read_csv(
        journeys_csv,
        parse_dates=['Start date', 'End date'],
        dtype={'Start station number': str, 'End station number': str}
    )

    # 2) Extract static station metadata from the per‑trip file:
    dep_meta = (
        df[['Start station number','Start station name','Start lat','Start lon','Start nbDocks']]
        .drop_duplicates()
        .rename(columns={
            'Start station number':'terminal',
            'Start station name':'station_name',
            'Start lat':'lat', 'Start lon':'lon', 'Start nbDocks':'nbDocks'
        })
    )
    arr_meta = (
        df[['End station number','End station name','End lat','End lon','End nbDocks']]
        .drop_duplicates()
        .rename(columns={
            'End station number':'terminal',
            'End station name':'station_name',
            'End lat':'lat', 'End lon':'lon', 'End nbDocks':'nbDocks'
        })
    )
    station_meta = (
        pd.concat([dep_meta, arr_meta], ignore_index=True)
          .drop_duplicates(subset=['terminal'])
          .set_index('terminal')
    )

    # 3) Floor to hour and count
    df['start_hour'] = df['Start date'].dt.floor('H')
    df['end_hour']   = df['End date'].dt.floor('H')

    deps = (
        df.groupby(['Start station number','start_hour'])
          .size()
          .rename('num_departures')
          .reset_index()
          .rename(columns={
              'Start station number':'terminal',
              'start_hour':'datetime'
          })
    )
    arrs = (
        df.groupby(['End station number','end_hour'])
          .size()
          .rename('num_arrivals')
          .reset_index()
          .rename(columns={
              'End station number':'terminal',
              'end_hour':'datetime'
          })
    )

    # 4) Where‑from (origins) & where‑to (destinations) means
    origins = (
        df.groupby(['End station number','end_hour'])
          [['Start lat','Start lon']]
          .mean()
          .reset_index()
          .rename(columns={
              'End station number':'terminal',
              'end_hour':'datetime',
              'Start lat':'avg_origin_lat',
              'Start lon':'avg_origin_lon'
          })
    )
    dests = (
        df.groupby(['Start station number','start_hour'])
          [['End lat','End lon']]
          .mean()
          .reset_index()
          .rename(columns={
              'Start station number':'terminal',
              'start_hour':'datetime',
              'End lat':'avg_dest_lat',
              'End lon':'avg_dest_lon'
          })
    )

    # 5) Merge all of the above
    sh = pd.merge(deps, arrs, on=['terminal','datetime'], how='outer').fillna(0)
    sh = pd.merge(sh, origins, on=['terminal','datetime'], how='left')
    sh = pd.merge(sh, dests,   on=['terminal','datetime'], how='left')

    # cast counts back to int
    sh['num_departures'] = sh['num_departures'].astype(int)
    sh['num_arrivals']   = sh['num_arrivals'].astype(int)

    # 6) Fill any missing avg_origin_* / avg_dest_* with the station's own lat/lon
    sh['avg_origin_lat'] = sh['avg_origin_lat'].fillna(sh['terminal'].map(station_meta['lat']))
    sh['avg_origin_lon'] = sh['avg_origin_lon'].fillna(sh['terminal'].map(station_meta['lon']))
    sh['avg_dest_lat']   = sh['avg_dest_lat'].fillna(sh['terminal'].map(station_meta['lat']))
    sh['avg_dest_lon']   = sh['avg_dest_lon'].fillna(sh['terminal'].map(station_meta['lon']))

    # 7) Merge weather
    weather = pd.read_csv(weather_csv, parse_dates=['datetime'])
    full = pd.merge(sh, weather, on='datetime', how='left')

    # 8) Attach static station metadata
    station_meta = station_meta.reset_index()
    full = pd.merge(full, station_meta, on='terminal', how='left')
    # now columns include: terminal, datetime, counts, avg_origin_*, avg_dest_*,
    # weather columns…, plus station_name, lat, lon, nbDocks

    return full

def split_into_clients(full_df: pd.DataFrame, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for term, group in full_df.groupby('terminal'):
        fname = f"station_{int(term):04d}.csv"
        path  = os.path.join(outdir, fname)
        group.to_csv(path, index=False)
        print(f" • Wrote {fname} ({len(group)} rows)")

if __name__ == "__main__":
    print("🔄 Building station‑hour feature tables …")
    full_df = aggregate_and_build_features(JOURNEYS_CSV, WEATHER_CSV)

    print(f"📂 Splitting into per‑station files in {OUTDIR!r} …")
    split_into_clients(full_df, OUTDIR)

    print("🎉 Done!")
