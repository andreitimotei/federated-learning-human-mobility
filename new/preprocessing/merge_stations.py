#!/usr/bin/env python3
# preprocessing/parse_and_merge.py

import xml.etree.ElementTree as ET
import pandas as pd

def parse_stations(xml_path: str) -> pd.DataFrame:
    """
    Parse stations.xml and return a DataFrame with columns:
      - station_id: the internal XML <id>
      - terminal: the integer terminalName (e.g. '001075' → 1075)
      - name, lat, long, nbDocks
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    records = []
    for st in root.findall('station'):
        station_id = int(st.find('id').text)
        term_str   = st.find('terminalName').text
        terminal   = int(term_str) if term_str is not None else None
        name       = st.find('name').text
        lat        = float(st.find('lat').text)
        lon        = float(st.find('long').text)
        nb_docks   = int(st.find('nbDocks').text)

        records.append({
            'station_id': station_id,
            'terminal':   terminal,
            'station_name': name,
            'lat':        lat,
            'lon':        lon,
            'nbDocks':    nb_docks
        })

    stations_df = pd.DataFrame.from_records(records)
    return stations_df

def merge_journey_with_stations(journeys_csv: str,
                                stations_df: pd.DataFrame,
                                output_csv: str):
    """
    Load journey_data, then merge station metadata on both departure and arrival.
    """
    # read station numbers as strings so we can clean them
    journeys = pd.read_csv(
        journeys_csv,
        parse_dates=['Start date', 'End date'],
        dtype={'Start station number': str, 'End station number': str}
    )

    # extract leading digits and convert to int (invalid → NaN)
    for col in ['Start station number', 'End station number']:
        journeys[col] = (
            journeys[col]
            .str.extract(r'^(\d+)', expand=False)    # grab digits
            .astype('Int64')                          # pandas nullable integer
        )

    # prepare for departure merge
    dep_meta = stations_df.rename(columns={
        'terminal': 'Start station number',
        'station_name': 'Start station name',
        'lat': 'Start lat',
        'lon': 'Start lon',
        'nbDocks': 'Start nbDocks'
    })[['Start station number', 'Start station name', 'Start lat', 'Start lon', 'Start nbDocks']]

    # prepare for arrival merge
    arr_meta = stations_df.rename(columns={
        'terminal': 'End station number',
        'station_name': 'End station name',
        'lat': 'End lat',
        'lon': 'End lon',
        'nbDocks': 'End nbDocks'
    })[['End station number', 'End station name', 'End lat', 'End lon', 'End nbDocks']]

    # merge on departure station
    journeys = journeys.merge(
        dep_meta,
        on='Start station number',
        how='left'
    )

    # merge on arrival station
    journeys = journeys.merge(
        arr_meta,
        on='End station number',
        how='left'
    )

    # write out
    journeys.to_csv(output_csv, index=False)
    print(f"Merged data written to {output_csv}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse stations.xml and merge with journey data"
    )
    parser.add_argument(
        "--stations-xml", default="raw-data/stations.xml",
        help="Path to stations.xml"
    )
    parser.add_argument(
        "--journeys-csv", default="processed-data/journey_data_2023.csv",
        help="Path to your journey data CSV"
    )
    parser.add_argument(
        "--output-csv", default="processed-data/journey_with_station_meta.csv",
        help="Where to write merged output"
    )
    args = parser.parse_args()

    # 1. Parse stations.xml
    stations_df = parse_stations(args.stations_xml)

    # 2. Merge with journeys
    merge_journey_with_stations(
        journeys_csv=args.journeys_csv,
        stations_df=stations_df,
        output_csv=args.output_csv
    )
