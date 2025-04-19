# stations/parse_stations.py

import xml.etree.ElementTree as ET
import pandas as pd

def parse_station_xml(xml_file, output_csv="stations_geolocated.csv"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    stations = []
    for station in root.findall('station'):
        try:
            name = station.find('name').text.strip()
            lat = float(station.find('lat').text.strip())
            long = float(station.find('long').text.strip())
            stations.append({"name": name, "latitude": lat, "longitude": long})
        except Exception as e:
            print("Skipped a station due to missing data:", e)

    df = pd.DataFrame(stations)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} stations to {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python parse_stations.py <stations.xml>")
    else:
        parse_station_xml(sys.argv[1])
