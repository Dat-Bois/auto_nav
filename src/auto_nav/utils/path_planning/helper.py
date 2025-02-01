import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def load_map(map_path: Path):
    return np.loadtxt(map_path, delimiter=',')

def to_ecef(lat, lon, alt):
    '''
    Altitude in meters
    Output in km for x, y, z
    '''
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f ** 2
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return [x/1000, y/1000, z/1000]

def three_way_angle(point1, point2, point3):
    '''
    Returns the 2d-angle between the three points
    '''
    point1 = np.array(point1[:2])
    point2 = np.array(point2[:2])
    point3 = np.array(point3[:2])
    v1 = point1 - point2
    v2 = point3 - point2
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    return np.degrees(angle)

if __name__ == '__main__':
    map_path = Path(__file__).parent / 'map.txt'
    map = load_map(map_path)
    map = [to_ecef(point[0], point[1], 100) for point in map]
    print(three_way_angle(map[0], map[1], map[2]))