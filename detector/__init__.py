"""Detector package initializer.
Expose submodules and common classes for easier imports.
"""
from .license_plate_detector import LicensePlateDetector
from .vehicle_detector import VehicleDetector
from .traffic_light_detector import detect_traffic_lights, detect_traffic_lights_with_color, draw_traffic_lights

__all__ = [
    'LicensePlateDetector',
    'VehicleDetector',
    'detect_traffic_lights',
    'detect_traffic_lights_with_color',
    'draw_traffic_lights'
]
