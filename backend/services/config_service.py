from models.config import MachineConfig, MaterialConfig, ToolingConfig
from typing import Dict, Any

def get_machine_config() -> MachineConfig:
    machines = [
        {
            "id": "B55-1 D22",
            "name": "B55-1 D22",
            "type": "milling",
            "specs": {
                "max_spindle_speed": 8000,
                "max_feed_rate": 1000,
                "work_envelope": {
                    "x": 762,
                    "y": 406,
                    "z": 508
                }
            }
        },
        {
            "id": "D45-3 A22", 
            "name": "D45-3 A22",
            "type": "milling",
            "specs": {
                "max_spindle_speed": 12000,
                "max_feed_rate": 1200,
                "work_envelope": {
                    "x": 800,
                    "y": 400,
                    "z": 510
                }
            }
        },
        {
            "id": "cnc-003",
            "name": "DMG Mori NHX-4000",
            "type": "milling", 
            "specs": {
                "max_spindle_speed": 15000,
                "max_feed_rate": 1500,
                "work_envelope": {
                    "x": 700,
                    "y": 500,
                    "z": 450
                }
            }
        }
    ]
    
    return MachineConfig(machines=machines)

def get_material_config() -> MaterialConfig:
    materials = [
        {
            "id": "al6061",
            "name": "鋁合金 6061-T6",
            "type": "aluminum",
            "properties": {
                "hardness": 25,
                "density": 2.7,
                "thermal_conductivity": 167,
                "recommended_speeds": {
                    "min": 2000,
                    "max": 5000
                }
            }
        },
        {
            "id": "steel1018",
            "name": "碳鋼 1018", 
            "type": "steel",
            "properties": {
                "hardness": 35,
                "density": 7.87,
                "thermal_conductivity": 51,
                "recommended_speeds": {
                    "min": 1500,
                    "max": 3500
                }
            }
        },
        {
            "id": "ss304",
            "name": "不鏽鋼 304",
            "type": "stainless_steel",
            "properties": {
                "hardness": 45,
                "density": 8.0,
                "thermal_conductivity": 16,
                "recommended_speeds": {
                    "min": 1000,
                    "max": 2500
                }
            }
        }
    ]
    
    return MaterialConfig(materials=materials)

def get_tooling_config() -> ToolingConfig:
    tools = [
        {
            "id": "endmill_flat",
            "name": "端銑刀 - 平底",
            "type": "end_mill",
            "geometry": "flat",
            "available_diameters": [3, 6, 8, 10, 12, 16, 20],
            "material_compatibility": ["aluminum", "steel", "stainless_steel"],
            "recommended_params": {
                "aluminum": {"speed_multiplier": 1.2, "feed_multiplier": 1.0},
                "steel": {"speed_multiplier": 1.0, "feed_multiplier": 0.8},
                "stainless_steel": {"speed_multiplier": 0.7, "feed_multiplier": 0.6}
            }
        },
        {
            "id": "endmill_ball",
            "name": "端銑刀 - 球頭",
            "type": "end_mill",
            "geometry": "ball",
            "available_diameters": [1, 2, 3, 6, 8, 10, 12, 16],
            "material_compatibility": ["aluminum", "steel", "stainless_steel"],
            "recommended_params": {
                "aluminum": {"speed_multiplier": 1.1, "feed_multiplier": 0.9},
                "steel": {"speed_multiplier": 0.9, "feed_multiplier": 0.7},
                "stainless_steel": {"speed_multiplier": 0.6, "feed_multiplier": 0.5}
            }
        },
        {
            "id": "facemill",
            "name": "面銑刀",
            "type": "face_mill",
            "geometry": "face",
            "available_diameters": [25, 32, 40, 50, 63, 80, 100],
            "material_compatibility": ["aluminum", "steel", "stainless_steel"],
            "recommended_params": {
                "aluminum": {"speed_multiplier": 1.3, "feed_multiplier": 1.2},
                "steel": {"speed_multiplier": 1.0, "feed_multiplier": 1.0},
                "stainless_steel": {"speed_multiplier": 0.8, "feed_multiplier": 0.8}
            }
        }
    ]
    
    return ToolingConfig(tools=tools)

def save_config(config_type: str, config_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "message": f"Configuration for {config_type} saved successfully",
        "config_type": config_type,
        "timestamp": "2024-01-15T10:30:00Z"
    }
