from models.optimization import OptimizationRequest, OptimizationResults, OptimizationConfig
from datetime import datetime
from typing import Optional
import uuid
import asyncio

async def run_optimization(optimization_data: OptimizationRequest) -> dict:
    optimization_id = f"OPT-{str(uuid.uuid4())[:8].upper()}"
    
    await asyncio.sleep(2)
    
    return {
        "optimization_id": optimization_id,
        "status": "completed",
        "message": "Optimization completed successfully"
    }

def get_optimization_results(optimization_id: str) -> Optional[OptimizationResults]:
    mock_results = {
        "optimization_id": optimization_id,
        "status": "completed",
        "summary": {
            "time_reduction": "-23%",
            "time_saved": "10.4 分鐘",
            "tool_life_improvement": "+15%",
            "tool_life_parts": "約多 150 件",
            "quality_score": "98.5%",
            "surface_roughness": "Ra 0.8",
            "cost_savings": "$127",
            "cost_per_part": "每件"
        },
        "charts": {
            "machining_time": {
                "original": 45.2,
                "optimized": 34.8,
                "improvement": 23
            },
            "tool_wear": {
                "original_life": 1000,
                "optimized_life": 1150,
                "improvement": 15
            }
        },
        "code_diff": {
            "original": [
                "G90 G54 G17 G49 G40 G80",
                "T1 M6",
                "G43 H1 Z25.",
                "S3000 M3",
                "G00 X0 Y0"
            ],
            "optimized": [
                "G90 G54 G17 G49 G40 G80",
                "T1 M6", 
                "G43 H1 Z25.",
                "S3500 M3",
                "G00 X0 Y0"
            ],
            "changes": [
                {"line": 4, "type": "modified", "old": "S3000 M3", "new": "S3500 M3"}
            ]
        },
        "simulation": {
            "3d_model_url": "/static/simulation/model.obj",
            "animation_data": "simulation_keyframes.json",
            "total_time": "34.8 minutes",
            "steps": 1250
        },
        "validation": {
            "safety_checks": [
                {"name": "所有安全檢查通過", "status": "passed"},
                {"name": "刀具碰撞分析：清除", "status": "passed"}
            ],
            "quality_checks": [
                {"name": "表面光潔度在公差範圍內", "status": "passed"},
                {"name": "尺寸精度：±0.02mm", "status": "passed"}
            ]
        },
        "created": datetime.utcnow().isoformat() + "Z"
    }
    
    return OptimizationResults(**mock_results)

def get_optimization_config() -> OptimizationConfig:
    config = {
        "machine": {
            "spindle_speed": {"min": 2000, "max": 5000, "default": 3000, "unit": "RPM"},
            "feed_rate": {"min": 300, "max": 800, "default": 500, "unit": "mm/min"},
            "max_spindle_speed": 5000,
            "max_feed_rate": 1000
        },
        "material": {
            "types": [
                {"id": "al6061", "name": "鋁合金 6061-T6", "hardness": 25},
                {"id": "steel1018", "name": "碳鋼 1018", "hardness": 35},
                {"id": "ss304", "name": "不鏽鋼 304", "hardness": 45}
            ],
            "default": "al6061"
        },
        "tooling": {
            "types": [
                {"id": "endmill_flat", "name": "端銑刀 - 平底", "diameter_range": [3, 20]},
                {"id": "endmill_ball", "name": "端銑刀 - 球頭", "diameter_range": [1, 16]},
                {"id": "facemill", "name": "面銑刀", "diameter_range": [25, 100]}
            ],
            "default_diameter": 6
        },
        "optimization": {
            "objectives": [
                {"id": "time", "name": "最小化加工時間", "priority": "high"},
                {"id": "quality", "name": "優化表面品質", "priority": "medium"},
                {"id": "tool_life", "name": "延長刀具壽命", "priority": "medium"}
            ]
        },
        "safety": {
            "max_spindle_speed": 5000,
            "max_feed_rate": 1000,
            "collision_detection": True,
            "emergency_stop": True
        }
    }
    
    return OptimizationConfig(**config)
