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
        "hyper_params": {
            "use_cnc_knowledge_base": True,
            "percentile_threshold": 0.95,
            "short_threshold": 0.2,
            "ae_thres": 0.1,
            "ap_thres": 0.1,
            "turning_G01_thres": 0.5,
            "pre_turning_thres": 1,
            "multiplier_max": 1.5,
            "multiplier_min": 1.0,
            "multiplier_air": 2.0,
            "apply_finishing": 1,
            "apply_ban_n": 1,
            "multiplier_finishing": 1.2,
            "target_pwc_strategy": "按刀具",
            "max_increase_step": 2000.0,
            "min_air_speed": 0.0,
            "max_air_speed": 48000.0
        },
        "sub_programs": {
            "5601": {
                "function": "內側開粗、上下左右Pocket精修",
                "tool": "",
                "tool_spec": "",
                "finishing": 0,
                "apply_afc": 1,
                "apply_air": 1,
                "apply_turning": 1,
                "multiplier_max": 1.5,
                "ban_n": ["N10", "N20", "N40", "N30", "N50", "N60", "N70", "N80", "N90"],
                "ban_row": []
            },
            "5202": {
                "function": "頂面開粗",
                "tool": "",
                "tool_spec": "",
                "finishing": 0,
                "apply_afc": 1,
                "apply_air": 1,
                "apply_turning": 1,
                "multiplier_max": 1.5,
                "ban_n": [],
                "ban_row": []
            },
            "5214": {
                "function": "耳機孔避位及下避位槽T型槽加工",
                "tool": "",
                "tool_spec": "",
                "finishing": 0,
                "apply_afc": 1,
                "apply_air": 1,
                "apply_turning": 1,
                "multiplier_max": 1.5,
                "ban_n": [],
                "ban_row": []
            },
            "5615": {
                "function": "下避位槽精修&PB裝配區加工",
                "tool": "",
                "tool_spec": "",
                "finishing": 1,
                "apply_afc": 1,
                "apply_air": 1,
                "apply_turning": 1,
                "multiplier_max": 1.2,
                "ban_n": [],
                "ban_row": []
            },
            "5516": {
                "function": "Heatsink-Pocket精修&上下避位槽清角及電池槽加工",
                "tool": "",
                "tool_spec": "",
                "finishing": 1,
                "apply_afc": 1,
                "apply_air": 1,
                "apply_turning": 1,
                "multiplier_max": 1.2,
                "ban_n": [],
                "ban_row": []
            }
        }
    }
    
    return OptimizationConfig(**config)
