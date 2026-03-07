from fastapi import APIRouter
from typing import Dict

router = APIRouter()

PROCESS_METADATA = {
    "turning": {
        "name": "CNC Turning",
        "description": "Rotating workpiece with stationary cutting tool",
        "icon": "🔄",
        "required_fields": ["spindle_speed", "feed_rate", "depth_of_cut", "tool_diameter"],
        "typical_ranges": {
            "spindle_speed": {"min": 100, "max": 4000, "unit": "RPM"},
            "feed_rate":     {"min": 0.05, "max": 0.8, "unit": "mm/rev"},
            "depth_of_cut":  {"min": 0.1, "max": 8.0, "unit": "mm"},
        }
    },
    "milling": {
        "name": "CNC Milling",
        "description": "Rotating multi-point cutter with moving workpiece",
        "icon": "⚙️",
        "required_fields": ["spindle_speed", "feed_rate", "depth_of_cut", "width_of_cut", "tool_diameter"],
        "typical_ranges": {
            "spindle_speed": {"min": 200, "max": 8000, "unit": "RPM"},
            "feed_rate":     {"min": 0.02, "max": 0.4, "unit": "mm/tooth"},
            "depth_of_cut":  {"min": 0.1, "max": 5.0, "unit": "mm"},
            "width_of_cut":  {"min": 0.5, "max": 50.0, "unit": "mm"},
        }
    },
    "drilling": {
        "name": "Drilling",
        "description": "Creating cylindrical holes with a rotating drill",
        "icon": "🔩",
        "required_fields": ["spindle_speed", "feed_rate", "depth_of_cut", "tool_diameter"],
        "typical_ranges": {
            "spindle_speed": {"min": 100, "max": 5000, "unit": "RPM"},
            "feed_rate":     {"min": 0.02, "max": 0.5, "unit": "mm/rev"},
            "depth_of_cut":  {"min": 1.0, "max": 100.0, "unit": "mm"},
        }
    },
    "grinding": {
        "name": "Grinding",
        "description": "Abrasive machining for high precision surface finish",
        "icon": "✨",
        "required_fields": ["spindle_speed", "feed_rate", "depth_of_cut", "width_of_cut"],
        "typical_ranges": {
            "spindle_speed": {"min": 1000, "max": 30000, "unit": "RPM"},
            "feed_rate":     {"min": 0.002, "max": 0.05, "unit": "mm/rev"},
            "depth_of_cut":  {"min": 0.001, "max": 0.5, "unit": "mm"},
            "width_of_cut":  {"min": 1, "max": 50, "unit": "mm"},
        }
    }
}

@router.get("/")
def get_processes() -> Dict:
    return PROCESS_METADATA

@router.get("/{process_type}")
def get_process(process_type: str):
    if process_type not in PROCESS_METADATA:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Process '{process_type}' not found")
    return PROCESS_METADATA[process_type]