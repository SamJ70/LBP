from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ProcessType(str, Enum):
    TURNING  = "turning"
    MILLING  = "milling"
    DRILLING = "drilling"
    GRINDING = "grinding"


class MaterialType(str, Enum):
    ALUMINUM        = "aluminum"
    STEEL_MILD      = "steel_mild"
    STEEL_STAINLESS = "steel_stainless"
    CAST_IRON       = "cast_iron"
    TITANIUM        = "titanium"
    COPPER          = "copper"


class ToolMaterial(str, Enum):
    HSS     = "hss"
    CARBIDE = "carbide"
    CERAMIC = "ceramic"
    CBN     = "cbn"
    DIAMOND = "diamond"


class MachiningInput(BaseModel):
    process_type:  ProcessType  = Field(..., description="Type of machining process")
    material:      MaterialType = Field(..., description="Workpiece material")
    tool_material: ToolMaterial = Field(..., description="Cutting tool material")
    spindle_speed: float = Field(..., gt=0, description="Spindle speed in RPM")
    feed_rate:     float = Field(..., gt=0, description="Feed rate in mm/rev or mm/tooth")
    depth_of_cut:  float = Field(..., gt=0, description="Depth of cut in mm")
    width_of_cut:  Optional[float] = Field(None, description="Width of cut in mm (milling/grinding)")
    tool_diameter: Optional[float] = Field(None, description="Tool diameter in mm")
    coolant_used:  bool  = Field(False, description="Whether coolant is used")
    model_id:      str   = Field("physics_based", description="Prediction engine to use")


class PredictionOutput(BaseModel):
    energy_consumption: float = Field(..., description="Estimated power in Watts")
    surface_roughness:  float = Field(..., description="Estimated Ra in micrometers")
    tool_wear_rate:     float = Field(..., description="Estimated tool wear rate mm/min")
    mrr:                float = Field(..., description="Material Removal Rate mm³/min")
    confidence_score:   float = Field(..., description="Model confidence 0-1")


class OptimizationResult(BaseModel):
    original_params:        MachiningInput
    original_predictions:   PredictionOutput
    optimized_params:       Dict[str, Any]
    optimized_predictions:  PredictionOutput
    energy_savings_percent: float
    quality_maintained:     bool
    optimization_notes:     List[str]
    model_used:             str
    ai_advice:              Optional[str] = None
    optimization_method:    Optional[str] = None


class ModelInfo(BaseModel):
    id:                 str
    name:               str
    description:        str
    type:               str
    available:          bool
    accuracy_metrics:   Optional[Dict[str, Any]] = None
    supported_processes: List[str]