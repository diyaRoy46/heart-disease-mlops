"""Request/response schemas for the prediction API.

Field semantics follow the UCI Heart Disease (Cleveland) codebook.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientData(BaseModel):
    """One patient record. `ca` and `thal` may be omitted (imputed)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "age": 57,
                    "sex": 1,
                    "cp": 4,
                    "trestbps": 140,
                    "chol": 241,
                    "fbs": 0,
                    "restecg": 1,
                    "thalach": 123,
                    "exang": 1,
                    "oldpeak": 0.2,
                    "slope": 2,
                    "ca": 0,
                    "thal": 7,
                }
            ]
        }
    )

    age: float = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        description="Chest pain type: 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic"
    )
    trestbps: float = Field(gt=0, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(gt=0, le=800, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        description="Resting ECG: 0 normal, 1 ST-T abnormality, 2 LV hypertrophy"
    )
    thalach: float = Field(gt=0, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(
        description="Slope of peak exercise ST segment: 1 up, 2 flat, 3 down"
    )
    ca: Literal[0, 1, 2, 3] | None = Field(
        default=None, description="Major vessels colored by fluoroscopy (0-3)"
    )
    thal: Literal[3, 6, 7] | None = Field(
        default=None, description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible defect"
    )


class PredictionResponse(BaseModel):
    prediction: Literal[0, 1] = Field(description="1 = heart disease predicted")
    label: str = Field(description="Human-readable prediction")
    probability: float = Field(ge=0, le=1, description="P(heart disease)")
    model_name: str = Field(description="Model family that served the prediction")
    trained_at: str = Field(description="Training timestamp of the served model")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
