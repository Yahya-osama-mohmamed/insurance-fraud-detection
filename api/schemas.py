"""Pydantic request/response models for the fraud-scoring API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]

AGE_BRACKETS = [
    (17, "16 to 17"), (20, "18 to 20"), (25, "21 to 25"), (30, "26 to 30"),
    (35, "31 to 35"), (40, "36 to 40"), (50, "41 to 50"), (65, "51 to 65"),
]


def age_bracket(age: int) -> str:
    """Bin a numeric age into the dataset's AgeOfPolicyHolder vocabulary."""
    for upper, bracket in AGE_BRACKETS:
        if age <= upper:
            return bracket
    return "over 65"


class ClaimInput(BaseModel):
    """One insurance claim, in the raw dataset vocabulary."""

    Month: str = Field(..., description="Accident month, e.g. 'Jan'")
    WeekOfMonth: int = Field(..., ge=1, le=5)
    DayOfWeek: str
    Make: str = Field(..., description="Vehicle make, e.g. 'Honda'")
    AccidentArea: Literal["Urban", "Rural"]
    DayOfWeekClaimed: str
    MonthClaimed: str
    WeekOfMonthClaimed: int = Field(..., ge=1, le=5)
    Sex: Literal["Male", "Female"]
    MaritalStatus: Literal["Single", "Married", "Widow", "Divorced"]
    Age: int | None = Field(None, ge=16, le=100,
                               description="Policy holder age; omit if unknown")
    Fault: Literal["Policy Holder", "Third Party"]
    PolicyType: str | None = Field(
        None, description="DERIVED server-side as '<VehicleCategory> - <BasePolicy>'; "
                          "any client-sent value is overwritten to prevent "
                          "contradictory inputs the model never saw in training")
    VehicleCategory: Literal["Sport", "Utility", "Sedan"]
    VehiclePrice: Literal["less than 20000", "20000 to 29000", "30000 to 39000",
                          "40000 to 59000", "60000 to 69000", "more than 69000"]
    RepNumber: int = Field(..., ge=1, le=16)
    Deductible: int
    DriverRating: int = Field(..., ge=1, le=4)
    Days_Policy_Accident: Literal["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"]
    Days_Policy_Claim: Literal["none", "8 to 15", "15 to 30", "more than 30"]
    PastNumberOfClaims: Literal["none", "1", "2 to 4", "more than 4"]
    AgeOfVehicle: Literal["new", "2 years", "3 years", "4 years",
                          "5 years", "6 years", "7 years", "more than 7"]
    AgeOfPolicyHolder: Literal["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65"] | None = Field(  # noqa: E501
        None, description="DERIVED from Age when Age is given (any client-sent "
                          "value is overwritten); required only when Age is omitted")
    PoliceReportFiled: YesNo
    WitnessPresent: YesNo
    AgentType: Literal["External", "Internal"]
    NumberOfSuppliments: Literal["none", "1 to 2", "3 to 5", "more than 5"]
    AddressChange_Claim: Literal["under 6 months", "1 year", "2 to 3 years",
                                 "4 to 8 years", "no change"]
    NumberOfCars: Literal["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"]
    Year: int = Field(..., ge=1990, le=2030)
    BasePolicy: Literal["Liability", "Collision", "All Perils"]

    @model_validator(mode="after")
    def enforce_consistency(self) -> "ClaimInput":
        """Single source of truth for dependent fields.

        In the training data, AgeOfPolicyHolder is a binning of Age and
        PolicyType is exactly '<VehicleCategory> - <BasePolicy>'. Accepting
        contradictory combinations would feed the model inputs it has never
        seen — so the derived fields are always recomputed here, regardless
        of what the client sent.
        """
        self.PolicyType = f"{self.VehicleCategory} - {self.BasePolicy}"
        if self.Age is not None:
            self.AgeOfPolicyHolder = age_bracket(self.Age)
        elif self.AgeOfPolicyHolder is None:
            raise ValueError("Provide either Age or AgeOfPolicyHolder.")
        return self

    model_config = ConfigDict(json_schema_extra={"example": {
        "Month": "Mar", "WeekOfMonth": 2, "DayOfWeek": "Friday", "Make": "Honda",
        "AccidentArea": "Urban", "DayOfWeekClaimed": "Monday", "MonthClaimed": "Mar",
        "WeekOfMonthClaimed": 3, "Sex": "Male", "MaritalStatus": "Married",
        "Age": 34, "Fault": "Policy Holder", "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan", "VehiclePrice": "20000 to 29000",
        "RepNumber": 7, "Deductible": 400, "DriverRating": 2,
        "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "1", "AgeOfVehicle": "5 years",
        "AgeOfPolicyHolder": "31 to 35", "PoliceReportFiled": "Yes",
        "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
        "NumberOfCars": "1 vehicle", "Year": 1995, "BasePolicy": "Collision",
    }})


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="P(fraud) in [0, 1]")
    flagged: bool = Field(..., description="True if probability >= operating threshold")
    threshold: float = Field(..., description="Operating threshold (max-F2, chosen on training data)")
    top_reasons: list[str] = Field(..., description="Top SHAP drivers, analyst-readable")


class QueueItem(BaseModel):
    received_at: str
    fraud_probability: float
    top_reasons: list[str]


class HealthResponse(BaseModel):
    status: str
    model_version: str
