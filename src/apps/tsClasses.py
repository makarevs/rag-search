from pydantic import BaseModel, Field, validator
from typing import List, Union, Dict
import pandas as pd
import numpy as np
from enum import Enum


class Frequency(str, Enum):
    DAILY = "D"
    BUSINESS_DAILY = "B"
    HOURLY = "H"
    MINUTELY = "T"
    SECONDLY = "S"
    MILLISECONDLY = "L"
    MICROSECONDLY = "U"
    NANOSECONDLY = "N"

class BaseMLModel(BaseModel):
    """
    Base model that other ML models will inherit from.
    """

    model_id: str = Field(..., description="Unique model identifier")
    model_version: str = Field(..., description="Version of the model")

    class Config:
        # Change the ORM mode so that we can return instances of this class.
        orm_mode = True


class Scalar(BaseMLModel):
    """
    A simple scalar value required for modelling
    """
    value: Union[int, float] = Field(..., description="The scalar value")

class TimeSeries(BaseMLModel):
    """
    Represents a series of data points indexed in time.
    """

    # Frequency of data
    frequency: Frequency = Field(Frequency.DAILY, description="The frequency of time series points")
    # Data points: Dict of datetime strings or Timestamp objects to float values
    data: Dict[Union[pd.Timestamp, str], float] = Field(..., description="time-indexed data points")

    @validator("data", pre=True)
    def parse_data(cls, data):
        # Parsing each timestamp in data from str to pd.Timestamp
        return {pd.to_datetime(k): v for k, v in data.items()}

    def resample(self, rule):
        """
        Method to resample the time series to a new frequency.
        """
        series = pd.Series(self.data)
        resampled = series.resample(rule=rule).asfreq()
        return TimeSeries(model_id=self.model_id, model_version=self.model_version, frequency=rule, data=dict(resampled))

class RelativeTimeSeries(TimeSeries):
    """
    Represents a relative time series, that is, a time series that is relative to a given base timestamp 
    (e.g. time since event A, etc.).
    """

    base_timestamp: Union[pd.Timestamp, str] = Field(..., description="The base timestamp for the series")

    @validator("base_timestamp", pre=True)
    def parse_base_timestamp(cls, v):
        return pd.to_datetime(v) if isinstance(v, str) else v

    def convert_to_absolute(self):
        """
        Converts relative timestamps in the data to absolute using the base timestamp
        """
        self.data = {self.base_timestamp + pd.Timedelta(k): v for k, v in self.data.items()}
        self.base_timestamp = None
        return self
    
    def prediction_duration(self):
        """Calculate the duration of the prediction period"""
        min_ts = min(self.data.keys())
        max_ts = max(self.data.keys())
        return max_ts - min_ts    
    
    def number_of_periods(self):
        """Determine the number of periods in the observation"""
        return len(self.data)
    
    def prediction_start(self):
        """Return the start of the prediction period"""
        return min(self.data.keys())

    def prediction_end(self):
        """Return the end of the prediction period"""
        return max(self.data.keys())
    
    def compute_time_delta(self):
        """Compute the time delta between predictions."""
        sorted_ts = sorted(self.data.keys())
        # Use List comprehension to find all deltas, and find min (most common)
        min_delta = min([sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)])
        return min_delta

    def is_regular_frequency(self):
        """Check if the frequency of the data is regular."""
        sorted_ts = sorted(self.data.keys())
        min_delta = self.compute_time_delta()
        return all(sorted_ts[i+1] - sorted_ts[i] == min_delta for i in range(len(sorted_ts) - 1))
    
# Let's define a union type for our various time series types
AnyTimeSeries = Union[TimeSeries, RelativeTimeSeries]
