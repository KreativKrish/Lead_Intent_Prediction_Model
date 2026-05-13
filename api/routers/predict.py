"""Prediction endpoints."""

from typing import List
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.utils.logger import get_logger
from ..dependencies import get_model, get_request_id
from ..schemas import LeadFeatures, PredictionResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    features: LeadFeatures,
    request_id: str = Depends(get_request_id),
    model=Depends(get_model),
) -> PredictionResponse:
    """Make single prediction on lead features.

    Args:
        features: Lead features.
        request_id: Request ID.
        model: Loaded model.

    Returns:
        Prediction response with probability and label.
    """
    try:
        logger.info(f"Making prediction for request {request_id}")

        feature_df = pd.DataFrame([features.dict()])
        prediction = model.predict(feature_df)[0]
        probability = model.predict_proba(feature_df)[0][1]

        logger.info(
            f"Prediction complete: label={prediction}, probability={probability:.4f}"
        )

        return PredictionResponse(
            prediction_id=f"pred_{request_id}",
            probability=float(probability),
            predicted_label=bool(prediction),
            model_version="v1.0.0",
            confidence=max(probability, 1 - probability),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@router.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    features_list: List[LeadFeatures],
    request_id: str = Depends(get_request_id),
    model=Depends(get_model),
) -> List[PredictionResponse]:
    """Make batch predictions on multiple leads.

    Args:
        features_list: List of lead features.
        request_id: Request ID.
        model: Loaded model.

    Returns:
        List of prediction responses.
    """
    try:
        logger.info(f"Making batch predictions for {len(features_list)} leads (request {request_id})")

        results = []
        for i, features in enumerate(features_list):
            feature_df = pd.DataFrame([features.dict()])
            prediction = model.predict(feature_df)[0]
            probability = model.predict_proba(feature_df)[0][1]

            pred_id = f"pred_{request_id}_{i}"
            results.append(
                PredictionResponse(
                    prediction_id=pred_id,
                    probability=float(probability),
                    predicted_label=bool(prediction),
                    model_version="v1.0.0",
                    confidence=max(probability, 1 - probability),
                )
            )

        logger.info(f"Batch predictions complete: {len(results)} predictions")
        return results

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")
