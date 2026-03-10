# app.py
import io
import json
import os
import boto3
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, Request, HTTPException
from baseline import BaselineManager
from processor import process_file
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

app = FastAPI(title="Anomaly Detection Pipeline")

# Configure local file logging for the FastAPI service.
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "app.log"

logger = logging.getLogger("anomaly_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=2)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

s3 = boto3.client("s3")
BUCKET_NAME = os.environ["BUCKET_NAME"]

# ── SNS subscription confirmation + message handler ──────────────────────────

@app.post("/notify")
async def handle_sns(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        msg_type = request.headers.get("x-amz-sns-message-type")

        logger.info("Received SNS message type=%s", msg_type)

        # SNS sends a SubscriptionConfirmation before it will deliver any messages.
        # Visiting the SubscribeURL confirms the subscription.
        if msg_type == "SubscriptionConfirmation":
            confirm_url = body["SubscribeURL"]
            requests.get(confirm_url, timeout=10)
            logger.info("Confirmed SNS subscription")
            return {"status": "confirmed"}

        if msg_type == "Notification":
            # The SNS message body contains the S3 event as a JSON string.
            s3_event = json.loads(body["Message"])
            for record in s3_event.get("Records", []):
                key = record["s3"]["object"]["key"]
                if key.startswith("raw/") and key.endswith(".csv"):
                    logger.info("Received new raw CSV file: %s", key)
                    background_tasks.add_task(process_file, BUCKET_NAME, key)

        return {"status": "ok"}

    except Exception as e:
        logger.exception("Error handling SNS notification: %s", e)
        print(f"Error handling SNS notification: {e}")
        raise HTTPException(status_code=500, detail="Failed to handle SNS notification")


# ── Query endpoints ───────────────────────────────────────────────────────────

@app.get("/anomalies/recent")
def get_recent_anomalies(limit: int = 50):
    """Return rows flagged as anomalies across the 10 most recent processed files."""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        keys = sorted(
            [
                obj["Key"]
                for page in pages
                for obj in page.get("Contents", [])
                if obj["Key"].endswith(".csv")
            ],
            reverse=True,
        )[:10]

        all_anomalies = []
        for key in keys:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            if "anomaly" in df.columns:
                flagged = df[df["anomaly"] == True].copy()
                flagged["source_file"] = key
                all_anomalies.append(flagged)

        if not all_anomalies:
            return {"count": 0, "anomalies": []}

        combined = pd.concat(all_anomalies).head(limit)
        return {"count": len(combined), "anomalies": combined.to_dict(orient="records")}

    except Exception as e:
        logger.exception("Error fetching recent anomalies: %s", e)
        print(f"Error fetching recent anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent anomalies")


@app.get("/anomalies/summary")
def get_anomaly_summary():
    """Aggregate anomaly rates across all processed files using their summary JSONs."""
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET_NAME, Prefix="processed/")

        summaries = []
        for page in pages:
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("_summary.json"):
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=obj["Key"])
                    summaries.append(json.loads(response["Body"].read()))

        if not summaries:
            return {"message": "No processed files yet."}

        total_rows = sum(s["total_rows"] for s in summaries)
        total_anomalies = sum(s["anomaly_count"] for s in summaries)

        return {
            "files_processed": len(summaries),
            "total_rows_scored": total_rows,
            "total_anomalies": total_anomalies,
            "overall_anomaly_rate": round(total_anomalies / total_rows, 4) if total_rows > 0 else 0,
            "most_recent": sorted(summaries, key=lambda x: x["processed_at"], reverse=True)[:5],
        }

    except Exception as e:
        logger.exception("Error fetching anomaly summary: %s", e)
        print(f"Error fetching anomaly summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomaly summary")


@app.get("/baseline/current")
def get_current_baseline():
    """Show the current per-channel statistics the detector is working from."""
    try:
        baseline_mgr = BaselineManager(bucket=BUCKET_NAME)
        baseline = baseline_mgr.load()

        channels = {}
        for channel, stats in baseline.items():
            if channel == "last_updated":
                continue
            channels[channel] = {
                "observations": stats["count"],
                "mean": round(stats["mean"], 4),
                "std": round(stats.get("std", 0.0), 4),
                "baseline_mature": stats["count"] >= 30,
            }

        return {
            "last_updated": baseline.get("last_updated"),
            "channels": channels,
        }

    except Exception as e:
        logger.exception("Error loading baseline: %s", e)
        print(f"Error loading baseline: {e}")
        raise HTTPException(status_code=500, detail="Failed to load baseline")


@app.get("/health")
def health():
    return {"status": "ok", "bucket": BUCKET_NAME, "timestamp": datetime.utcnow().isoformat()}