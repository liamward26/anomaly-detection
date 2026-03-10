#!/usr/bin/env python3
import json
import math
import boto3
from datetime import datetime
from typing import Optional, List
import logging

s3 = boto3.client("s3")
logger = logging.getLogger("anomaly_app")


class BaselineManager:
    """
    Maintains a per-channel running baseline using Welford's online algorithm,
    which computes mean and variance incrementally without storing all past data.
    """

    def __init__(
        self,
        bucket: str,
        baseline_key: str = "state/baseline.json",
        log_key: str = "logs/app.log",
        local_log_path: str = "logs/app.log",
    ):
        self.bucket = bucket
        self.baseline_key = baseline_key
        self.log_key = log_key
        self.local_log_path = local_log_path

    def load(self) -> dict:
        try:
            response = s3.get_object(Bucket=self.bucket, Key=self.baseline_key)
            logger.info("Loaded baseline from s3://%s/%s", self.bucket, self.baseline_key)
            return json.loads(response["Body"].read())
        except s3.exceptions.NoSuchKey:
            logger.info("No baseline found yet at s3://%s/%s; starting fresh", self.bucket, self.baseline_key)
            return {}
        except Exception as e:
            logger.exception("Error loading baseline: %s", e)
            print(f"Error loading baseline: {e}")
            raise

    def save(self, baseline: dict):
        try:
            baseline["last_updated"] = datetime.utcnow().isoformat()

            s3.put_object(
                Bucket=self.bucket,
                Key=self.baseline_key,
                Body=json.dumps(baseline, indent=2),
                ContentType="application/json"
            )

            logger.info("Saved baseline to s3://%s/%s", self.bucket, self.baseline_key)

            # Sync a copy of the local log file to S3 whenever baseline.json is pushed.
            try:
                s3.upload_file(self.local_log_path, self.bucket, self.log_key)
                logger.info("Synced log file to s3://%s/%s", self.bucket, self.log_key)
            except Exception as e:
                logger.exception("Error syncing log file to S3: %s", e)
                print(f"Error syncing log file to S3: {e}")

        except Exception as e:
            logger.exception("Error saving baseline: %s", e)
            print(f"Error saving baseline: {e}")
            raise

    def update(self, baseline: dict, channel: str, new_values: List[float]) -> dict:
        """
        Welford's online algorithm for numerically stable mean and variance.
        Each channel tracks: count, mean, M2 (sum of squared deviations).
        Variance = M2 / count, std = sqrt(variance).
        """
        try:
            if channel not in baseline:
                baseline[channel] = {"count": 0, "mean": 0.0, "M2": 0.0}

            state = baseline[channel]

            for value in new_values:
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                delta2 = value - state["mean"]
                state["M2"] += delta * delta2

            # Only compute std once we have enough observations
            if state["count"] >= 2:
                variance = state["M2"] / state["count"]
                state["std"] = math.sqrt(variance)
            else:
                state["std"] = 0.0

            baseline[channel] = state
            logger.info(
                "Updated baseline for channel=%s count=%s mean=%.4f std=%.4f",
                channel,
                state["count"],
                state["mean"],
                state["std"],
            )
            return baseline

        except Exception as e:
            logger.exception("Error updating baseline for channel %s: %s", channel, e)
            print(f"Error updating baseline for channel {channel}: {e}")
            raise

    def get_stats(self, baseline: dict, channel: str) -> Optional[dict]:
        return baseline.get(channel)