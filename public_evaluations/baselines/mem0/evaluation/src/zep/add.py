import argparse
import json
import os
import time
from typing import Optional

from dotenv import load_dotenv
from tqdm import tqdm
from zep_cloud import Message
from zep_cloud.client import Zep
from zep_cloud.core.api_error import ApiError

load_dotenv()


class ZepAdd:
    def __init__(self, data_path=None, rate_limit_delay: float = 1.0, max_retries: int = 5):
        self.zep_client = Zep(api_key=os.getenv("ZEP_API_KEY"))
        self.data_path = data_path
        self.data = None
        self.rate_limit_delay = rate_limit_delay  # Delay between API calls in seconds
        self.max_retries = max_retries  # Maximum number of retries for rate limited requests
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def _make_api_call_with_retry(self, api_call_func, *args, **kwargs):
        """
        Make an API call with exponential backoff retry logic for rate limiting.
        """
        for attempt in range(self.max_retries):
            try:
                result = api_call_func(*args, **kwargs)
                # Add a small delay between successful requests to avoid hitting rate limits
                time.sleep(self.rate_limit_delay)
                return result
            except ApiError as e:
                if e.status_code == 429:  # Rate limit exceeded
                    if attempt < self.max_retries - 1:
                        # Exponential backoff: wait longer with each retry
                        wait_time = (2 ** attempt) * 5  # 5, 10, 20, 40 seconds
                        print(f"Rate limit hit. Waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Max retries ({self.max_retries}) reached. Giving up.")
                        raise
                else:
                    # Re-raise non-rate-limit errors immediately
                    raise
        
    def process_conversation(self, run_id, item, idx):
        conversation = item["conversation"]

        user_id = f"run_id_{run_id}_experiment_user_{idx}"
        session_id = f"run_id_{run_id}_experiment_session_{idx}"

        print("Starting to add memories... for user", user_id)

        # # delete all memories for the two users
        # self.zep_client.user.delete(user_id=user_id)
        # self.zep_client.memory.delete(session_id=session_id)

        # Add user with retry logic
        self._make_api_call_with_retry(
            self.zep_client.user.add, 
            user_id=user_id
        )
        
        # Add session with retry logic
        self._make_api_call_with_retry(
            self.zep_client.memory.add_session,
            user_id=user_id,
            session_id=session_id,
        )

        print("Starting to add memories... for user", user_id)
        for key in tqdm(conversation.keys(), desc=f"Processing user {user_id}"):
            if key in ["speaker_a", "speaker_b"] or "date" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            for chat in tqdm(chats, desc=f"Adding chats for {key}", leave=False):
                # Add memory with retry logic and rate limiting
                self._make_api_call_with_retry(
                    self.zep_client.memory.add,
                    session_id=session_id,
                    messages=[
                        Message(
                            role=chat["speaker"],
                            role_type="user",
                            content=f"{timestamp}: {chat['text']}",
                        )
                    ],
                )

    def process_all_conversations(self, run_id):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        for idx, item in tqdm(enumerate(self.data)):
            if idx < 3:
                continue
            self.process_conversation(run_id, item, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--rate_limit_delay", type=float, default=1.0, help="Delay between API calls in seconds")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for rate limited requests")
    args = parser.parse_args()
    
    zep_add = ZepAdd(
        data_path="../../dataset/locomo10.json",
        rate_limit_delay=args.rate_limit_delay,
        max_retries=args.max_retries
    )
    zep_add.process_all_conversations(args.run_id)
