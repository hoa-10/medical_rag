from typing import List, Dict, Optional
import json
import os
from datetime import datetime

# This class is used to manage the conversation history and save it to a file.
class ConservationMemory:
    def __init__(self, file_path= "conservation_history.json",max_history_length = 1000):
        self.file_path = file_path
        self.max_history_length = max_history_length
        self.history = self._load_history()
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
    def get_all_history(self):
        return self.history

    def _load_history(self) -> List[Dict[str, str]]:
        """"load conservation history from the file if it exit"""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading history : {e}")
                return []
        return []
    
    def save_history(self):
        """Save conversation history to file"""
        try:
            if len(self.history) > self.max_history_length:
                self.history = self.history[-self.max_history_length:]
            with open(self.file_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"Errors saving history : {e}")

    def add_interaction(self, user_message: str, bot_response: str, metadata: Dict = None):
        """add new interation to the history"""
        if metadata is None:
            metadata = {}
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            "user": user_message,
            "bot": bot_response,
            'metadata': metadata,
        })
        
        self.save_history()
    
    def get_recent_history(self, num_interactions: int = 5) -> List[Dict[str, str]]:
        """Get the most recent interactions"""
        return self.history[-num_interactions:] if len(self.history) > 0 else []

    def get_session_history(self, session_id: str = None) -> list[dict]:
        '''GEt history from specific session'''
        if session_id == None:
            session_id = self.session_id
        return [interaction for interaction in self.history
                if interaction.get("session_id") == session_id]
    
    def clear_current_session(self):
        """Clear only the current session history"""
        self.history = [interaction for interaction in self.history 
                       if interaction.get("session_id") != self.session_id]
        self.save_history()
    def clear_history(self):
        """Clear the history conservation"""
        self.history = []
        self.save_history()