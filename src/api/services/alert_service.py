"""
Service for managing study alerts and notifications.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ...core.config import get_settings


class AlertService:
    """Service for managing study alerts."""
    
    def __init__(self):
        """Initialize alert service."""
        self.settings = get_settings()
        self.alerts_file = Path("alerts_config.json")
        self._load_alerts()
    
    def _load_alerts(self) -> Dict:
        """Load alerts configuration."""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    return json.load(f)
            except:
                return {"alerts": []}
        return {"alerts": []}
    
    def _save_alerts(self, alerts_data: Dict):
        """Save alerts configuration."""
        with open(self.alerts_file, 'w') as f:
            json.dump(alerts_data, f, indent=2, default=str)
    
    def create_alert(
        self,
        cancer_type: str,
        search_terms: Optional[List[str]] = None,
        frequency: str = "daily",
        enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Create a new alert.
        
        Args:
            cancer_type: Cancer type to monitor
            search_terms: Additional search terms
            frequency: Alert frequency (daily, weekly, monthly)
            enabled: Whether alert is enabled
            
        Returns:
            Alert configuration
        """
        alerts_data = self._load_alerts()
        
        alert = {
            "alert_id": f"alert_{len(alerts_data.get('alerts', [])) + 1}",
            "cancer_type": cancer_type,
            "search_terms": search_terms or [],
            "frequency": frequency,
            "enabled": enabled,
            "created_at": datetime.now().isoformat(),
            "last_checked": None,
            "matches_found": 0
        }
        
        alerts_data.setdefault("alerts", []).append(alert)
        self._save_alerts(alerts_data)
        
        return alert
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts."""
        alerts_data = self._load_alerts()
        return alerts_data.get("alerts", [])
    
    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an alert."""
        alerts_data = self._load_alerts()
        
        for alert in alerts_data.get("alerts", []):
            if alert["alert_id"] == alert_id:
                alert.update(updates)
                self._save_alerts(alerts_data)
                return alert
        
        raise ValueError(f"Alert {alert_id} not found")
    
    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert."""
        alerts_data = self._load_alerts()
        
        alerts = alerts_data.get("alerts", [])
        alerts_data["alerts"] = [a for a in alerts if a["alert_id"] != alert_id]
        
        self._save_alerts(alerts_data)
        return True
    
    def check_alerts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Check all enabled alerts for new studies.
        
        Returns:
            Dictionary mapping alert IDs to new studies
        """
        from .literature_search_service import LiteratureSearchService
        
        search_service = LiteratureSearchService()
        alerts_data = self._load_alerts()
        results = {}
        
        for alert in alerts_data.get("alerts", []):
            if not alert.get("enabled", True):
                continue
            
            # Determine days back based on frequency
            frequency_days = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30
            }
            days_back = frequency_days.get(alert["frequency"], 7)
            
            # Search for new studies
            new_studies = search_service.search_radiation_oncology(
                cancer_type=alert["cancer_type"],
                days_back=days_back,
                max_results=50
            )
            
            # Update last checked
            alert["last_checked"] = datetime.now().isoformat()
            alert["matches_found"] = len(new_studies)
            
            results[alert["alert_id"]] = new_studies
        
        self._save_alerts(alerts_data)
        return results
