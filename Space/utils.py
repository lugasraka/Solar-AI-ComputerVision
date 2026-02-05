"""
SolarVision AI - Utilities
Auto-shutdown timer and helper functions
"""

import threading
import time
from datetime import datetime, timedelta

class AutoShutdownTimer:
    """
    Auto-shutdown timer for Gradio demo
    Resets on user activity, shuts down after inactivity period
    """
    
    def __init__(self, timeout_minutes=30, warning_minutes=25):
        """
        Initialize auto-shutdown timer
        
        Args:
            timeout_minutes: Minutes of inactivity before shutdown
            warning_minutes: Minutes before timeout to show warning
        """
        self.timeout_minutes = timeout_minutes
        self.warning_minutes = warning_minutes
        self.last_activity = datetime.now()
        self.is_running = False
        self.timer_thread = None
        self.warning_callback = None
        self.shutdown_callback = None
        
    def start(self, warning_callback=None, shutdown_callback=None):
        """Start the auto-shutdown timer"""
        self.warning_callback = warning_callback
        self.shutdown_callback = shutdown_callback
        self.is_running = True
        self.last_activity = datetime.now()
        
        self.timer_thread = threading.Thread(target=self._monitor, daemon=True)
        self.timer_thread.start()
        print(f"[INFO] Auto-shutdown timer started: {self.timeout_minutes} minutes")
    
    def _monitor(self):
        """Monitor thread that checks for inactivity"""
        while self.is_running:
            time.sleep(10)  # Check every 10 seconds
            
            if not self.is_running:
                break
            
            elapsed = (datetime.now() - self.last_activity).total_seconds() / 60
            remaining = self.timeout_minutes - elapsed
            
            # Trigger warning
            if remaining <= (self.timeout_minutes - self.warning_minutes) and remaining > 0:
                if self.warning_callback:
                    self.warning_callback(f"Warning: Demo will auto-close in {int(remaining)} minutes")
            
            # Trigger shutdown
            if elapsed >= self.timeout_minutes:
                print("[INFO] Auto-shutdown triggered due to inactivity")
                if self.shutdown_callback:
                    self.shutdown_callback()
                self.is_running = False
                break
    
    def reset(self):
        """Reset the timer (call on user activity)"""
        self.last_activity = datetime.now()
    
    def get_remaining_time(self):
        """Get remaining time in minutes"""
        elapsed = (datetime.now() - self.last_activity).total_seconds() / 60
        remaining = max(0, self.timeout_minutes - elapsed)
        return remaining
    
    def get_status_text(self):
        """Get formatted status text for display"""
        remaining = self.get_remaining_time()
        minutes = int(remaining)
        seconds = int((remaining - minutes) * 60)
        return f"Auto-close in: {minutes:02d}:{seconds:02d}"
    
    def stop(self):
        """Stop the timer"""
        self.is_running = False
        if self.timer_thread:
            self.timer_thread.join(timeout=1)


class ProgressTracker:
    """Track and report progress for batch operations"""
    
    def __init__(self, total_items, update_callback=None):
        """
        Initialize progress tracker
        
        Args:
            total_items: Total number of items to process
            update_callback: Function to call with progress updates (current, total, percentage)
        """
        self.total_items = total_items
        self.current = 0
        self.update_callback = update_callback
        self.start_time = time.time()
    
    def update(self, increment=1):
        """Update progress"""
        self.current += increment
        percentage = int((self.current / self.total_items) * 100)
        
        if self.update_callback:
            self.update_callback(self.current, self.total_items, percentage)
        
        return percentage
    
    def get_eta(self):
        """Get estimated time remaining in seconds"""
        if self.current == 0:
            return None
        
        elapsed = time.time() - self.start_time
        rate = self.current / elapsed
        remaining_items = self.total_items - self.current
        eta = remaining_items / rate
        
        return eta
    
    def get_progress_text(self):
        """Get formatted progress text"""
        percentage = int((self.current / self.total_items) * 100)
        eta = self.get_eta()
        
        if eta:
            eta_min = int(eta / 60)
            eta_sec = int(eta % 60)
            return f"Progress: {self.current}/{self.total_items} ({percentage}%) - ETA: {eta_min}:{eta_sec:02d}"
        else:
            return f"Progress: {self.current}/{self.total_items} ({percentage}%)"


def format_confidence(confidence):
    """Format confidence score for display"""
    if confidence >= 0.9:
        return f"{confidence:.1%} (High)"
    elif confidence >= 0.7:
        return f"{confidence:.1%} (Medium)"
    else:
        return f"{confidence:.1%} (Low)"


def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence >= 0.9:
        return "#27ae60"  # Green
    elif confidence >= 0.7:
        return "#f39c12"  # Orange
    else:
        return "#e74c3c"  # Red


def create_class_badge(class_name, confidence=None):
    """Create HTML badge for class display"""
    colors = {
        'Bird-drop': '#3498db',
        'Clean': '#27ae60',
        'Dusty': '#f39c12',
        'Electrical-damage': '#e74c3c',
        'Physical-Damage': '#9b59b6',
        'Snow-Covered': '#95a5a6'
    }
    
    color = colors.get(class_name, '#34495e')
    
    if confidence:
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{class_name} ({confidence:.1%})</span>'
    else:
        return f'<span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{class_name}</span>'


if __name__ == '__main__':
    # Test timer
    timer = AutoShutdownTimer(timeout_minutes=0.1, warning_minutes=0.05)
    
    def warning(msg):
        print(f"[WARNING] {msg}")
    
    def shutdown():
        print("[SHUTDOWN] Demo closing...")
    
    timer.start(warning_callback=warning, shutdown_callback=shutdown)
    
    # Simulate activity
    for i in range(5):
        time.sleep(2)
        timer.reset()
        print(f"Activity {i+1}, remaining: {timer.get_remaining_time():.1f} min")
    
    # Let it timeout
    print("Waiting for timeout...")
    time.sleep(10)
