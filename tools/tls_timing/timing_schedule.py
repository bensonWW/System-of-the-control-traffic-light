"""
Timing Schedule Lookup Module

Reads timing schedules from integrated_timing_plan.json and provides
functions to look up the appropriate plan number based on current time
and day of the week.

Usage:
    from timing_schedule import get_plan_for_time
    
    # Using system time
    plan = get_plan_for_time("IJLJW")
    
    # Using specific datetime
    plan = get_plan_for_time("IJLJW", datetime(2026, 1, 17, 14, 30))
"""
import json
import os
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

# Path to integrated timing plan JSON
TIMING_JSON_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "source", "integrated_timing_plan.json"
)


@lru_cache(maxsize=1)
def _load_timing_data() -> List[Dict]:
    """Load and cache the timing plan JSON (only once)."""
    with open(TIMING_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_schedule_for_icid(icid: str, segmenttype: int) -> List[Tuple[time, str]]:
    """
    Get the time schedule for a specific ICID and day type.
    
    Args:
        icid: Intersection Controller ID (e.g., "IJLJW")
        segmenttype: Day type (1=Monday, 7=Sunday)
    
    Returns:
        List of (start_time, plan_id) tuples sorted by time
    """
    data = _load_timing_data()
    
    # Find entries matching icid and segmenttype
    matching = [
        entry for entry in data
        if entry.get("icid") == icid and entry.get("segmenttype") == segmenttype
    ]
    
    if not matching:
        # Fallback to any entry for this icid (use first segmenttype found)
        matching = [entry for entry in data if entry.get("icid") == icid]
        if not matching:
            return []
    
    # Use the first matching entry's subsegment
    entry = matching[0]
    subsegments = entry.get("subsegment", [])
    
    schedule = []
    for seg in subsegments:
        time_str = seg.get("time", "0000")
        plan_id = str(seg.get("planid（SeqNo）", "1"))
        
        # Parse time string "HHMM" to time object
        try:
            hour = int(time_str[:2])
            minute = int(time_str[2:4]) if len(time_str) >= 4 else 0
            schedule.append((time(hour, minute), plan_id))
        except (ValueError, IndexError):
            continue
    
    # Sort by time
    schedule.sort(key=lambda x: (x[0].hour, x[0].minute))
    return schedule


def get_plan_for_time(icid: str, dt: Optional[datetime] = None) -> str:
    """
    Look up the appropriate timing plan number for a given intersection and time.
    
    Args:
        icid: Intersection Controller ID (e.g., "IJLJW")
        dt: datetime to check (defaults to system time if None)
    
    Returns:
        Plan number as string (e.g., "01", "09")
    """
    if dt is None:
        dt = datetime.now()
    
    # segmenttype: 1=Monday(0), 2=Tuesday(1), ..., 7=Sunday(6)
    # Python weekday(): 0=Monday, 6=Sunday
    segmenttype = dt.weekday() + 1
    
    schedule = _get_schedule_for_icid(icid, segmenttype)
    
    if not schedule:
        return "01"  # Default fallback
    
    # Find the applicable plan
    current_time = dt.time()
    plan = schedule[0][1]  # Default to first entry
    
    for start_time, plan_num in schedule:
        if current_time >= start_time:
            plan = plan_num
        else:
            break
    
    return plan


def get_timing_details(icid: str, plan_id: str, segmenttype: int = 1) -> Optional[Dict]:
    """
    Get the detailed timing information for a specific plan.
    
    Args:
        icid: Intersection Controller ID
        plan_id: Plan number (e.g., "9" or "09")
        segmenttype: Day type (1-7)
    
    Returns:
        Dict with cycletime, subplan (phases), etc. or None if not found
    """
    data = _load_timing_data()
    
    # Normalize plan_id
    plan_num = float(plan_id.lstrip("0") or "0")
    
    for entry in data:
        if (entry.get("icid") == icid and 
            entry.get("planid（SeqNo）") == plan_num and
            entry.get("segmenttype") == segmenttype):
            return {
                "icid": icid,
                "plan_id": plan_id,
                "cycletime": entry.get("cycletime"),
                "offset": entry.get("offset"),
                "subplan": entry.get("subplan", []),
                "icname": entry.get("icname_plan"),
            }
    
    return None


def get_all_icids() -> List[str]:
    """Get list of all unique ICIDs in the timing data."""
    data = _load_timing_data()
    return list(set(entry.get("icid", "") for entry in data if entry.get("icid")))


def get_schedule_table(icid: str) -> Dict[int, List[Tuple[time, str]]]:
    """
    Get the complete weekly schedule for an ICID.
    
    Returns:
        Dict mapping segmenttype (1-7) to list of (time, plan_id) tuples
    """
    result = {}
    for day in range(1, 8):
        schedule = _get_schedule_for_icid(icid, day)
        if schedule:
            result[day] = schedule
    return result


if __name__ == "__main__":
    # Test with current system time
    print("=" * 60)
    print("Timing Schedule Test (using system time)")
    print("=" * 60)
    
    now = datetime.now()
    print(f"\nSystem Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Day of week: {now.strftime('%A')} (segmenttype={now.weekday() + 1})")
    
    # Test IJLJW
    icid = "IJLJW"
    plan = get_plan_for_time(icid)
    print(f"\nPlan for {icid}: {plan}")
    
    # Show schedule for today
    segmenttype = now.weekday() + 1
    schedule = _get_schedule_for_icid(icid, segmenttype)
    print(f"\nToday's schedule for {icid}:")
    for start_time, plan_num in schedule:
        print(f"  {start_time.strftime('%H:%M')} -> Plan {plan_num}")
    
    # Get timing details
    details = get_timing_details(icid, plan, segmenttype)
    if details:
        print(f"\nTiming details for Plan {plan}:")
        print(f"  Cycle time: {details['cycletime']}s")
        print(f"  Phases:")
        for sp in details.get("subplan", []):
            print(f"    Phase {sp.get('subphaseid')}: G={sp.get('green')}s, Y={sp.get('yellow')}s, R={sp.get('allred')}s")
