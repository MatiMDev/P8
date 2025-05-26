# Available commands for the autonomous driving agent
AVAILABLE_COMMANDS = [
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "stop",
    "slow_down",
    "speed_up",
    "park",
    "follow_lane",
    "change_lane"
]

# Waypoint constraints
WAYPOINT_CONSTRAINTS = {
    "max_distance": 50.0,  # Maximum distance in meters for any waypoint
    "min_distance": 0.5,   # Minimum distance in meters for any waypoint
    "max_angle": 90.0,     # Maximum angle in degrees for any turn
    "min_angle": 5.0,      # Minimum angle in degrees for any turn
    "max_speed": 30.0,     # Maximum speed in m/s
    "min_speed": 0.0       # Minimum speed in m/s
}

# System prompt template
SYSTEM_PROMPT = """You are an assistant for an autonomous driving agent. Given a front-view camera image and a user instruction, return a structured navigation plan.

Available commands: {commands}

Waypoint constraints:
- Maximum distance: {max_distance} meters
- Minimum distance: {min_distance} meters
- Maximum turn angle: {max_angle} degrees
- Minimum turn angle: {min_angle} degrees
- Maximum speed: {max_speed} m/s
- Minimum speed: {min_speed} m/s

Ensure all waypoints and actions comply with these constraints.""" 