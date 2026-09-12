import math
import numpy as np

def global_to_local(ego_x, ego_y, ego_vx, ego_vy, target_x, target_y, target_vx, target_vy):
    """
    Transforms target's position and velocity into the ego agent's local coordinate frame.
    In the local frame:
      - Ego is at (0, 0)
      - The positive X-axis points in the direction of ego's velocity vector (ego_vx, ego_vy)
      
    Args:
        ego_x, ego_y: Ego agent's position
        ego_vx, ego_vy: Ego agent's velocity vector (determines heading)
        target_x, target_y: Target agent's position
        target_vx, target_vy: Target agent's velocity vector
        
    Returns:
        local_x, local_y, local_vx, local_vy
    """
    # 1. Translation
    dx = target_x - ego_x
    dy = target_y - ego_y
    
    # 2. Rotation
    # Calculate ego heading. If ego is stationary, assume heading is along positive X (or keep previous heading, but here we default to 0)
    if abs(ego_vx) < 1e-5 and abs(ego_vy) < 1e-5:
        theta = 0.0
    else:
        theta = math.atan2(ego_vy, ego_vx)
        
    # Rotate by -theta
    cos_theta = math.cos(-theta)
    sin_theta = math.sin(-theta)
    
    local_x = dx * cos_theta - dy * sin_theta
    local_y = dx * sin_theta + dy * cos_theta
    
    local_vx = target_vx * cos_theta - target_vy * sin_theta
    local_vy = target_vx * sin_theta + target_vy * cos_theta
    
    return local_x, local_y, local_vx, local_vy

def generate_lidar(ego_x, ego_y, ego_vx, ego_vy, obstacles, num_bins=16, max_range=15.0):
    """
    Generates a simulated 2D LiDAR distance grid for objects in blind spots.
    
    Args:
        ego_x, ego_y: Ego position
        ego_vx, ego_vy: Ego velocity (for orientation)
        obstacles: list of tuples (x, y) for agents outside FOV (gray/black agents)
        num_bins: number of polar sectors
        max_range: maximum distance the LiDAR can detect. Defaults to 15.0 meters.
        
    Returns:
        numpy array of shape (num_bins,) containing distances to closest obstacle in each bin.
        If a bin is empty, the distance is max_range.
    """
    lidar = np.full(num_bins, max_range, dtype=np.float32)
    
    if not obstacles:
        return lidar
        
    if abs(ego_vx) < 1e-5 and abs(ego_vy) < 1e-5:
        theta = 0.0
    else:
        theta = math.atan2(ego_vy, ego_vx)
        
    bin_size = (2 * math.pi) / num_bins
    
    for (ox, oy) in obstacles:
        dx = ox - ego_x
        dy = oy - ego_y
        dist = math.hypot(dx, dy)
        
        if dist > max_range:
            continue
            
        # Calculate angle of obstacle relative to ego's heading
        # math.atan2 gives angle in [-pi, pi]
        abs_angle = math.atan2(dy, dx)
        rel_angle = (abs_angle - theta) % (2 * math.pi) # Map to [0, 2pi)
        
        # Determine which bin this falls into
        bin_idx = int(rel_angle / bin_size)
        
        # Keep the closest distance in that bin
        if dist < lidar[bin_idx]:
            lidar[bin_idx] = dist
            
    return lidar

def local_to_global_velocity(ego_vx, ego_vy, local_vx, local_vy):
    """
    Transforms a velocity vector from the local frame back to the global frame.
    """
    if abs(ego_vx) < 1e-5 and abs(ego_vy) < 1e-5:
        theta = 0.0
    else:
        theta = math.atan2(ego_vy, ego_vx)
        
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    global_vx = local_vx * cos_theta - local_vy * sin_theta
    global_vy = local_vx * sin_theta + local_vy * cos_theta
    
    return global_vx, global_vy

