import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ==========================================
# 1. CONFIGURATION
# ==========================================
FPS_DRONE = 30
FPS_SERVER = 5
LATENCY_SEC = 0.5  # 500ms processing delay
HISTORY_LEN = 300  # INCREASED: Drone remembers last 10 seconds (Safety Net)

# Communication Channels
q_drone_to_server = multiprocessing.Queue()
q_server_to_drone = multiprocessing.Queue()

# ==========================================
# 2. THE "SERVER" PROCESS (Fixed Logic)
# ==========================================
def server_process(input_q, output_q):
    """
    Simulates a ground station that might get overwhelmed.
    """
    print("🖥️  Server Process Started")
    
    while True:
        # --- THE FIX: QUEUE DRAINING ---
        # Don't process items one by one. Grab the LATEST and dump the rest.
        if input_q.empty():
            time.sleep(0.01)
            continue
            
        latest_packet = None
        while not input_q.empty():
            latest_packet = input_q.get()
        
        # Now we only work on the freshest frame
        frame_id, true_pos = latest_packet
        
        # Simulate the heavy processing time (Lag)
        time.sleep(LATENCY_SEC)
        
        # Send Correction
        # In a real app, this is where you'd run YOLOv8
        output_q.put((frame_id, true_pos))

# ==========================================
# 3. THE "DRONE" SIMULATION
# ==========================================
class DroneSimulation:
    def __init__(self):
        self.frame_id = 0
        self.history = deque(maxlen=HISTORY_LEN)
        
        # Physics State
        self.angle = 0
        self.radius = 10
        self.est_pos = np.array([10.0, 0.0]) 
        
        # Drift: Drone constantly hallucinates moving "Up and Right"
        self.drift_velocity = np.array([0.03, 0.03]) 
        
    def update_physics(self):
        self.frame_id += 1
        
        # 1. True Target Moves (Circle)
        self.angle += 0.05
        true_x = self.radius * np.cos(self.angle)
        true_y = self.radius * np.sin(self.angle)
        
        # 2. Drone Drifts
        # It moves correctly relative to circle, but adds error every frame
        circle_motion = np.array([-self.radius * np.sin(self.angle)*0.05, 
                                   self.radius * np.cos(self.angle)*0.05])
        
        self.est_pos += circle_motion + self.drift_velocity
        
        # 3. SAVE HISTORY
        self.history.append((self.frame_id, self.est_pos.copy()))
        
        return np.array([true_x, true_y])

    def check_for_correction(self):
        """Looks for messages from the slow server"""
        try:
            # Check queue without blocking
            # We drain this too, just in case
            packet = None
            while not q_server_to_drone.empty():
                packet = q_server_to_drone.get_nowait()
            
            if packet is None: return False
            
            s_frame_id, s_pos = packet
            
            # === THE RETRODICTION LOGIC ===
            # Find what we thought at that time
            past_belief = None
            for h_fid, h_pos in self.history:
                if h_fid == s_frame_id:
                    past_belief = h_pos
                    break
            
            if past_belief is not None:
                # Calculate the "Lie" (Drift)
                # "Server says I was at [10,10]. I thought I was at [12,12]. I was wrong by [-2,-2]"
                error = s_pos - past_belief
                
                # Apply that realization to NOW
                self.est_pos += error
                print(f"Frame {self.frame_id}: Snapped back! (Error: {error[0]:.2f}, {error[1]:.2f})")
                return True
            else:
                print(f"⚠️ History miss! Server sent Frame {s_frame_id}, but oldest memory is {self.history[0][0]}")
                
        except Exception as e:
            pass
        return False

# ==========================================
# 4. MAIN LOOP
# ==========================================
if __name__ == "__main__":
    # Mac/Linux Safety check
    multiprocessing.set_start_method('spawn', force=True)

    p = multiprocessing.Process(target=server_process, args=(q_drone_to_server, q_server_to_drone))
    p.daemon = True
    p.start()
    
    drone = DroneSimulation()
    
    # Visualization
    fig, ax = plt.subplots()
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect('equal')
    
    drone_dot, = ax.plot([], [], 'bo', markersize=8, label='Drone Estimate (Drifting)')
    true_dot, = ax.plot([], [], 'gx', markersize=8, label='True Target')
    ax.legend()
    ax.set_title("OOSM Correction Simulation")

    def animate(i):
        # 1. Physics Step
        true_pos = drone.update_physics()
        
        # 2. Send to Server (Rate Limited)
        if drone.frame_id % int(FPS_DRONE / FPS_SERVER) == 0:
            q_drone_to_server.put((drone.frame_id, true_pos))
            
        # 3. Apply Corrections
        corrected = drone.check_for_correction()
        
        # 4. Draw
        drone_dot.set_data([drone.est_pos[0]], [drone.est_pos[1]])
        true_dot.set_data([true_pos[0]], [true_pos[1]])
        
        if corrected:
            ax.set_title(f"Frame {drone.frame_id}: ✅ CORRECTION APPLIED")
            # Flash the screen slightly (optional visual cue)
            drone_dot.set_color('r')
        else:
            ax.set_title(f"Frame {drone.frame_id}: Drifting...")
            drone_dot.set_color('b')

        return drone_dot, true_dot

    ani = FuncAnimation(fig, animate, interval=1000/FPS_DRONE, blit=False)
    plt.show()