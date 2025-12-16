### Robot bolt path planning simulation

import numpy as np
import matplotlib.pyplot as plt
import cv2

### Define robot model (SCARA)
# import roboticstoolbox as rtb
# robot = rtb.models.DH.Cobra600() # Decided against use
robot_base = np.array([0.0, 0.0])
L1 = 0.5
L2 = 0.5
robot_reach = L1 + L2

### Define environment, anchored at (0,0)
max_bolts = 5
min_bolts = 0
obj_size = (1.0, 1.0)
img_pixels = 500
bolt_pixels = 10

### Generate bolts
def generate_bolts():
    img = np.zeros((img_pixels, img_pixels), dtype=np.uint8)
    num_bolts = np.random.randint(min_bolts, max_bolts+1)
    bolt_coords = np.random.rand(num_bolts, 2)
    for x,y in bolt_coords:
        px = int(x / obj_size[0] * (img_pixels - 1))
        py = int((1 - y / obj_size[1]) * (img_pixels - 1))
        cv2.circle(img, (px, py), bolt_pixels, (255), -1)
    return img

### CV bolt detection
def detect_bolts(img):
    detected = []
    # Preprocess image, detect circles
    blur = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=bolt_pixels+5,
                               param1=300, param2=15,
                               minRadius=3, maxRadius=2*bolt_pixels)
    if circles is not None:
        for c in circles[0]:
            x, y = c[0], c[1]
            real_x = x / img_pixels * obj_size[0]
            real_y = 1 - (y / img_pixels * obj_size[1])
            detected.append([real_x, real_y])
    if len(detected) == 0: # If no bolts detected
        return np.empty((0, 2), dtype=float)
    return np.array(detected)

### Path planning using TSP
def plan_path(bolts, start=robot_base):
    pts = bolts.copy()
    ordered = []
    current = start
    while len(pts) > 0:
        dists = np.linalg.norm(pts - current, axis=1)
        idx = np.argmin(dists)
        ordered.append(pts[idx])
        current = pts[idx]
        pts = np.delete(pts, idx, axis=0)
    if len(ordered) == 0: # If no bolts detected
        return np.empty((0, 2), dtype=float)
    return np.array(ordered)

### SCARA Inverse Kinematics for joint angles
def scara_ik(x, y):
    theta2 = np.arccos((x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2))
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2

### SCARA Forward Kinematics for joint positions
def scara_fk(theta1, theta2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return np.array([x1, y1]), np.array([x2, y2])

### Check SCARA robot reachability
def in_reach(pt):
    return np.linalg.norm(pt - robot_base) <= robot_reach

### SCARA robot motion
def scara_motion(path,ax,fig):
    import matplotlib.animation as ani

    if len(path) == 0: # No path, no motion
        ax.set_xlim(obj_size[0]-1.1, obj_size[1]+0.1)
        ax.set_ylim(obj_size[0]-1.5, obj_size[1]+0.1)
        ax.set_aspect('equal')
        ax.set_title('No Robot Motion')
        return None

    thetas = []
    for x, y in path:
        theta1, theta2 = scara_ik(x, y)
        thetas.append((theta1, theta2)) 
    thetas = np.array(thetas)

    ax.set_xlim(obj_size[0]-1.1, obj_size[1]+0.1)
    ax.set_ylim(obj_size[0]-1.5, obj_size[1]+0.1)
    ax.set_aspect('equal')
    ax.set_title('SCARA Robot Motion')

    # Object
    obj = plt.Rectangle((0, 0), obj_size[0], obj_size[1], linewidth=1, edgecolor='black', facecolor='grey')
    ax.add_patch(obj)

    # Robot links
    link1, = ax.plot([], [], 'b-', lw=4)
    link2, = ax.plot([], [], 'r-', lw=4)
    end_eff, = ax.plot([], [], 'go', markersize=8)

    # Animation update function
    def update(frame):
        theta1, theta2 = thetas[frame]
        joint1, joint2 = scara_fk(theta1, theta2)

        link1.set_data([robot_base[0], joint1[0]], [robot_base[1], joint1[1]])
        link2.set_data([joint1[0], joint2[0]], [joint1[1], joint2[1]])
        end_eff.set_data([joint2[0]], [joint2[1]])
        return link1, link2, end_eff
    anim = ani.FuncAnimation(fig, update, frames=len(thetas), blit=True, 
                             interval=500, repeat=True)

    return anim

### Execute on multiple objects
def execute():
    N_obj = 3 # Number of objects

    for i in range(N_obj):
        # Generate bolt image
        bolts = generate_bolts()

        # Detect bolts
        detected_bolts = detect_bolts(bolts)

        # Determine reachable bolts
        reachable_bolts = np.array([b for b in detected_bolts if in_reach(b)])
        unreachable_bolts = np.array([b for b in detected_bolts if not in_reach(b)])

        # Plan path for reachable bolts
        bolt_order = plan_path(reachable_bolts)
        if len(bolt_order) > 0:
            robot_path = np.vstack(([robot_base], bolt_order, [robot_base]))
        else:
            robot_path = np.empty((0, 2), dtype=float)

        print(f"Object {i+1}: {len(detected_bolts)} bolts detected, "
              f"{len(reachable_bolts)} reachable")
        
        # Initialize plots
        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle(f"Object {i+1} Bolt Detection and Robot Path")
        
        # Left plot: Bolt detection and path
        ax1.imshow(bolts, cmap="gray", extent=[0, 1, 0, 1]) # Results of CV
        if detected_bolts.size > 0:
            ax1.scatter(detected_bolts[:, 0], detected_bolts[:, 1], 
                        c='orange', marker='*', s=100, label="Detected Bolts")
        if reachable_bolts.size > 0:
            ax1.scatter(reachable_bolts[:, 0], reachable_bolts[:, 1], 
                        c='green', marker='o', s=25, label="Reachable Bolts")
        if unreachable_bolts.size > 0:
            ax1.scatter(unreachable_bolts[:, 0], unreachable_bolts[:, 1], 
                        c='red', marker='x',s=100, label="Unreachable Bolts")
        if robot_path.size > 0:
            ax1.plot(robot_path[:, 0], robot_path[:, 1], 'r--', label="Planned Path")
        ax1.scatter(robot_base[0], robot_base[1], c="magenta", s=100, label="Robot Base")
        ax1.set_title(f"Bolt Detection and Path Planning")
        ax1.legend(loc='upper left',bbox_to_anchor=(1,1))
        ax1.set_xlim([obj_size[0]-1.1, obj_size[1]+0.1])
        ax1.set_ylim([obj_size[0]-1.1, obj_size[1]+0.1])
        ax1.set_aspect('equal')

        # Right plot: SCARA robot motion
        anim = scara_motion(robot_path, ax2, fig)
        
        plt.tight_layout()
        plt.show()

### MAIN EXECUTION
execute()
