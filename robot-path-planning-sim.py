### Robot bolt path planning simulation

import numpy as np
import matplotlib.pyplot as plt
import cv2

### Define robot model (SCARA)
#import roboticstoolbox as rtb
#robot = rtb.models.DH.Cobra600() # Decided against use
robot_base = np.array([0.0, 0.0])
L1 = 0.5
L2 = 0.5
robot_reach = L1 + L2

### Generate bolts
def generate_bolts(max_bolts=5, obj_size=(1.0,1.0), img_pixels=500, bolt_pixels=10):
    img = np.zeros((img_pixels, img_pixels), dtype=np.uint8)
    num_bolts = np.random.randint(3, max_bolts+1)
    bolt_coords = np.random.rand(num_bolts, 2)
    for x,y in bolt_coords:
        px = int(x / obj_size[0] * (img_pixels - 1))
        py = int((1 - y / obj_size[1]) * (img_pixels - 1))
        cv2.circle(img, (px, py), bolt_pixels, (255), -1)
    return img

### CV bolt detection
def detect_bolts(img, obj_size=(1.0,1.0), img_pixels=500):
    detected = []
    # Preprocess image, detect circles
    blur = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=15,
                               param1=50, param2=15,
                               minRadius=3, maxRadius=30)
    if circles is not None:
        for c in circles[0]:
            x, y = c[0], c[1]
            real_x = x / img_pixels * obj_size[0]
            real_y = 1 - (y / img_pixels * obj_size[1])
            detected.append([real_x, real_y])
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
    return np.array(ordered)

### SCARA Inverse Kinematics for joint angles
def scara_ik(x, y):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    #theta2 = np.arctan2(np.sqrt(1 - D**2), D)
    theta2 = np.arccos(D)
    theta1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    return theta1, theta2

### SCARA Forward Kinematics for joint positions
def scara_fk(theta1, theta2):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return np.array([x1, y1]), np.array([x2, y2])


### Check robot reachability
def in_reach(pt):
    return np.linalg.norm(pt - robot_base) <= robot_reach
    
### SCARA robot motion
def scara_motion(path):
    import matplotlib.animation as ani

    thetas = []
    for x, y in path:
        theta1, theta2 = scara_ik(x, y)
        thetas.append((theta1, theta2)) 
    thetas = np.array(thetas)

    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.5, 1.1)
    ax.set_aspect('equal')
    ax.set_title('SCARA Robot Motion')

    obj = plt.Rectangle((0, 0), 1.0, 1.0, linewidth=1, edgecolor='black', facecolor='grey')
    ax.add_patch(obj)

    link1, = ax.plot([], [], 'b-', lw=4)
    link2, = ax.plot([], [], 'r-', lw=4)
    end_eff, = ax.plot([], [], 'go', markersize=8)

    def update(frame):
        theta1, theta2 = thetas[frame]
        joint1, joint2 = scara_fk(theta1, theta2)

        link1.set_data([robot_base[0], joint1[0]], [robot_base[1], joint1[1]])
        link2.set_data([joint1[0], joint2[0]], [joint1[1], joint2[1]])
        end_eff.set_data([joint2[0]], [joint2[1]])
        return link1, link2, end_eff
    anim = ani.FuncAnimation(fig, update, frames=len(thetas), blit=True, interval=500)
    plt.show()

    return anim

### Execute on multiple objects
def run_simulation():
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
        robot_path = np.vstack(([robot_base], bolt_order, [robot_base]))
        print(f"Object {i+1}: Bolts at \n{detected_bolts}")
        
        # Visualization of bolt locations
        plt.figure()
        plt.imshow(bolts, cmap="gray", extent=[0, 1, 0, 1]) # Results of CV
        if detected_bolts.size > 0:
            plt.scatter(detected_bolts[:, 0], detected_bolts[:, 1], c='orange', marker='*',label="Detected Bolts")
        if reachable_bolts.size > 0:
            plt.scatter(reachable_bolts[:, 0], reachable_bolts[:, 1], c='green', marker='o',label="Reachable Bolts")
        if unreachable_bolts.size > 0:
            plt.scatter(unreachable_bolts[:, 0], unreachable_bolts[:, 1], c='red', marker='x',label="Unreachable Bolts")
        if robot_path.size > 0:
            plt.plot(robot_path[:, 0], robot_path[:, 1], 'r--', label="Planned Path")
        plt.scatter(robot_base[0], robot_base[1], c="magenta", s=100, label="Robot Base")
        plt.title(f"Object {i+1} Bolt Detection and Robot Path")
        plt.legend()
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.gca().set_aspect('equal')
        plt.show()

        # Simulate SCARA robot motion
        scara_motion(robot_path)

### MAIN EXECUTION
run_simulation()

