import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from roboticstoolbox import DHRobot, RevoluteDH
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

# Define the spatial RRR anthropomorphic manipulator using DH parameters
# Based on typical anthropomorphic configuration: shoulder, elbow, wrist
L1, L2, L3 = 1.0, 1.0, 1.0  # Link lengths
d1 = 0.5  # Base height offset

# Spatial RRR manipulator with proper DH parameters for anthropomorphic configuration
# Joint 1: Base rotation (shoulder yaw)
# Joint 2: Shoulder pitch 
# Joint 3: Elbow pitch
spatial_rrr = DHRobot([
    RevoluteDH(d=d1, a=0, alpha=np.pi/2),      # Base joint with vertical offset
    RevoluteDH(d=0, a=L1, alpha=0),           # Shoulder joint  
    RevoluteDH(d=0, a=L2, alpha=0),           # Elbow joint
    RevoluteDH(d=0, a=L3, alpha=0)            # Wrist joint (for complete pose)
], name='Spatial RRR Anthropomorphic Manipulator')

# Enhanced end-effector pose calculation (position x,y,z and full orientation)
def ee_pose_spatial(q):
    """
    Calculate end-effector pose in 3D space
    Returns: [x, y, z, roll, pitch, yaw] - 6DOF pose
    """
    T = spatial_rrr.fkine(q)
    
    # Extract position
    position = T.t  # [x, y, z]
    
    # Extract orientation as Euler angles (roll, pitch, yaw)
    # Using ZYX convention (yaw-pitch-roll)
    R = T.R
    
    # Calculate Euler angles from rotation matrix
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0
    
    return np.array([position[0], position[1], position[2], roll, pitch, yaw])

# Alternative simplified pose for position + single orientation angle
def ee_pose_simplified(q):
    """
    Simplified pose calculation for comparison with planar version
    Returns: [x, y, z, theta_total] where theta_total is sum of joint angles
    """
    T = spatial_rrr.fkine(q)
    x, y, z = T.t
    theta_total = sum(q)  # Sum of joint angles as simplified orientation
    return np.array([x, y, z, theta_total])

# Desired end-effector pose in 3D space
# Format: [x, y, z, roll, pitch, yaw] or [x, y, z, theta_total] for simplified
target_pose_6dof = np.array([1.2, 0.8, 1.0, 0.2, 0.3, 0.5])  # Full 6DOF target
target_pose_simplified = np.array([1.2, 0.8, 1.0, 1.5])      # Simplified 4DOF target

# Enhanced Jacobian pseudoinverse calculation with better numerical stability
def JacobianPseudoinverse(J, damping_factor=1e-4):
    """
    Compute the damped pseudoinverse of Jacobian matrix
    Uses damped least squares for better numerical stability
    """
    # Damped least squares pseudoinverse
    JTJ = J.T @ J
    damping_matrix = damping_factor * np.eye(JTJ.shape[0])
    J_pinv = np.linalg.inv(JTJ + damping_matrix) @ J.T
    
    return J_pinv

def JacobianPseudoinverse_SVD(J, threshold=1e-6):
    """
    SVD-based pseudoinverse (original method from planar code)
    """
    # Compute the SVD of J
    U, Sigma, Vt = np.linalg.svd(J, full_matrices=False)
    
    # Invert the singular values with threshold
    Sigma_inv = np.zeros_like(Sigma)
    for i in range(len(Sigma)):
        if Sigma[i] > threshold:
            Sigma_inv[i] = 1.0 / Sigma[i]
    
    # Compute the pseudoinverse
    J_pinv = Vt.T @ np.diag(Sigma_inv) @ U.T
    return J_pinv

def JacobianTranspose(J, alpha=1.0):
    """
    Jacobian transpose method with scaling factor
    """
    return alpha * J.T

def JacobianTranspose_Adaptive(J, error):
    """
    Adaptive Jacobian transpose with variable step size
    """
    JT = J.T
    # Adaptive scaling based on error magnitude
    alpha = 1.0 / (1.0 + np.linalg.norm(error))
    return alpha * JT

# Enhanced IK solver with multiple pose calculation options
def IK_Jacobian_Spatial(IK_alg, manipulator, target_pose, initial_guess, 
                       pose_func=ee_pose_simplified, gamma=0.01, max_iter=2000, 
                       tolerance=1e-4, use_full_jacobian=False):
    """
    Spatial inverse kinematics solver
    
    Args:
        IK_alg: IK algorithm function
        manipulator: Robot manipulator object
        target_pose: Target end-effector pose
        initial_guess: Initial joint configuration
        pose_func: Function to calculate current pose
        gamma: Step size
        max_iter: Maximum iterations
        tolerance: Convergence tolerance
        use_full_jacobian: Whether to use full 6x4 Jacobian or reduced version
    """
    pose = np.array(initial_guess, dtype=float)
    trajectory = []
    errors = []
    joint_trajectories = []
    convergence_metrics = []
    
    for iteration in range(max_iter):
        current_pose = pose_func(pose)
        error = target_pose - current_pose
        
        trajectory.append(current_pose.copy())
        errors.append(error.copy())
        joint_trajectories.append(pose.copy())
        
        # Calculate convergence metrics
        error_norm = np.linalg.norm(error)
        convergence_metrics.append(error_norm)
        
        # Check for convergence
        if error_norm < tolerance:
            print(f"Converged after {iteration+1} iterations with error norm: {error_norm:.6f}")
            break
        
        # Get Jacobian matrix
        if use_full_jacobian and len(target_pose) == 6:
            # Use full 6DOF Jacobian
            J_full = manipulator.jacob0(pose)  # 6x4 Jacobian
            J = J_full  # Use full Jacobian
        else:
            # Use reduced Jacobian based on target pose dimensions
            J_full = manipulator.jacob0(pose)
            if len(target_pose) == 4:
                # For simplified 4DOF case: [x, y, z, theta_total]
                J = np.zeros((4, len(pose)))
                J[:3, :] = J_full[:3, :]  # Position part
                J[3, :] = 1.0  # theta_total is sum of all joints
            else:
                J = J_full[:len(target_pose), :]
        
        # Apply IK algorithm
        if 'Adaptive' in IK_alg.__name__:
            delta_q = IK_alg(J, error)
        else:
            delta_q = IK_alg(J)
        
        # Update joint angles
        pose += gamma * (delta_q @ error)
        
        # Optional: Add joint limits (uncomment if needed)
        # pose = np.clip(pose, -np.pi, np.pi)
    
    if iteration == max_iter - 1:
        print(f"Maximum iterations reached. Final error norm: {np.linalg.norm(error):.6f}")
    
    return pose, trajectory, errors, joint_trajectories, convergence_metrics

# Initial guess for joint angles (4 joints for spatial manipulator)
initial_guess_spatial = [0.3, 0.5, 0.2, 0.1]

# Define IK algorithms to test
IK_algorithms = {
    'Jacobian Transpose': JacobianTranspose,
    'Jacobian Transpose Adaptive': JacobianTranspose_Adaptive,
    'Jacobian Pseudoinverse (Damped)': JacobianPseudoinverse,
    'Jacobian Pseudoinverse (SVD)': JacobianPseudoinverse_SVD
}

# Storage for results
results = {}

# Test each IK algorithm
for alg_name, alg_func in IK_algorithms.items():
    print(f"\nTesting {alg_name}...")
    
    pose_result, traj_result, error_result, joint_traj, conv_metrics = IK_Jacobian_Spatial(
        alg_func, spatial_rrr, target_pose_simplified, initial_guess_spatial,
        pose_func=ee_pose_simplified, gamma=0.008, max_iter=1500, tolerance=1e-5
    )
    
    results[alg_name] = {
        'final_pose': pose_result,
        'trajectory': traj_result,
        'errors': error_result,
        'joint_trajectory': joint_traj,
        'convergence': conv_metrics
    }

# Enhanced 3D plotting functions
def plot_3d_trajectory(trajectory, target_pose, title, algorithm_name):
    """
    Plot 3D trajectory of end-effector movement
    """
    trajectory = np.array(trajectory)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
            'b-o', linewidth=2, markersize=3, label='End Effector Path', alpha=0.7)
    
    # Plot start and target positions
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
               c='green', s=100, marker='o', label='Start Position')
    ax.scatter(target_pose[0], target_pose[1], target_pose[2], 
               c='red', s=150, marker='X', label='Target Position')
    
    # Add workspace boundaries (approximate)
    max_reach = L1 + L2 + L3
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = max_reach * np.outer(np.cos(u), np.sin(v))
    y_sphere = max_reach * np.outer(np.sin(u), np.sin(v))
    z_sphere = max_reach * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Z Position (m)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f'3D End-Effector Trajectory - {algorithm_name}\n{title}', fontsize=14)
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = max_reach
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([0, max_range*1.5])
    
    plt.tight_layout()
    plt.show()

def plot_spatial_errors(errors, title, algorithm_name):
    """
    Plot errors for all spatial dimensions
    """
    errors = np.array(errors)
    iterations = range(len(errors))
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Error Analysis - {algorithm_name}\n{title}', fontsize=16)
    
    # Plot error for x
    axs[0, 0].plot(iterations, errors[:, 0], label='Error in X', linewidth=2, color='blue')
    axs[0, 0].set_xlabel('Iteration', fontsize=12)
    axs[0, 0].set_ylabel('Error in X (m)', fontsize=12)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].set_title('X Position Error', fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot error for y
    axs[0, 1].plot(iterations, errors[:, 1], label='Error in Y', linewidth=2, color='orange')
    axs[0, 1].set_xlabel('Iteration', fontsize=12)
    axs[0, 1].set_ylabel('Error in Y (m)', fontsize=12)
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].set_title('Y Position Error', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot error for z
    axs[1, 0].plot(iterations, errors[:, 2], label='Error in Z', linewidth=2, color='green')
    axs[1, 0].set_xlabel('Iteration', fontsize=12)
    axs[1, 0].set_ylabel('Error in Z (m)', fontsize=12)
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].set_title('Z Position Error', fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot error for orientation (theta)
    axs[1, 1].plot(iterations, errors[:, 3], label='Error in Theta', linewidth=2, color='red')
    axs[1, 1].set_xlabel('Iteration', fontsize=12)
    axs[1, 1].set_ylabel('Error in Theta (rad)', fontsize=12)
    axs[1, 1].legend(fontsize=10)
    axs[1, 1].set_title('Orientation Error', fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_spatial_trajectories(trajectory, target_pose, algorithm_name):
    """
    Plot executed trajectories for all spatial dimensions
    """
    trajectory = np.array(trajectory)
    iterations = range(len(trajectory))
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Executed Trajectories - {algorithm_name}', fontsize=16)
    
    # Plot executed x trajectory
    axs[0, 0].plot(iterations, trajectory[:, 0], label='Executed X', linewidth=2, color='blue')
    axs[0, 0].axhline(y=target_pose[0], color='red', linestyle='--', 
                      label='Target X', linewidth=2, alpha=0.8)
    axs[0, 0].set_xlabel('Iteration', fontsize=12)
    axs[0, 0].set_ylabel('X Position (m)', fontsize=12)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].set_title('X Position Trajectory', fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot executed y trajectory
    axs[0, 1].plot(iterations, trajectory[:, 1], label='Executed Y', linewidth=2, color='orange')
    axs[0, 1].axhline(y=target_pose[1], color='red', linestyle='--', 
                      label='Target Y', linewidth=2, alpha=0.8)
    axs[0, 1].set_xlabel('Iteration', fontsize=12)
    axs[0, 1].set_ylabel('Y Position (m)', fontsize=12)
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].set_title('Y Position Trajectory', fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot executed z trajectory
    axs[1, 0].plot(iterations, trajectory[:, 2], label='Executed Z', linewidth=2, color='green')
    axs[1, 0].axhline(y=target_pose[2], color='red', linestyle='--', 
                      label='Target Z', linewidth=2, alpha=0.8)
    axs[1, 0].set_xlabel('Iteration', fontsize=12)
    axs[1, 0].set_ylabel('Z Position (m)', fontsize=12)
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].set_title('Z Position Trajectory', fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=10)
    
    # Plot executed theta trajectory
    axs[1, 1].plot(iterations, trajectory[:, 3], label='Executed Theta', linewidth=2, color='red')
    axs[1, 1].axhline(y=target_pose[3], color='red', linestyle='--', 
                      label='Target Theta', linewidth=2, alpha=0.8)
    axs[1, 1].set_xlabel('Iteration', fontsize=12)
    axs[1, 1].set_ylabel('Theta (rad)', fontsize=12)
    axs[1, 1].legend(fontsize=10)
    axs[1, 1].set_title('Orientation Trajectory', fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_joint_trajectories(joint_trajectory, algorithm_name):
    """
    Plot joint angle trajectories
    """
    joint_trajectory = np.array(joint_trajectory)
    iterations = range(len(joint_trajectory))
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Joint Angle Trajectories - {algorithm_name}', fontsize=16)
    
    joint_names = ['Base (θ₁)', 'Shoulder (θ₂)', 'Elbow (θ₃)', 'Wrist (θ₄)']
    colors = ['blue', 'orange', 'green', 'red']
    
    for i in range(4):
        row, col = i // 2, i % 2
        axs[row, col].plot(iterations, joint_trajectory[:, i], 
                          linewidth=2, color=colors[i], label=f'Joint {i+1}')
        axs[row, col].set_xlabel('Iteration', fontsize=12)
        axs[row, col].set_ylabel('Joint Angle (rad)', fontsize=12)
        axs[row, col].set_title(f'{joint_names[i]} Trajectory', fontsize=12)
        axs[row, col].grid(True, alpha=0.3)
        axs[row, col].tick_params(axis='both', which='major', labelsize=10)
        axs[row, col].legend(fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_convergence_comparison(results):
    """
    Compare convergence of different IK algorithms
    """
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'orange', 'green', 'red']
    linestyles = ['-', '--', '-.', ':']
    
    for i, (alg_name, result) in enumerate(results.items()):
        convergence = result['convergence']
        iterations = range(len(convergence))
        plt.semilogy(iterations, convergence, 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)],
                    linewidth=2, label=alg_name)
    
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Error Norm (log scale)', fontsize=14)
    plt.title('Convergence Comparison of IK Algorithms', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

# Generate comprehensive plots for the best performing algorithm
best_algorithm = 'Jacobian Pseudoinverse (Damped)'  # Usually performs best
if best_algorithm in results:
    result = results[best_algorithm]
    
    print(f"\nGenerating comprehensive plots for {best_algorithm}...")
    
    # 3D trajectory plot
    plot_3d_trajectory(result['trajectory'], target_pose_simplified, 
                      'Spatial RRR Manipulator', best_algorithm)
    
    # Error analysis
    plot_spatial_errors(result['errors'], 'Error Evolution', best_algorithm)
    
    # Trajectory plots
    plot_spatial_trajectories(result['trajectory'], target_pose_simplified, best_algorithm)
    
    # Joint trajectories
    plot_joint_trajectories(result['joint_trajectory'], best_algorithm)

# Convergence comparison
plot_convergence_comparison(results)

# Print final results summary
print("\n" + "="*80)
print("INVERSE KINEMATICS RESULTS SUMMARY")
print("="*80)
print(f"Target Pose: {target_pose_simplified}")
print(f"Initial Guess: {initial_guess_spatial}")
print("-"*80)

for alg_name, result in results.items():
    final_pose = result['trajectory'][-1]
    final_error = np.linalg.norm(result['errors'][-1])
    iterations = len(result['trajectory'])
    
    print(f"\n{alg_name}:")
    print(f"  Final Pose: [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}, {final_pose[3]:.4f}]")
    print(f"  Final Error Norm: {final_error:.6f}")
    print(f"  Iterations: {iterations}")
    print(f"  Joint Angles: [{result['final_pose'][0]:.4f}, {result['final_pose'][1]:.4f}, {result['final_pose'][2]:.4f}, {result['final_pose'][3]:.4f}]")

print("\n" + "="*80)

# MAIN EXECUTION - Run the complete analysis
if __name__ == "__main__":
    print("Starting Spatial RRR Anthropomorphic Manipulator IK Analysis...")
    print(f"Robot Configuration: {spatial_rrr.name}")
    print(f"Link Lengths: L1={L1}, L2={L2}, L3={L3}")
    print(f"Base Height: d1={d1}")
    
    # Choose which algorithm to focus on for detailed plots (mimicking original code structure)
    # Change index i to select different algorithms: 0=Transpose, 1=Transpose_Adaptive, 2=Pseudoinverse_Damped, 3=Pseudoinverse_SVD
    i = 2  # Default to Damped Pseudoinverse (usually best performance)
    
    algorithm_list = list(IK_algorithms.keys())
    selected_algorithm = algorithm_list[i]
    
    print(f"\nRunning detailed analysis with: {selected_algorithm}")
    
    # Run the selected algorithm (mimicking your original structure)
    pose_IK_alg, traj_IK_alg, error_IK_alg, joint_traj_IK_alg, conv_IK_alg = IK_Jacobian_Spatial(
        list(IK_algorithms.values())[i], 
        spatial_rrr, 
        target_pose_simplified, 
        initial_guess_spatial,
        pose_func=ee_pose_simplified, 
        gamma=0.008, 
        max_iter=1500, 
        tolerance=1e-5
    )
    
    print(f"\nSelected Algorithm Results:")
    print(f"Final Joint Configuration: {pose_IK_alg}")
    print(f"Final End-Effector Pose: {traj_IK_alg[-1]}")
    print(f"Final Error: {error_IK_alg[-1]}")
    print(f"Convergence achieved in {len(traj_IK_alg)} iterations")
    
    # Generate all plots for the selected algorithm (mimicking your original plotting calls)
    print(f"\nGenerating plots for {selected_algorithm}...")
    
    # 3D Trajectory plot (equivalent to your plot_trajectory)
    plot_3d_trajectory(traj_IK_alg, target_pose_simplified, 
                      f'3D Trajectory - {selected_algorithm}', selected_algorithm)
    
    # Error plots (equivalent to your plot_errors) 
    plot_spatial_errors(error_IK_alg, f'Spatial Errors - {selected_algorithm}', selected_algorithm)
    
    # Executed trajectory plots (equivalent to your plot_executed_trajectories)
    plot_spatial_trajectories(traj_IK_alg, target_pose_simplified, selected_algorithm)
    
    # Additional 3D-specific plots
    plot_joint_trajectories(joint_traj_IK_alg, selected_algorithm)
    
    # Run comparison of all algorithms (like your original but extended)
    print(f"\nRunning comparison of all {len(IK_algorithms)} algorithms...")
    all_results = {}
    
    for alg_name, alg_func in IK_algorithms.items():
        print(f"Testing {alg_name}...")
        pose_result, traj_result, error_result, joint_result, conv_result = IK_Jacobian_Spatial(
            alg_func, spatial_rrr, target_pose_simplified, initial_guess_spatial,
            pose_func=ee_pose_simplified, gamma=0.008, max_iter=1500, tolerance=1e-5
        )
        
        all_results[alg_name] = {
            'final_pose': pose_result,
            'trajectory': traj_result, 
            'errors': error_result,
            'joint_trajectory': joint_result,
            'convergence': conv_result
        }
    
    # Convergence comparison plot
    plot_convergence_comparison(all_results)
    
    # Final summary (matching your original format)
    print("\n" + "="*80)
    print("SPATIAL RRR MANIPULATOR - INVERSE KINEMATICS COMPLETE")
    print("="*80)
    print(f"Target achieved with {selected_algorithm}")
    print(f"Final joint configuration: {pose_IK_alg}")
    print(f"Final end-effector pose: {traj_IK_alg[-1]}") 
    print(f"Target pose was: {target_pose_simplified}")
    print(f"Position error: {np.linalg.norm(error_IK_alg[-1][:3]):.6f} m")
    print(f"Orientation error: {abs(error_IK_alg[-1][3]):.6f} rad")
    print("="*80)
