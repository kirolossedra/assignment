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
def JacobianPseudoinverse(J, damping_factor=0.1):
    """
    Compute the damped pseudoinverse of Jacobian matrix
    Uses damped least squares for better numerical stability
    Higher damping factor makes it more conservative but stable
    """
    # Damped least squares pseudoinverse with realistic damping
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

# Define the three main IK algorithms for comparison
IK_algorithms = {
    'Jacobian Pseudoinverse': JacobianPseudoinverse_SVD,
    'Jacobian Damped-Least-Squares': JacobianPseudoinverse,
    'Jacobian Transpose': JacobianTranspose
}

# Enhanced plotting functions for ALL algorithms comparison
def plot_all_3d_trajectories(all_results, target_pose):
    """
    Plot 3D trajectories of all algorithms in separate subplots
    """
    fig = plt.figure(figsize=(18, 6))
    
    colors = ['blue', 'orange', 'green']
    algorithm_names = list(all_results.keys())
    
    for i, (alg_name, result) in enumerate(all_results.items()):
        trajectory = np.array(result['trajectory'])
        
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                color=colors[i], linewidth=2, marker='o', markersize=2, 
                label='End Effector Path', alpha=0.8)
        
        # Plot start and target positions
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                   c='green', s=100, marker='o', label='Start')
        ax.scatter(target_pose[0], target_pose[1], target_pose[2], 
                   c='red', s=150, marker='X', label='Target')
        
        # Add workspace boundary
        max_reach = L1 + L2 + L3
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x_sphere = max_reach * np.outer(np.cos(u), np.sin(v))
        y_sphere = max_reach * np.outer(np.sin(u), np.sin(v))
        z_sphere = max_reach * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{alg_name}\n{len(trajectory)} iterations')
        ax.legend(fontsize=8)
        
        # Set equal aspect ratio
        ax.set_xlim([-max_reach, max_reach])
        ax.set_ylim([-max_reach, max_reach])
        ax.set_zlim([0, max_reach*1.5])
    
    plt.suptitle('3D End-Effector Trajectories - All Algorithms', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_all_spatial_errors(all_results, target_pose):
    """
    Plot errors for all spatial dimensions for ALL algorithms
    """
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Error Analysis - All IK Algorithms Comparison', fontsize=16)
    
    colors = ['blue', 'orange', 'green']
    error_labels = ['X Error (m)', 'Y Error (m)', 'Z Error (m)', 'Theta Error (rad)']
    algorithm_names = list(all_results.keys())
    
    for col, (alg_name, result) in enumerate(all_results.items()):
        errors = np.array(result['errors'])
        iterations = range(len(errors))
        
        for row in range(4):
            axs[row, col].plot(iterations, errors[:, row], 
                              color=colors[col], linewidth=2, alpha=0.8)
            axs[row, col].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axs[row, col].set_ylabel(error_labels[row])
            axs[row, col].grid(True, alpha=0.3)
            axs[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            if row == 0:
                axs[row, col].set_title(f'{alg_name}')
            if row == 3:
                axs[row, col].set_xlabel('Iteration')
    
    plt.tight_layout()
    plt.show()

def plot_all_spatial_trajectories(all_results, target_pose):
    """
    Plot executed trajectories for all spatial dimensions for ALL algorithms
    """
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Position/Orientation Trajectories - All IK Algorithms', fontsize=16)
    
    colors = ['blue', 'orange', 'green']
    traj_labels = ['X Position (m)', 'Y Position (m)', 'Z Position (m)', 'Theta (rad)']
    algorithm_names = list(all_results.keys())
    
    for col, (alg_name, result) in enumerate(all_results.items()):
        trajectory = np.array(result['trajectory'])
        iterations = range(len(trajectory))
        
        for row in range(4):
            axs[row, col].plot(iterations, trajectory[:, row], 
                              color=colors[col], linewidth=2, alpha=0.8, 
                              label='Executed')
            axs[row, col].axhline(y=target_pose[row], color='red', 
                                 linestyle='--', linewidth=2, alpha=0.8, 
                                 label='Target')
            axs[row, col].set_ylabel(traj_labels[row])
            axs[row, col].grid(True, alpha=0.3)
            axs[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            if row == 0:
                axs[row, col].set_title(f'{alg_name}')
                axs[row, col].legend(fontsize=8)
            if row == 3:
                axs[row, col].set_xlabel('Iteration')
    
    plt.tight_layout()
    plt.show()

def plot_all_joint_trajectories(all_results):
    """
    Plot joint angle trajectories for ALL algorithms
    """
    fig, axs = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Joint Angle Trajectories - All IK Algorithms', fontsize=16)
    
    colors = ['blue', 'orange', 'green']
    joint_names = ['Base (θ₁)', 'Shoulder (θ₂)', 'Elbow (θ₃)', 'Wrist (θ₄)']
    algorithm_names = list(all_results.keys())
    
    for col, (alg_name, result) in enumerate(all_results.items()):
        joint_trajectory = np.array(result['joint_trajectory'])
        iterations = range(len(joint_trajectory))
        
        for row in range(4):
            axs[row, col].plot(iterations, joint_trajectory[:, row], 
                              color=colors[col], linewidth=2, alpha=0.8)
            axs[row, col].set_ylabel('Angle (rad)')
            axs[row, col].grid(True, alpha=0.3)
            axs[row, col].tick_params(axis='both', which='major', labelsize=10)
            
            if row == 0:
                axs[row, col].set_title(f'{alg_name}')
            if col == 0:
                axs[row, col].text(-0.15, 0.5, joint_names[row], 
                                  transform=axs[row, col].transAxes, 
                                  rotation=90, va='center', fontweight='bold')
            if row == 3:
                axs[row, col].set_xlabel('Iteration')
    
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

# MAIN EXECUTION - Complete Analysis with All Algorithms
if __name__ == "__main__":
    print("Starting Spatial RRR Anthropomorphic Manipulator IK Analysis...")
    print(f"Robot Configuration: {spatial_rrr.name}")
    print(f"Link Lengths: L1={L1}, L2={L2}, L3={L3}")
    print(f"Base Height: d1={d1}")
    print(f"Target Pose: {target_pose_simplified}")
    print(f"Initial Guess: {initial_guess_spatial}")
    
    # Storage for results
    all_results = {}
    
    # Test each IK algorithm
    print("\n" + "="*60)
    print("RUNNING ALL IK ALGORITHMS...")
    print("="*60)
    
    for alg_name, alg_func in IK_algorithms.items():
        print(f"\nTesting {alg_name}...")
        
        pose_result, traj_result, error_result, joint_traj, conv_metrics = IK_Jacobian_Spatial(
            alg_func, spatial_rrr, target_pose_simplified, initial_guess_spatial,
            pose_func=ee_pose_simplified, gamma=0.008, max_iter=1500, tolerance=1e-5
        )
        
        all_results[alg_name] = {
            'final_pose': pose_result,
            'trajectory': traj_result,
            'errors': error_result,
            'joint_trajectory': joint_traj,
            'convergence': conv_metrics
        }
    
    # Generate comprehensive plots for ALL algorithms
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE PLOTS FOR ALL ALGORITHMS...")
    print("="*60)
    
    # 1. 3D trajectory plots for all algorithms
    print("Generating 3D trajectory comparison...")
    plot_all_3d_trajectories(all_results, target_pose_simplified)
    
    # 2. Error analysis for all algorithms
    print("Generating error analysis for all algorithms...")
    plot_all_spatial_errors(all_results, target_pose_simplified)
    
    # 3. Position/orientation trajectory plots for all algorithms
    print("Generating position/orientation trajectories for all algorithms...")
    plot_all_spatial_trajectories(all_results, target_pose_simplified)
    
    # 4. Joint trajectory plots for all algorithms
    print("Generating joint trajectory plots for all algorithms...")
    plot_all_joint_trajectories(all_results)
    
    # 5. Convergence comparison
    print("Generating convergence comparison...")
    plot_convergence_comparison(all_results)
    
    # Print comprehensive results summary
    print("\n" + "="*80)
    print("COMPREHENSIVE INVERSE KINEMATICS RESULTS SUMMARY")
    print("="*80)
    print(f"Target Pose: {target_pose_simplified}")
    print(f"Initial Guess: {initial_guess_spatial}")
    print("-"*80)
    
    for alg_name, result in all_results.items():
        final_pose = result['trajectory'][-1]
        final_error = np.linalg.norm(result['errors'][-1])
        iterations = len(result['trajectory'])
        
        print(f"\n{alg_name}:")
        print(f"  Final Pose: [{final_pose[0]:.4f}, {final_pose[1]:.4f}, {final_pose[2]:.4f}, {final_pose[3]:.4f}]")
        print(f"  Final Error Norm: {final_error:.6f}")
        print(f"  Iterations: {iterations}")
        print(f"  Joint Angles: [{result['final_pose'][0]:.4f}, {result['final_pose'][1]:.4f}, {result['final_pose'][2]:.4f}, {result['final_pose'][3]:.4f}]")
        
        # Individual error breakdown
        final_errors = result['errors'][-1]
        print(f"  Position Error: [{final_errors[0]:.4f}, {final_errors[1]:.4f}, {final_errors[2]:.4f}] m")
        print(f"  Orientation Error: {final_errors[3]:.4f} rad")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("All plots generated showing:")
    print("• 3D trajectories for all algorithms")
    print("• Individual error evolution (X, Y, Z, Theta) for each algorithm") 
    print("• Position/orientation trajectories for each algorithm")
    print("• Joint angle evolution for each algorithm")
    print("• Convergence rate comparison")
    print("="*80)
