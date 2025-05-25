"""
Demonstrates an object identification and grasping pipeline using a simulated robot.

The script continuously performs the following cycle:
1.  Captures a view from the simulation and generates a point cloud.
2.  Segments objects in the view, creating a visual prompt.
3.  Uses a multimodal model (GPT-4o) with a text query to identify target objects.
4.  If no targets are found, the script exits.
5.  Selects a target and calculates its center.
6.  Crops the point cloud around the target.
7.  Predicts grasp poses for the cropped point cloud.
8.  Transforms grasps to the robot's frame and filters for reachability.
9.  If no valid grasps, restarts the loop.
10. Executes a random valid grasp, drops the object, and advances the simulation.
"""


import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from typing import List
from client import GeneralBionixClient, PointCloudData, Grasp
from sim import (
    SimGrasp, 
    ObjectInfo, 
    CUBE_ORIENTATION, 
    CUBE_SCALING, 
    SPHERE_ORIENTATION, 
    SPHERE_SCALING, 
    TRAY_ORIENTATION, 
    TRAY_SCALING, 
    TRAY_POS, 
    CUBE_RGBA, 
    SPHERE_RGBA, 
    SPHERE_MASS
)
from utils import clean_and_downsample_pcd, smooth_depth_map, visualize_pcd, upsample_pcd, compute_mask_center_of_mass, visualize_point_cloud_comparison_meshcat, visualize_depth_map_meshcat, visualize_complete_pipeline_meshcat, compute_optimal_grasp_point, enhance_point_cloud_for_grasping, compute_optimal_grasp_point_3d, verify_grasp_pipeline_consistency, validate_grasps_on_object, map_2d_to_3d_accurate, project_3d_to_image_plane
from vis_grasps import launch_visualizer
from vis_grasps import vis_grasps_meshcat
from transform import transform_pcd_cam_to_rob, transform_cam_to_rob

from visual_prompt.visual_prompt import VisualPrompterGrounding
from visual_prompt.utils import display_image


# GPT-4o prompt
USER_QUERY = "Identify the red cubes. If no red cubes are available then return an empty list."

# User TODO
API_KEY = "" # Use your API key here - empty string works for local fallbacks
OS = "MAC" # "MAC" or "LINUX"


# Define simulation objects
SIMULATION_OBJECTS = [
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.3, 0.0, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="cube_small.urdf",
        position=[0.23, 0.05, 0.025],
        orientation=CUBE_ORIENTATION,
        scaling=CUBE_SCALING,
        color=CUBE_RGBA
    ),
    ObjectInfo(
        urdf_path="tray/traybox.urdf",
        position=TRAY_POS,
        orientation=TRAY_ORIENTATION,
        scaling=TRAY_SCALING
    ),
    ObjectInfo(
        urdf_path="sphere2.urdf",
        position=[0.35, 0.1, 0.025],
        orientation=SPHERE_ORIENTATION,
        scaling=SPHERE_SCALING,
        color=SPHERE_RGBA,
        mass=SPHERE_MASS
    )
]


FREQUENCY = 30
DOWN_SAMPLE = 4 # Don't change this
URDF_PATH = "piper_description/urdf/piper_description_virtual_eef_free_gripper.urdf"
CONFIG_PATH = 'visual_prompt/config/visual_prompt_config.yaml'





def main():
    """Main execution function for the grasp prediction pipeline."""
    # Initialize the simulation environment
    env = SimGrasp(urdf_path=URDF_PATH, frequency=FREQUENCY, objects=SIMULATION_OBJECTS)
    client = GeneralBionixClient(api_key=API_KEY)
    vis = launch_visualizer()
    grounder = VisualPrompterGrounding(CONFIG_PATH, debug=True)

    # Run one complete demo (change to 'while True:' for continuous operation)
    for demo_iteration in range(1):
        # Render camera image and generate initial point cloud from the simulation.
        color, depth, _ = env.render_camera()
        # Store original depth for comparison
        depth_original = depth.copy()
        # Apply depth smoothing for better point cloud quality
        depth = smooth_depth_map(depth)
        
        # Visualize depth map comparison in MeshCat
        # visualize_depth_map_meshcat(vis, depth_original, depth)
        
        pcd = env.create_pointcloud(color, depth)
        # Prepare visual prompt: segment objects, create masks, and mark them on the image.
        image, seg = env.obs['image'], env.obs['seg']
        obj_ids = np.unique(seg)[1:]
        all_masks = np.stack([seg == objID for objID in obj_ids])
        marker_data = {'masks': all_masks, 'labels': obj_ids}
        visual_prompt, _ = grounder.prepare_image_prompt(image.copy(), marker_data)
        marked_image_grounding = visual_prompt[-1]
        print("Displaying the visual prompt sent to GPT-4o...")
        display_image(marked_image_grounding, (6,6))
        print("Calling GPT-4o...")
        # Identify target objects using GPT-4o with the text query and visual prompt.
        _, _, target_ids = grounder.request(text_query=USER_QUERY,image=image.copy(),data=marker_data)
        if len(target_ids) == 0:
            print("No target objects identified by GPT-4o. Exiting.")
            exit(0)
        
        # Randomly select one of the identified target objects.
        target_idx = len(target_ids) - 1
        selected_target_id = target_ids[target_idx]
        print(f"Picked obj {selected_target_id}")
        # Compute the 2D center of mass of the selected target object's mask.
        center_x, center_y = compute_mask_center_of_mass(marker_data["masks"][marker_data["labels"].tolist().index(selected_target_id)])
        assert center_x is not None and center_y is not None, "No object clicked"
        
        # Store original point cloud for comparison
        pcd_original = pcd
        
        # Clean and downsample the point cloud for faster processing.
        # This replaces the old uniform downsampling with:
        # 1. Statistical outlier removal for cleaner point clouds
        # 2. Voxel-based downsampling for better spatial consistency
        pcd_ds = clean_and_downsample_pcd(pcd, voxel_size=0.005)
        
        # Visualize point cloud cleaning comparison in MeshCat
        # visualize_point_cloud_comparison_meshcat(vis, pcd_original, pcd_ds, "cleaning_and_downsampling")
        
        # Optional: Print point cloud statistics for debugging
        print(f"📊 Original point cloud: {len(pcd.points)} points")
        print(f"📊 Cleaned & downsampled: {len(pcd_ds.points)} points")
        
        # Enhanced grasp point detection: Use 3D geometric analysis instead of simple 2D methods
        
        # Get the mask for the selected target object
        target_mask = marker_data["masks"][marker_data["labels"].tolist().index(selected_target_id)]
        
        # Get camera intrinsics from the simulation for accurate 3D analysis
        camera_intrinsics = env.realsensed435_cam[0]["intrinsics"]
        print(f"📷 Using camera intrinsics: fx={camera_intrinsics[0,0]:.1f}, fy={camera_intrinsics[1,1]:.1f}, cx={camera_intrinsics[0,2]:.1f}, cy={camera_intrinsics[1,2]:.1f}")
        
        # Initialize target points for cropping and grasp generation
        center_x, center_y = None, None
        target_for_prediction_rob = None
        valid_3d_point_cam = None

        # Attempt 1: Use 3D optimal grasp point detection
        print("🔬 Attempting 3D optimal grasp point detection...")
        valid_3d_point_cam = compute_optimal_grasp_point_3d(pcd_original, target_mask, depth, camera_intrinsics)

        if valid_3d_point_cam is not None:
            print(f"🎯 Method 1 (3D Optimal): Found 3D point in camera frame: [{valid_3d_point_cam[0]:.3f}, {valid_3d_point_cam[1]:.3f}, {valid_3d_point_cam[2]:.3f}]")
            # Project to 2D for cropping
            projected_2d = project_3d_to_image_plane(valid_3d_point_cam.reshape(1, -1), camera_intrinsics)
            center_x, center_y = int(projected_2d[0, 0]), int(projected_2d[0, 1])
            # Transform to robot frame for grasp generation
            _, target_for_prediction_rob = transform_cam_to_rob(np.eye(3), valid_3d_point_cam)
            print(f"   Projected to 2D: ({center_x}, {center_y}), Transformed to Robot: [{target_for_prediction_rob[0]:.3f}, {target_for_prediction_rob[1]:.3f}, {target_for_prediction_rob[2]:.3f}]")
        else:
            print("⚠️ Method 1 (3D Optimal) failed.")

        # Attempt 2: Fallback to 2D geometric analysis if 3D failed
        if target_for_prediction_rob is None:
            print("🔬 Attempting 2D geometric grasp point detection...")
            optimal_2d_result = compute_optimal_grasp_point(target_mask, depth)
            if optimal_2d_result is not None:
                center_x, center_y = optimal_2d_result
                valid_3d_point_cam = map_2d_to_3d_accurate(center_x, center_y, depth, camera_intrinsics)
                if valid_3d_point_cam is not None:
                    print(f"🎯 Method 2 (2D Geometric): Found 2D point ({center_x}, {center_y}), mapped to 3D Cam: [{valid_3d_point_cam[0]:.3f}, {valid_3d_point_cam[1]:.3f}, {valid_3d_point_cam[2]:.3f}]")
                    _, target_for_prediction_rob = transform_cam_to_rob(np.eye(3), valid_3d_point_cam)
                    print(f"   Transformed to Robot: [{target_for_prediction_rob[0]:.3f}, {target_for_prediction_rob[1]:.3f}, {target_for_prediction_rob[2]:.3f}]")
                else:
                    print("⚠️ Method 2 (2D Geometric) failed to map 2D to 3D.")
            else:
                print("⚠️ Method 2 (2D Geometric) failed to find 2D point.")

        # Attempt 3: Final fallback to basic center of mass
        if target_for_prediction_rob is None:
            print("🔬 Attempting 2D center of mass grasp point detection...")
            com_result = compute_mask_center_of_mass(target_mask)
            if com_result is not None:
                center_x, center_y = com_result
                valid_3d_point_cam = map_2d_to_3d_accurate(center_x, center_y, depth, camera_intrinsics)
                if valid_3d_point_cam is not None:
                    print(f"🎯 Method 3 (Center of Mass): Found 2D point ({center_x}, {center_y}), mapped to 3D Cam: [{valid_3d_point_cam[0]:.3f}, {valid_3d_point_cam[1]:.3f}, {valid_3d_point_cam[2]:.3f}]")
                    _, target_for_prediction_rob = transform_cam_to_rob(np.eye(3), valid_3d_point_cam)
                    print(f"   Transformed to Robot: [{target_for_prediction_rob[0]:.3f}, {target_for_prediction_rob[1]:.3f}, {target_for_prediction_rob[2]:.3f}]")
                else:
                    print("⚠️ Method 3 (Center of Mass) failed to map 2D to 3D.")
            else:
                print("⚠️ Method 3 (Center of Mass) failed to find 2D point.")
        
        assert center_x is not None and center_y is not None, "No valid 2D grasp point found for cropping after all fallbacks."
        assert target_for_prediction_rob is not None, "No valid 3D target point in robot frame found for grasp generation after all fallbacks."
        
        print(f" চূড়ান্ত लक्ष्य: Cropping at 2D ({center_x}, {center_y}), Grasping around 3D Robot [{target_for_prediction_rob[0]:.3f}, {target_for_prediction_rob[1]:.3f}, {target_for_prediction_rob[2]:.3f}]")

        # Crop the point cloud around the target object using the determined 2D point
        print("Requesting Point Cloud Cropping service...")
        if valid_3d_point_cam is not None:
            print(f"💡 Using 3D point for cropping: {valid_3d_point_cam}")
            cropped_pcd_data = client.crop_point_cloud(pcd_ds, center_x, center_y, target_point_cam_frame=valid_3d_point_cam)
        else:
            print(f"💡 Using 2D point ({center_x}, {center_y}) for cropping, as 3D point is not available.")
            cropped_pcd_data = client.crop_point_cloud(pcd_ds, center_x, center_y)

        # Convert service response back to Open3D point cloud format
        cropped_pcd_cam_frame = o3d.geometry.PointCloud()
        cropped_pcd_cam_frame.points = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.points))
        cropped_pcd_cam_frame.colors = o3d.utility.Vector3dVector(np.array(cropped_pcd_data.colors))

        # Apply additional enhancement specifically for grasping
        print("🚀 Applying grasp-optimized enhancement to cropped point cloud...")
        cropped_pcd_cam_frame = enhance_point_cloud_for_grasping(cropped_pcd_cam_frame)

        # Visualize cropped vs original point cloud in MeshCat
        # visualize_point_cloud_comparison_meshcat(vis, pcd_ds, cropped_pcd_cam_frame, "cropping_and_enhancement")

        # Optional: Visualize the cleaned and cropped point cloud for debugging
        # visualize_pcd(cropped_pcd_cam_frame, "Cleaned and Cropped Point Cloud")

        # Use the cleaned and cropped point cloud directly (no upsampling needed with voxel filtering)
        cropped_pcd_crop_full_cam_frame = cropped_pcd_cam_frame

        # 🎬 Create comprehensive visualization showing the entire processing pipeline
        # visualize_complete_pipeline_meshcat(
        #     vis, 
        #     depth_original, 
        #     depth, 
        #     pcd_original, 
        #     pcd_ds, 
        #     cropped_pcd_cam_frame
        # )

        # -------------------------------------------------------------------------
        # Step 6: Coordinate Frame Transformations
        # -------------------------------------------------------------------------
        print("Transforming point clouds to robot coordinate frame...")
        # Transform cropped point cloud from camera frame to robot base frame
        # This is necessary because grasp planning works in robot coordinates
        cropped_pcd_robot_frame = transform_pcd_cam_to_rob(cropped_pcd_crop_full_cam_frame)
        
        # Also transform full scene point cloud for visualization
        pcd_robot_frame = transform_pcd_cam_to_rob(pcd)
        
        # Prepare cropped point cloud data for grasp prediction service
        cropped_pcd_data_robot_frame = PointCloudData(
            points=np.array(cropped_pcd_robot_frame.points).tolist(),
            colors=np.array(cropped_pcd_robot_frame.colors).tolist()
        )

        # -------------------------------------------------------------------------
        # Step 7: Grasp Prediction Service
        # -------------------------------------------------------------------------
        print("Requesting Grasp Prediction service...")
        
        # Call external ML service to predict grasp poses on the cropped object
        # Returns 6DOF grasp poses (position + orientation) in robot frame
        grasp_prediction_response = client.predict_grasps(cropped_pcd_data_robot_frame)
        predicted_grasps_robot_frame = grasp_prediction_response.grasps if grasp_prediction_response else []

        print(f"Generated {len(predicted_grasps_robot_frame)} potential grasp candidates")

        # -------------------------------------------------------------------------
        # Step 7.5: Validate Grasp Pipeline Consistency
        # -------------------------------------------------------------------------
        print("🔍 Validating grasp pipeline consistency...")
        
        # Validate the entire pipeline from 2D detection to 3D grasps
        validation_results = verify_grasp_pipeline_consistency(
            target_mask=target_mask,
            grasp_2d=(center_x, center_y),
            cropped_pcd=cropped_pcd_robot_frame,  # Use ROBOT frame for consistency with grasps
            generated_grasps=predicted_grasps_robot_frame,
            camera_intrinsics=camera_intrinsics,
            depth_image=depth
        )
        
        # Additional validation: Check if grasps are actually positioned on target object
        # This is now largely handled by verify_grasp_pipeline_consistency, 
        # but we can keep a separate check if needed or make tolerance stricter.
        print("🔍 Validating grasps are positioned on target object (using robot frame PCD)...")
        object_grasps = validate_grasps_on_object(
            predicted_grasps_robot_frame, 
            cropped_pcd_robot_frame, # Ensure this is the PCD in ROBOT frame
            tolerance=0.05  # Stricter tolerance: 5cm
        )
        
        if len(object_grasps) == 0:
            print("⚠️ WARNING: No grasps are positioned on the target object!")
            print("This indicates a coordinate frame or positioning issue.")
            
            # Debug: Show where the 2D grasp point maps to in 3D
            grasp_3d_cam = map_2d_to_3d_accurate(center_x, center_y, depth, camera_intrinsics)
            if grasp_3d_cam is not None:
                print(f"🔍 2D grasp point maps to 3D camera frame: [{grasp_3d_cam[0]:.3f}, {grasp_3d_cam[1]:.3f}, {grasp_3d_cam[2]:.3f}]")
                
                # Transform to robot frame for comparison
                grasp_rot_identity = np.eye(3)
                _, grasp_3d_robot = transform_cam_to_rob(grasp_rot_identity, grasp_3d_cam)
                print(f"🔍 2D grasp point in robot frame: [{grasp_3d_robot[0]:.3f}, {grasp_3d_robot[1]:.3f}, {grasp_3d_robot[2]:.3f}]")
                
                # Compare with generated grasp positions
                print("🔍 Generated grasp positions vs target location:")
                for i, grasp in enumerate(predicted_grasps_robot_frame):
                    pos = grasp.translation
                    distance = np.linalg.norm(np.array(pos) - grasp_3d_robot)
                    print(f"   Grasp {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] - {distance:.3f}m from target")
        else:
            print(f"✅ {len(object_grasps)} grasps are properly positioned on the object (robot frame check)")

        # -------------------------------------------------------------------------
        # Step 8: Grasp Filtering Service
        # -------------------------------------------------------------------------
        print("Requesting Grasp Filtering service...")
        
        # Call external service to filter grasps for kinematic reachability
        # This ensures the robot can actually achieve the predicted grasp poses
        filter_response = client.filter_grasps(predicted_grasps_robot_frame)
        valid_grasp_idxs = filter_response.valid_grasp_idxs
        valid_grasp_joint_angles = filter_response.valid_grasp_joint_angles

        # Check if any valid grasps were found
        if not valid_grasp_idxs:
            print("No valid grasps found after filtering.")
            return

        # Extract valid grasps from the full set of predictions
        valid_grasps: List[Grasp] = [predicted_grasps_robot_frame[i] for i in valid_grasp_idxs]
        
        print(f"✅ Found {len(valid_grasps)} kinematically valid grasps")

        # -------------------------------------------------------------------------
        # Step 9: Grasp Selection and Visualization
        # -------------------------------------------------------------------------
        
        # Select the first valid grasp for execution
        # In a real application, you might rank grasps by quality metrics
        chosen_grasp_idx = 0
        chosen_grasp = valid_grasps[chosen_grasp_idx]
        
        print(f"Selected grasp {chosen_grasp_idx + 1} out of {len(valid_grasps)} valid options")
        print(f"Grasp position: [{chosen_grasp.translation[0]:.3f}, {chosen_grasp.translation[1]:.3f}, {chosen_grasp.translation[2]:.3f}]")

        # Visualize all valid grasps in 3D viewer
        print("Launching 3D visualization of valid grasps...")
        print("Check the MeshCat visualizer to see the grasp poses")
        
        # Clear any previous visualizations to avoid confusion
        vis.delete()
        
        vis_grasps_meshcat(vis, valid_grasps, pcd_robot_frame)

        # Add visual debug marker at chosen grasp location in simulation
        env.add_debug_point(chosen_grasp.translation)

        # -------------------------------------------------------------------------
        # Step 10: Grasp Execution
        # -------------------------------------------------------------------------
        print("Executing the selected grasp...")
        
        # Get joint angles for the chosen grasp from filtering service
        grasp_joint_angles = valid_grasp_joint_angles[chosen_grasp_idx]
        
        # Convert grasp to pose format [x, y, z, qx, qy, qz, qw]
        grasp_pos = chosen_grasp.translation
        grasp_rot_matrix = np.array(chosen_grasp.rotation)
        grasp_quat = R.from_matrix(grasp_rot_matrix).as_quat()  # [x, y, z, w]
        
        # Convert to [x, y, z, qx, qy, qz, qw] format expected by execute_grasp_sequence
        target_pose = grasp_pos + grasp_quat.tolist()
        
        print(f"🎯 Executing grasp at position [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
        print("🤖 Performing grasp sequence: approach → grasp → close → lift...")
        env.execute_grasp_sequence(target_pose)
        
        # Transport grasped object to tray
        print("Transporting object to tray...")
        env.drop_object_in_tray()
        
        # Keep simulation running so you can see the result
        print("🎬 Task completed! Simulation will continue for 10 seconds so you can see the result...")
        for i in range(10):
            print(f"⏰ {10-i} seconds remaining...")
            for _ in range(30):  # 1 second at 30Hz
                env.step_simulation()
        print("🏁 Demo finished!")


if __name__ == "__main__":
    main()
