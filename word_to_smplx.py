import os
import sys
import torch
import smplx
import numpy as np
import imageio
import json
from scipy.ndimage import gaussian_filter1d

# Import rendering dependencies
try:
    import pyrender
    import trimesh
except ImportError:
    pyrender = None
    trimesh = None

# --- Joint Index Reference (SMPL-X) ---
# 0: Global orientation
# 1: Root (pelvis)
# 2: Left hip
# 3: Right hip
# 4: Spine1
# 5: Left knee
# 6: Right knee
# 7: Spine2
# 8: Left ankle
# 9: Right ankle
# 10: Spine3
# 11: Left foot
# 12: Right foot
# 13: Neck
# 14: Left shoulder
# 15: Right shoulder
# 16: Head
# 17: Left elbow
# 18: Right elbow
# 19: Left wrist
# 20: Right wrist
# (Hand joints start at 40 in trial.py)

class WordToSMPLX:
    def __init__(self, model_path="models", gender='neutral', viewport_width=640, viewport_height=480):
        model_dir = os.path.join(model_path, 'smplx')
        smplx_subdir = os.path.join(model_dir, 'smplx')
        model_file = os.path.join(smplx_subdir, f"SMPLX_{gender.upper()}.npz")
        if not os.path.exists(model_file):
            raise ValueError(f"Model file not found at: {model_file}")
        
        self.smplx_model = smplx.create(
            model_path=model_dir,
            model_type='smplx',
            gender=gender,
            use_pca=False,  # Disable PCA to allow full finger control for sign language
            num_pca_comps=45,  # Full hand pose dimensions
            create_global_orient=True,
            create_body_pose=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_jaw_pose=True,
            create_leye_pose=True,
            create_reye_pose=True,
            create_betas=True,
            create_expression=True,
            create_transl=True,
            num_betas=10,
            num_expression_coeffs=10,
            flat_hand_mean=False,  # Allow curved hand poses for better clenching
            batch_size=1
        )
        
        # Renderer setup (optional)
        if pyrender and trimesh:
            self.camera = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
            self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            self.cam_pose = np.eye(4)
            self.cam_pose[2, 3] = 2.0
            self.cam_pose[1, 3] = -0.2

            self.renderer = pyrender.OffscreenRenderer(
                viewport_width=viewport_width,
                viewport_height=viewport_height
            )
        else:
            self.renderer = None

    @staticmethod
    def smooth_and_clamp_hand_pose(hand_pose_np, sigma=0.3, clamp_min=-2.0, clamp_max=2.0):
        # Smoothing
        smoothed = np.copy(hand_pose_np)
        for i in range(smoothed.shape[1]):
            smoothed[:, i] = gaussian_filter1d(smoothed[:, i], sigma=sigma)
        # Clamping
        smoothed = np.clip(smoothed, clamp_min, clamp_max)
        return smoothed

    def load_pose_sequence(self, pkl_path):
        # Always load to CPU, allow for CUDA-originated files
        with open(pkl_path, "rb") as f:
            data = torch.load(f, map_location='cpu', weights_only=False)
        return data

    def render_animation(self, pose_data, save_path=None, fps=15):
        smplx_data = pose_data['smplx']
        
        # If it's a numpy array of arrays, stack to shape [N, D]
        if isinstance(smplx_data, np.ndarray) and isinstance(smplx_data[0], np.ndarray):
            smplx_params = np.stack(smplx_data)  # shape: [N, D]
            N = smplx_params.shape[0]
            
            # Indices for SMPL-X parameters
            global_orient = torch.tensor(smplx_params[:, 0:3], dtype=torch.float32)
            body_pose = torch.tensor(smplx_params[:, 3:66], dtype=torch.float32)
            
            # Process hand poses with full anatomical constraints
            left_hand_pose_np = smplx_params[:, 66:111]
            right_hand_pose_np = smplx_params[:, 111:156]
            
            # Apply selective smoothing that preserves finger clenching movements
            left_hand_pose_np = self.smooth_and_clamp_hand_pose(left_hand_pose_np)
            right_hand_pose_np = self.smooth_and_clamp_hand_pose(right_hand_pose_np)
            
            # Process hand poses with anatomical constraints
            left_hand_pose = torch.tensor(left_hand_pose_np, dtype=torch.float32)
            right_hand_pose = torch.tensor(right_hand_pose_np, dtype=torch.float32)
        else:
            raise ValueError("Unexpected structure in 'smplx' key.")

        frames = []
        for i in range(N):
            go = global_orient[i].unsqueeze(0).clone()
            # Add 180-degree rotation around X-axis to flip the model right-side up
            go[0, 0] += np.pi  # Rotate 180° around X axis
            
            bp = body_pose[i].unsqueeze(0)
            lhp = left_hand_pose[i].unsqueeze(0)
            rhp = right_hand_pose[i].unsqueeze(0)
            
            # Additional hand pose validation before SMPL-X model call
            if torch.isnan(lhp).any() or torch.isnan(rhp).any():
                print(f"Warning: NaN detected in hand poses at frame {i}, using neutral pose")
                lhp = torch.zeros_like(lhp)
                rhp = torch.zeros_like(rhp)
            
            try:
                output = self.smplx_model(
                    body_pose=bp,
                    right_hand_pose=rhp,
                    left_hand_pose=lhp,
                    global_orient=go,
                    betas=torch.zeros((1, 10)),
                    return_verts=True
                )
            except Exception as e:
                print(f"Error in SMPL-X model at frame {i}: {e}")
                # Use neutral pose as fallback
                output = self.smplx_model(
                    body_pose=bp,
                    right_hand_pose=torch.zeros_like(rhp),
                    left_hand_pose=torch.zeros_like(lhp),
                    global_orient=go,
                    betas=torch.zeros((1, 10)),
                    return_verts=True
                )
            
            if self.renderer and pyrender and trimesh:
                vertices = output.vertices.detach().cpu().numpy().squeeze()
                mesh = trimesh.Trimesh(vertices=vertices, faces=self.smplx_model.faces)
                scene = pyrender.Scene()
                mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
                scene.add(mesh_pyrender)
                scene.add(self.camera, pose=self.cam_pose)
                scene.add(self.light, pose=self.cam_pose)
                color, _ = self.renderer.render(scene)
                frames.append(color)
            else:
                print("Rendering not available. Returning pose parameters only.")
                return output
                
        if save_path:
            imageio.mimsave(save_path, frames, fps=fps)
            
        return frames

def convert_to_cpu(input_path, output_path):
    with open(input_path, "rb") as f:
        data = torch.load(f, map_location='cpu', weights_only=False)
    for k in data:
        if hasattr(data[k], 'cpu'):
            data[k] = data[k].cpu()
    torch.save(data, output_path)

def mirror_pose(pose):
    mirrored = pose.clone()
    mirrored[..., 1::3] *= -1  # Flip Y
    mirrored[..., 2::3] *= -1  # Flip Z
    return mirrored

if __name__ == "__main__":
    print("Word to SMPL-X Animation Generator (Cleaned)")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models")
    dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu")  # Use the CPU-only dataset
    mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
    
    # Load mapping
    with open(mapping_path, "r") as f:
        gloss_map = json.load(f)
        
    # Invert mapping: word -> filename
    word_to_pkl = {v.lower(): k for k, v in gloss_map.items()}
    
    animator = WordToSMPLX(model_path=model_path)
    
    # Print all available words
    print("Available words:")
    print(", ".join(sorted(word_to_pkl.keys())))
    
    word = input("Enter a word (e.g., april, announce): ").strip().lower()
    
    try:
        if word not in word_to_pkl:
            raise ValueError(f"Word '{word}' not found in dataset.")
            
        pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
        pose_data = animator.load_pose_sequence(pkl_file)
        print(f"Loaded pose data for '{word}' from {pkl_file}")
        
        # Debug: Print keys and types
        print("Pose data keys:", pose_data.keys())
        print("global_orient:", type(pose_data.get('global_orient')))
        print("body_pose:", type(pose_data.get('body_pose')))
        print("right_hand_pose:", type(pose_data.get('right_hand_pose')))
        
        # Save animation
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{word}_animation.mp4")
        
        animator.render_animation(pose_data, save_path=output_path, fps=15)
        print(f"Animation saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()