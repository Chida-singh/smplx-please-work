import streamlit as st
import os
import json
import subprocess
from word_to_smplx import WordToSMPLX
import imageio
import numpy as np

st.set_page_config(page_title="SMPL-X Animation Demo", layout="centered")
st.title("SMPL-X Animation Demo")

# Load available words from mapping
current_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(current_dir, "filtered_video_to_gloss.json")
dataset_dir = os.path.join(current_dir, "word-level-dataset-cpu")
output_dir = os.path.join(current_dir, "output")
os.makedirs(output_dir, exist_ok=True)

with open(mapping_path, "r") as f:
    gloss_map = json.load(f)
word_to_pkl = {v.lower(): k for k, v in gloss_map.items()}
all_words = sorted(word_to_pkl.keys())

# UI: Multi-select for words
st.markdown("### Select one or more words to animate:")
selected_words = st.multiselect("Words", all_words)

# UI: Button to generate animation(s)
if st.button("Generate Animation"):
    if not selected_words:
        st.warning("Please select at least one word.")
    else:
        animator = WordToSMPLX(model_path=os.path.join(current_dir, "models"))
        video_paths = []
        for word in selected_words:
            pkl_file = os.path.join(dataset_dir, word_to_pkl[word])
            output_path = os.path.join(output_dir, f"{word}_animation.mp4")
            # Generate animation if not already present
            if not os.path.exists(output_path):
                pose_data = animator.load_pose_sequence(pkl_file)
                animator.render_animation(pose_data, save_path=output_path, fps=15)
            video_paths.append(output_path)
            st.success(f"Animation for '{word}' ready.")
        # If only one word, show its video
        if len(video_paths) == 1:
            st.video(video_paths[0])
        else:
            # Concatenate videos
            combined_path = os.path.join(output_dir, "combined_animation.mp4")
            # Use imageio to concatenate
            clips = []
            for path in video_paths:
                reader = imageio.get_reader(path)
                clips.extend([im for im in reader])
                reader.close()
            # Save combined video
            imageio.mimsave(combined_path, clips, fps=15)
            st.success("Combined animation ready!")
            st.video(combined_path)

st.markdown("---")
st.markdown("**Instructions:**\n- Select one or more words from the list.\n- Click 'Generate Animation' to view the animation.\n- If you select multiple words, their animations will be played in sequence as a single video.")

def blend_pose_sequences(seq_a, seq_b, n_blend=5):
    # seq_a, seq_b: [N, D] numpy arrays
    if n_blend == 0 or len(seq_a) < n_blend or len(seq_b) < n_blend:
        return np.vstack([seq_a, seq_b])
    blend = []
    for i in range(n_blend):
        alpha = (i + 1) / (n_blend + 1)
        blended = (1 - alpha) * seq_a[-n_blend + i] + alpha * seq_b[i]
        blend.append(blended)
    blend = np.stack(blend)
    return np.vstack([seq_a[:-n_blend], blend, seq_b[n_blend:]]) 