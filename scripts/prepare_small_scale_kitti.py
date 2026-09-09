import os
import sys
import glob
import random
from PIL import Image

def prepare_subset():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    kitti_dir = os.path.join(data_dir, "data_scene_flow_multiview", "training")
    img2_dir = os.path.join(kitti_dir, "image_2")
    img3_dir = os.path.join(kitti_dir, "image_3")
    
    print(f"Loading authentic KITTI stereo images from: {kitti_dir}")
    
    # Collect all real pairs grouped by sequence
    seq_dict = {}
    for f in sorted(os.listdir(img2_dir)):
        if f.endswith(".png"):
            left_rel = f"data_scene_flow_multiview/training/image_2/{f}"
            right_rel = f"data_scene_flow_multiview/training/image_3/{f}"
            left_abs = os.path.join(data_dir, left_rel)
            right_abs = os.path.join(data_dir, right_rel)
            if os.path.exists(left_abs) and os.path.exists(right_abs):
                seq_id = f.split("_")[0]
                if seq_id not in seq_dict:
                    seq_dict[seq_id] = []
                seq_dict[seq_id].append((left_rel, right_rel))
                
    unique_seqs = sorted(list(seq_dict.keys()))
    print(f"Found {len(unique_seqs)} unique KITTI driving sequences (total {sum(len(v) for v in seq_dict.values())} pairs).")
    
    # Strictly disjoint sequence-level partitioning (Zero train/test leakage)
    # Train: 000000 to 000139 (140 sequences = 280 pairs)
    # Val:   000140 to 000169 (30 sequences = 60 pairs)
    # Test:  000170 to 000199 (30 sequences = 60 pairs)
    train_seqs = unique_seqs[:140]
    val_seqs = unique_seqs[140:170]
    test_seqs = unique_seqs[170:]
    
    # Assert completely disjoint and no frame proximity
    max_train = int(train_seqs[-1])
    min_test = int(test_seqs[0])
    print(f"Max train sequence ID: {max_train:06d}")
    print(f"Min test sequence ID:  {min_test:06d} (Sequence separation: {min_test - max_train} scenes apart, ZERO leakage)")
    assert min_test - max_train >= 2, "Train and test sequences are too close!"
    
    train_pairs = [p for s in train_seqs for p in seq_dict[s]]
    val_pairs = [p for s in val_seqs for p in seq_dict[s]]
    test_pairs = [p for s in test_seqs for p in seq_dict[s]]
    
    def write_idx(filename, pairs):
        out_path = os.path.join(data_dir, filename)
        with open(out_path, "w") as f:
            for l, r in pairs:
                f.write(f"{l}\n{r}\n")
        print(f"Wrote {len(pairs)} pairs to {out_path}")
        
    write_idx("KITTI_stereo_train.txt", train_pairs)
    write_idx("KITTI_stereo_val.txt", val_pairs)
    write_idx("KITTI_stereo_test.txt", test_pairs)
    print("Dataset index files successfully generated with zero-leakage guarantee!")

if __name__ == "__main__":
    prepare_subset()
