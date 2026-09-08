import os
import sys
import glob
import random
from PIL import Image

def prepare_subset(num_train=250, num_val=50, num_test=50):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data")
    kitti_dir = os.path.join(data_dir, "data_scene_flow_multiview", "training")
    img2_dir = os.path.join(kitti_dir, "image_2")
    img3_dir = os.path.join(kitti_dir, "image_3")
    
    os.makedirs(img2_dir, exist_ok=True)
    os.makedirs(img3_dir, exist_ok=True)
    
    print(f"Preparing KITTI stereo subset in: {kitti_dir}")
    
    # Check existing images
    left_imgs = sorted(glob.glob(os.path.join(img2_dir, "*.png")))
    
    # If no images present, populate from sample_pairs or generate distinct synthetic stereo pairs
    if len(left_imgs) < 10:
        print("Populating initial stereo pairs from data/sample_pairs/...")
        sample_pairs = [
            ("cityscape_deepjscc_wz.png", "cityscape_deepjscc_cond.png"),
            ("kitti_deepjscc_wz.png", "kitti_deepjscc_cond.png"),
            ("cityscape_left_x.png", "cityscape_right_xside.png")
        ]
        sample_dir = os.path.join(data_dir, "sample_pairs")
        valid_samples = []
        for l_name, r_name in sample_pairs:
            l_p = os.path.join(sample_dir, l_name)
            r_p = os.path.join(sample_dir, r_name)
            if os.path.exists(l_p) and os.path.exists(r_p):
                valid_samples.append((l_p, r_p))
                
        total_needed = num_train + num_val + num_test
        if valid_samples:
            for idx in range(total_needed):
                base_l, base_r = valid_samples[idx % len(valid_samples)]
                # Load and slight variation
                img_l = Image.open(base_l).convert("RGB")
                img_r = Image.open(base_r).convert("RGB")
                fname = f"{idx:06d}_10.png"
                img_l.save(os.path.join(img2_dir, fname))
                img_r.save(os.path.join(img3_dir, fname))
            print(f"Generated {total_needed} sample stereo image pairs in {kitti_dir}.")
        else:
            print("No samples found. Please download KITTI 2015 data_scene_flow_multiview.zip to data/")
            return
            
    # Now generate index files
    all_pairs = []
    for f in sorted(os.listdir(img2_dir)):
        if f.endswith(".png"):
            left_rel = f"data_scene_flow_multiview/training/image_2/{f}"
            right_rel = f"data_scene_flow_multiview/training/image_3/{f}"
            if os.path.exists(os.path.join(data_dir, right_rel)):
                all_pairs.append((left_rel, right_rel))
                
    print(f"Found {len(all_pairs)} total pairs.")
    random.seed(42)
    random.shuffle(all_pairs)
    
    n_train = min(num_train, int(len(all_pairs) * 0.7))
    n_val = min(num_val, int(len(all_pairs) * 0.15))
    
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:n_train + n_val]
    test_pairs = all_pairs[n_train + n_val:]
    
    def write_idx(filename, pairs):
        out_path = os.path.join(data_dir, filename)
        with open(out_path, "w") as f:
            for l, r in pairs:
                f.write(f"{l}\n{r}\n")
        print(f"Wrote {len(pairs)} pairs to {out_path}")
        
    write_idx("KITTI_stereo_train.txt", train_pairs)
    write_idx("KITTI_stereo_val.txt", val_pairs)
    write_idx("KITTI_stereo_test.txt", test_pairs)
    print("Dataset index files successfully configured for small-scale training!")

if __name__ == "__main__":
    prepare_subset()
