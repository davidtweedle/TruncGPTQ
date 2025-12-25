import argparse


def get_args():
    parser = argparse.ArgumentParser(description="GPTQ-like Quantization")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-1.5B", help="HuggingFace model identifier")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4"], help="Calibration dataset")
    parser.add_argument("--n_samples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length for calibration")

    parser.add_argument("--w_bits", type=int, default=4, help="Target weight bits")
    parser.add_argument("--sketch_ratio", type=float, default=1.0, help="Ratio of sketch size to input dimension (d = ratio * n)")
    parser.add_argument("--k_iter", type=int, default=0, help="Power iterations for randomized SVD")
    parser.add_argument("--eps", type=float, default=1e-2, help="Singular value truncation threshold")
    parser.add_argument("--save_path", type=str, default="./quantized_model", help="Path to save output")

    return parser.parse_args()
