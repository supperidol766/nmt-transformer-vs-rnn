import os
from huggingface_hub import snapshot_download

REPO_ID = "balabababa/nmt-transformer-vs-rnn-weights"

def main():
    os.makedirs("checkpoints", exist_ok=True)

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir="checkpoints",
        local_dir_use_symlinks=False,
        allow_patterns=["weights/*.pt"],
    )

    print("Files are in:")
    print("  checkpoints/weights/weights_seq2seq.pt")
    print("  checkpoints/weights/weights_transformer.pt")

if __name__ == "__main__":
    main()
