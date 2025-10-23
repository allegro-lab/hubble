import argparse
import huggingface_hub

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True)
    parser.add_argument('--revision', type=str, default='main')
    parser.add_argument('--local_dir', type=str, required=True)
    args = parser.parse_args()

    huggingface_hub.snapshot_download(args.repo_id, revision=args.revision, local_dir=args.local_dir)
