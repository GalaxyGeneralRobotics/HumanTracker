"""Download verified policy artifacts, never external source repositories.

Run: python -m humantracker.download_weights --tracker all
Existing files are verified and never silently replaced.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

from humantracker.eval.paths import repo_path

WEIGHTS = {
    "gmt": [
        ("gmt/pretrained.pt",
         "https://raw.githubusercontent.com/zixuan417/humanoid-general-motion-tracking/2a590de25a1eb08e47491977a738549c22f16e1f/assets/pretrained_checkpoints/pretrained.pt",
         "bf965da77571a5bb0dbb291b349b387a258c1b34cb86173cc7502f9ec2015755"),
    ],
    "twist2": [
        ("twist2/twist2_1017_25k.onnx",
         "https://raw.githubusercontent.com/amazon-far/TWIST2/d5c7108e9ef82d1b8770e5b692f27a1294f3aa8a/assets/ckpts/twist2_1017_25k.onnx",
         "126ab9089b6ca0be8b6582fc9ef17b0b6c6d6577378f9324a2e10d580fe48f2e"),
    ],
    "hgpt": [
        ("hgpt/pns_wo_priv264.onnx",
         "https://raw.githubusercontent.com/GalaxyGeneralRobotics/Humanoid-GPT/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv264.onnx",
         "bbaac94ba08d30bd96879e0c5878d60a38180f1cb819a9336cc6b56d49f333f9"),
    ],
    "hgpt216": [
        ("hgpt/pns_wo_priv216.onnx",
         "https://raw.githubusercontent.com/GalaxyGeneralRobotics/Humanoid-GPT/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv216.onnx",
         "1ca1f475dc647afb736f4cdbd55d45103c36b7f5059d830b05d6aef3a5fc3a1e"),
    ],
    "sonic": [
        ("sonic_v1_1/model_encoder.onnx",
         "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1/model_encoder.onnx",
         "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"),
        ("sonic_v1_1/model_decoder.onnx",
         "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/sonic_v1_1/model_decoder.onnx",
         "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"),
    ],
    "sonic_release": [
        ("sonic/model_encoder.onnx",
         "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_encoder.onnx",
         "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3"),
        ("sonic/model_decoder.onnx",
         "https://huggingface.co/nvidia/GEAR-SONIC/resolve/main/model_decoder.onnx",
         "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed"),
    ],
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(destination: Path, url: str, expected_sha256: str) -> Path:
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError(f"SHA-256 mismatch; existing file left unchanged: {destination}")
        print(f"[verified] {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="." + destination.name,
                                         suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
        if sha256(temporary) != expected_sha256:
            raise ValueError(f"Downloaded SHA-256 mismatch: {url}")
        # link() refuses to overwrite a destination created concurrently.
        destination.hardlink_to(temporary)
        print(f"[downloaded] {destination}")
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=["all", *WEIGHTS], default="all")
    parser.add_argument("--output_dir", default="storage/checkpoints/trackers",
                        help="absolute path or path relative to the HumanTracker repository")
    args = parser.parse_args()
    groups = WEIGHTS if args.tracker == "all" else {args.tracker: WEIGHTS[args.tracker]}
    for artifacts in groups.values():
        for relative, url, digest in artifacts:
            download(repo_path(args.output_dir) / relative, url, digest)


if __name__ == "__main__":
    main()
