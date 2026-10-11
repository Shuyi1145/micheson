from __future__ import annotations

import argparse
import json
from pathlib import Path

from michelson.camera import DemoCamera, UsbCamera
from michelson.gui import run_gui


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="迈克耳孙干涉波长视觉测量")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--demo", action="store_true", help="使用合成干涉图样，不连接摄像头")
    parser.add_argument("--camera", type=int, help="临时覆盖 USB 摄像头编号")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.camera is not None:
        config["camera"]["index"] = args.camera
    camera = DemoCamera(config["camera"]) if args.demo else UsbCamera(config["camera"])
    run_gui(camera, config, demo_mode=args.demo)


if __name__ == "__main__":
    main()

