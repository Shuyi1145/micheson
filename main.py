# -*- coding: gb2312 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from michelson.camera import DemoCamera, UsbCamera


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="迈克耳孙干涉波长视觉测量")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")), help="配置文件路径")
    parser.add_argument("--demo", action="store_true", help="使用合成干涉图样，不连接摄像头")
    parser.add_argument("--camera", type=int, help="临时覆盖 USB 摄像头编号")
    parser.add_argument("--web", action="store_true", help="启用网页图传")
    parser.add_argument("--headless", action="store_true", help="仅图传，不打开桌面测量界面（需 --web）")
    parser.add_argument("--host", default="127.0.0.1", help="图传监听地址；局域网用 0.0.0.0")
    parser.add_argument("--port", type=int, default=8080, help="图传端口，默认 8080")
    parser.add_argument("--video", help="逐帧分析本地视频并导出原图/开闭运算对比")
    parser.add_argument("--play-video", help="循环播放本地视频演示，可与 --web 配合")
    parser.add_argument("--roi-center", type=int, nargs=2, metavar=("X", "Y"), help="原视频像素坐标")
    parser.add_argument("--video-start", type=float, default=0, help="视频分析起点 / s")
    parser.add_argument("--video-end", type=float, help="视频分析终点 / s")
    parser.add_argument("--no-preview-export", action="store_true", help="仅分析视频，不导出对比 MP4")
    args = parser.parse_args()
    if args.headless and not args.web:
        parser.error("--headless 需要同时指定 --web")
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1-65535 范围内")

    config = load_config(args.config)
    if args.play_video and (args.video or args.demo or args.camera is not None):
        parser.error("--play-video 不能与 --video/--demo/--camera 混用")
    if args.video:
        if args.demo or args.web or args.headless or args.camera is not None:
            parser.error("视频离线分析不能与 demo/web/headless/camera 混用")
        from michelson.video import analyze_video
        directory, result = analyze_video(args.video, config,
            center=None if args.roi_center is None else tuple(args.roi_center),
            start_s=args.video_start, end_s=args.video_end,
            export_preview=not args.no_preview_export,
            progress=lambda count, stamp: print(f"已分析 {count} 帧，视频时间 {stamp:.1f} s"))
        print(f"结果目录：{directory}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.camera is not None:
        config["camera"]["index"] = args.camera
    using_video = not args.demo and args.camera is None
    if using_video:
        from michelson.playback import VideoPlayback
        video_path = Path(args.play_video or config.get("video", {}).get("path", "拍摄视频.mp4"))
        if not video_path.is_absolute():
            video_path = Path(args.config).resolve().parent / video_path
        camera = VideoPlayback(video_path)
        configured_path = Path(config.get("video", {}).get("path", "拍摄视频.mp4"))
        if not configured_path.is_absolute():
            configured_path = Path(args.config).resolve().parent / configured_path
        if video_path.resolve() != configured_path.resolve():
            # A different file must not inherit the current experiment's readings.
            config = dict(config, video={"path": str(video_path)})
    else:
        camera = DemoCamera(config["camera"]) if args.demo else UsbCamera(config["camera"])
    source = camera
    from michelson.preview import DisplaySettings
    display_settings = DisplaySettings(config.get("preview", {}))
    server = None
    try:
        if args.web:
            from michelson.webstream import FrameRelay, PreviewServer, StreamCamera
            relay = FrameRelay(display_settings)
            if using_video:
                relay.source_info = source.playback_info
            server = PreviewServer(relay, args.host, args.port)
            camera = StreamCamera(camera, relay)
            server.start()
            print(f"网页图传端口：{server.address[1]}；本机 http://127.0.0.1:{server.address[1]}")
            if args.host == "0.0.0.0":
                print("局域网访问：http://香橙派IP:" + str(server.address[1]))
                print("无登录认证，仅限可信网络；不要转发此端口到公网。")
        if args.headless:
            import time
            while True:
                ok, _, _ = camera.read()
                if not ok:
                    time.sleep(0.1)
        else:
            from michelson.gui import run_gui
            run_gui(camera, config, demo_mode=args.demo,
                    playback=source if using_video else None,
                    center=tuple(args.roi_center) if args.roi_center else (
                        tuple(config["video"]["center_px"])
                        if using_video and config.get("video", {}).get("center_px") else None),
                    display_settings=display_settings)
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        if using_video:
            source.release()
        if server is not None:
            server.close()


if __name__ == "__main__":
    main()

