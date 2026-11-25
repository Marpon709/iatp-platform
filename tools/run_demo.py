"""运行无需地图 Key 和完整研究数据的本地样例展示。"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def configure():
    parser = argparse.ArgumentParser(description='IATP 本地样例与计算记录展示')
    parser.add_argument('--port', type=int, default=8890)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port 须为 1 至 65535')
    os.environ.setdefault('IATP_SHOWCASE_BACKEND_ROOT', str(ROOT / 'resources/backend'))
    os.environ.setdefault('IATP_SHOWCASE_STATIC_ROOT', str(ROOT / 'editable-app/web'))
    os.environ.setdefault('IATP_SHOWCASE_WORK_ROOT', str(ROOT / '.local/demo'))
    sys.path.insert(0, str(ROOT / 'resources/bridge'))
    return args


if __name__ == '__main__':
    args = configure()
    from service import main
    print('本地演示：http://127.0.0.1:%d/' % args.port, flush=True)
    print('可浏览样例、复核保存的路线结果和查询记录；不启动完整空间计算引擎。', flush=True)
    sys.argv = [sys.argv[0], '--skip-backend-start', '--port', str(args.port)]
    main()
