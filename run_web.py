"""Start the real backend and keep it running until the user presses Ctrl+C."""
from pathlib import Path
import os
import sys


def main():
    os.chdir(Path(__file__).resolve().parent)
    try:
        import uvicorn
    except ImportError:
        print('Chua cai thu vien web. Trong VS Code, chay:')
        print(f'"{sys.executable}" -m pip install -r requirements-web.txt')
        return 1
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '8000'))
    print(f'\nMo trinh duyet tai http://127.0.0.1:{port}')
    print('GIU CUA SO NAY MO khi dung web. Nhan Ctrl+C de dung.\n')
    uvicorn.run('app.main:app', host=host, port=port)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
