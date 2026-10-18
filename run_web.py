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
    print('\nMo trinh duyet tai http://127.0.0.1:8000')
    print('GIU CUA SO NAY MO khi dung web. Nhan Ctrl+C de dung.\n')
    uvicorn.run('app.main:app', host='127.0.0.1', port=8000)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
