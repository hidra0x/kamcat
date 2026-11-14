"""
GitHub Push Tool — заливает проект в репозиторий через API.
"""

import os
import sys
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

GITHUB_API = "https://api.github.com"


def get_user(token):
    req = urllib.request.Request(
        f"{GITHUB_API}/user",
        headers={"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)["login"]
    except urllib.error.HTTPError as e:
        print(f"❌ Ошибка авторизации: {e.code}")
        return None


def create_repo(token, repo_name, description=""):
    data = json.dumps({
        "name": repo_name,
        "description": description,
        "private": False,
        "auto_init": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{GITHUB_API}/user/repos",
        data=data,
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        if e.code == 422 and "already exists" in body:
            print(f"⚠️ Репозиторий уже существует, используем его")
            return {"name": repo_name}
        print(f"❌ Ошибка создания репозитория: {e.code} — {body[:200]}")
        return None


def upload_file(token, owner, repo, path, content_bytes, message="Add file"):
    data = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("utf-8"),
    }
    req = urllib.request.Request(
        f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
        data=json.dumps(data).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        if e.code == 422:
            return None  # файл уже есть
        print(f"  ❌ {path}: {e.code} — {body[:100]}")
        return None


def should_skip(path):
    skip_dirs = {"venv", "__pycache__", ".git", "data", "test", "test_batch",
                 "node_modules", ".cache", "trailer/node_modules", "wheels"}
    skip_suffixes = {".pyc", ".pyo", ".pth", ".pt", ".onnx", ".safetensors",
                     ".zip", ".tar", ".gz", ".webm", ".mp4", ".ico", ".lnk"}
    parts = path.parts
    for d in skip_dirs:
        if d in parts:
            return True
    if path.suffix in skip_suffixes:
        return True
    try:
        if path.stat().st_size > 10 * 1024 * 1024:
            return True
    except Exception:
        return True
    return False


def push_project(token, owner, repo, project_dir):
    project_dir = Path(project_dir)
    files = []
    for p in project_dir.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(project_dir)
                if not should_skip(rel):
                    files.append(p)
            except Exception:
                continue

    print(f"\n📤 Файлов к загрузке: {len(files)}")
    print("=" * 60)

    ok = 0
    for i, p in enumerate(files, 1):
        rel = p.relative_to(project_dir).as_posix()
        print(f"  [{i}/{len(files)}] {rel}")
        try:
            content = p.read_bytes()
            result = upload_file(token, owner, repo, rel, content, message=f"Add {rel}")
            if result:
                ok += 1
        except Exception as e:
            print(f"    ❌ {e}")

    print("=" * 60)
    print(f"✅ Загружено: {ok}/{len(files)}")
    print(f"\n🔗 https://github.com/{owner}/{repo}")


def main():
    print("=" * 60)
    print("🐙 GitHub Push Tool — KamCat")
    print("=" * 60)

    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("\n🔑 Вставь Personal Access Token (ghp_...):")
        print("   (ввод невидимый — это нормально)")
        token = input("Токен: ").strip()

    if not token:
        print("❌ Токен не введён")
        return

    print("\n🔍 Проверяем токен...")
    owner = get_user(token)
    if not owner:
        return
    print(f"✅ Авторизован: {owner}")

    repo = input("\n📦 Имя репозитория [kamcat]: ").strip() or "kamcat"
    desc = input("📝 Описание [AI Toolkit: cat/dog classifier, YOLO, SD, own Transformer]: ").strip()
    if not desc:
        desc = "AI Toolkit: cat/dog classifier, YOLO, SD, own Transformer"

    print(f"\n📦 Создаём репозиторий {owner}/{repo}...")
    if not create_repo(token, repo, desc):
        return

    print(f"\n📤 Заливаем проект...")
    push_project(token, owner, repo, ".")

    print(f"\n🎉 Готово! https://github.com/{owner}/{repo}")


if __name__ == "__main__":
    main()