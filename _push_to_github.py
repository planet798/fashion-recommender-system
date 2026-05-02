"""Push all source files to github via Contents API (individual commits)"""
import os, sys, json, base64, subprocess, glob, urllib.request, urllib.error, urllib.parse

REPO = "planet798/fashion-recommender-system"
BRANCH = "main"

token = subprocess.run(
    ["C:/Program Files/GitHub CLI/gh.exe", "auth", "token"],
    capture_output=True, text=True
).stdout.strip()

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github.v3+json",
}

def api(method, path, data=None, timeout=30):
    url = f"https://api.github.com{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    if body:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = resp.read().decode()
            return json.loads(result) if result else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"  API Error {e.code}: {err[:200]}")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

# Collect files
include_exts = ('.py', '.sql', '.txt', '.md')
files = ['.gitignore', 'README.md']
for ext in include_exts:
    files.extend(glob.glob(f"**/*{ext}", recursive=True))

exclude_patterns = ['.venv', '__pycache__', '.git', '.claude',
                    'data/', 'datasets/', 'results/', 'models/', 'node_modules']
files = [f for f in files
         if not any(p in f.replace('\\', '/') for p in exclude_patterns)]
files = sorted(set(files))

print(f"Uploading {len(files)} files individually...", flush=True)

count = 0
for fpath in files:
    path_in_repo = fpath.replace('\\', '/')

    with open(fpath, 'rb') as f:
        content = f.read()

    b64 = base64.b64encode(content).decode()

    encoded_path = urllib.parse.quote(path_in_repo, safe='')
    result = api("PUT", f"/repos/{REPO}/contents/{encoded_path}", {
        "message": f"Add {path_in_repo}",
        "content": b64,
        "branch": BRANCH,
    })

    if result:
        count += 1
        print(f"  [{count}/{len(files)}] {path_in_repo}", flush=True)
    else:
        print(f"  FAILED: {path_in_repo}", flush=True)

print(f"\nDone! {count}/{len(files)} files uploaded to https://github.com/{REPO}", flush=True)
