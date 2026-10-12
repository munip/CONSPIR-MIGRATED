"""List open ConspiracyBench pull requests, least-reviewed first.

Usage: python list_open_prs.py [--me HF_USERNAME]

Skips PRs authored by --me. If --me is omitted, tries the logged-in Hugging Face
user (huggingface_hub.whoami). Only reads public data; posts nothing.
"""
import argparse
import json
import subprocess

API = ("https://huggingface.co/api/spaces/Station-house/ConspiracyBench/discussions"
       "?type=pull_request&status=open")


def logged_in_user():
    try:
        from huggingface_hub import whoami
        return whoami()["name"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", help="your HF username, so your own PRs are skipped")
    args = ap.parse_args()
    me = args.me or logged_in_user()

    # curl rather than urllib: python.org installs on macOS often lack CA certificates.
    data = json.loads(subprocess.run(["curl", "-sf", API], capture_output=True,
                                     text=True, check=True).stdout)
    prs = data.get("discussions", data)

    rows = []
    for pr in prs:
        author = (pr.get("author") or {}).get("name", "?")
        if me and author.lower() == me.lower():
            continue
        rows.append((pr.get("numComments", 0), pr.get("num"), author, pr.get("title")))

    print(f"{len(rows)} open PR(s) not authored by {me or '(unknown user)'}")
    print("comments=1 usually means only the author's description, no reviews yet\n")
    for comments, num, author, title in sorted(rows, key=lambda r: (r[0], -r[1])):
        print(f"#{num:<5} comments={comments:<3} {author:<20} {title}")


if __name__ == "__main__":
    main()
