#!/usr/bin/env python3
"""Provision the GH200 for this run on Vultr. The API key comes from VULTR_API_KEY.

    python vultr.py stock                       # which regions have the plan now (no key needed)
    python vultr.py os                          # OS images the account can deploy (pick an arm64 Ubuntu)
    python vultr.py sshkeys
    python vultr.py create --os-id N --sshkey-id ID --yes   # billable: $1.99/h until destroyed
    python vultr.py status ID
    python vultr.py destroy ID --yes

Vultr lists the GH200 as preemptible only; the create body asks for that with "preemptible": true.
Check the instance in the console after the first create: neither that field nor the arm64 OS id
could be confirmed from Vultr's public docs on 2026-10-15.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

API = "https://api.vultr.com/v2"
PLAN, REGION = "vbm-72c-480gb-gh200-gpu", "fra"


def call(method: str, path: str, body: dict | None = None, auth: bool = True) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth:
        key = os.environ.get("VULTR_API_KEY") or subprocess.run(  # else the macOS Keychain entry
            ["security", "find-generic-password", "-s", "vultr-api-key", "-w"], capture_output=True, text=True, check=True).stdout.strip()
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(API + path, method=method, headers=headers,
                                 data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:  # e.g. 401 "Unauthorized IP address": add this IP to the key's allow-list
        sys.exit(f"{method} {path}: {e.code} {e.read().decode()}")
    return json.loads(raw) if raw else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("stock", "os", "sshkeys", "addkey", "create", "status", "destroy"))
    ap.add_argument("id", nargs="?")
    ap.add_argument("--os-id", type=int)
    ap.add_argument("--sshkey-id")
    ap.add_argument("--yes", action="store_true", help="confirm a billable or destructive call")
    args = ap.parse_args()

    if args.cmd == "stock":
        regions = [r["id"] for r in call("GET", "/regions?per_page=500", auth=False)["regions"]]
        have = [r for r in regions if PLAN in call("GET", f"/regions/{r}/availability?type=all", auth=False).get("available_plans", [])]
        print(f"{PLAN} in stock in: {', '.join(have) or 'nowhere'}")
    elif args.cmd == "os":
        for o in call("GET", "/os?per_page=500")["os"]:
            print(o["id"], o["arch"], o["name"])
    elif args.cmd == "sshkeys":
        for k in call("GET", "/ssh-keys")["ssh_keys"]:
            print(k["id"], k["name"])
    elif args.cmd == "addkey":  # id = path to a public key file
        key = call("POST", "/ssh-keys", {"name": "cb-decision-judges-v1", "ssh_key": open(args.id).read().strip()})["ssh_key"]
        print(key["id"], key["name"])
    elif args.cmd == "create":
        if not (args.yes and args.os_id and args.sshkey_id):
            sys.exit("create needs --os-id, --sshkey-id and --yes (billable)")
        bm = call("POST", "/bare-metals", {"region": REGION, "plan": PLAN, "os_id": args.os_id, "sshkey_id": [args.sshkey_id],
                                          "label": "cb-decision-judges-v1", "hostname": "cb-judges", "preemptible": True,
                                          "activation_email": False})["bare_metal"]
        print(bm["id"], bm["status"], bm.get("main_ip"))
    elif args.cmd == "status":
        bm = call("GET", f"/bare-metals/{args.id}")["bare_metal"]
        print(bm["id"], bm["status"], bm["power_status"], bm["server_status"], bm["main_ip"], bm["os"], bm["plan"])
    elif args.cmd == "destroy":
        if not args.yes:
            sys.exit("destroy needs --yes")
        call("DELETE", f"/bare-metals/{args.id}")
        print(f"deleted {args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
