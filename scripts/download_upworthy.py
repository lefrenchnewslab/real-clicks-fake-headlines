# Block E Stage 0 (login node). Walk the OSF folder tree of jd64p and download package-level CSVs for the three splits.
import json, os, re, time, urllib.request
ROOT = "https://api.osf.io/v2/nodes/jd64p/files/osfstorage/"
def get(u):
    for a in range(5):
        try: return json.load(urllib.request.urlopen(u, timeout=60))
        except Exception as e: print("retry", u, e); time.sleep(10)
    raise SystemExit("OSF unreachable: " + u)
listing = []
def walk(u, path=""):
    while u:
        d = get(u)
        for x in d["data"]:
            a = x["attributes"]; name = path + "/" + a["name"]
            if a["kind"] == "folder": walk(x["relationships"]["files"]["links"]["related"]["href"], name)
            else: listing.append({"path": name, "size": a.get("size"), "download": x["links"]["download"], "modified": a.get("date_modified")})
        u = d["links"].get("next")
walk(ROOT)
os.makedirs("data/upworthy", exist_ok=True)
json.dump(listing, open("data/upworthy/osf_listing.json", "w"), indent=1)
for f in listing: print(f["size"], f["path"])
want = [f for f in listing if f["path"].startswith("/upworthy-archive-datasets") and re.search(r"(exploratory|confirmatory|holdout|hold-out).*packages.*\.csv$", f["path"], re.I)]
print("WANT", [w["path"] for w in want])
for f in want:
    dst = "data/upworthy/" + os.path.basename(f["path"])
    if os.path.exists(dst) and os.path.getsize(dst) == f["size"]: continue
    urllib.request.urlretrieve(f["download"], dst); print("downloaded", dst, os.path.getsize(dst))
