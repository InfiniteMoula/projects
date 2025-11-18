import requests
import math

# --------- À PERSONNALISER ---------
OWNER = "InfiniteMoula"
REPO = "projects"
GITHUB_TOKEN = "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"  # colle ton token ici entre guillemets
PER_PAGE = 100  # max autorisé par GitHub
# -----------------------------------

API_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/commits"


def fetch_all_commits():
    session = requests.Session()
    session.headers.update({
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "commit-fetcher-script"
    })

    page = 1
    all_commits = []

    while True:
        params = {
            "per_page": PER_PAGE,
            "page": page,
        }
        print(f"Fetching page {page}...")
        resp = session.get(API_URL, params=params)

        if resp.status_code != 200:
            print(f"Erreur API GitHub: {resp.status_code} - {resp.text}")
            break

        commits_page = resp.json()
        if not commits_page:
            # plus de commits
            break

        all_commits.extend(commits_page)
        page += 1

    print(f"Total commits récupérés: {len(all_commits)}")
    return all_commits


def save_commits_to_csv(commits, filename="commits.csv"):
    import csv

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha", "author_name", "author_email", "date", "message"])

        for c in commits:
            sha = c.get("sha", "")
            commit_info = c.get("commit", {})
            author = commit_info.get("author", {}) or {}
            author_name = author.get("name", "")
            author_email = author.get("email", "")
            date = author.get("date", "")
            message = (commit_info.get("message") or "").replace("\n", " ").strip()

            writer.writerow([sha, author_name, author_email, date, message])

    print(f"Commits sauvegardés dans {filename}")


if __name__ == "__main__":
    commits = fetch_all_commits()
    if commits:
        save_commits_to_csv(commits)
