import logging
import sqlite3
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class WikiCache:
    def __init__(self, db_location: str = "wikicache.db") -> None:
        self.con = sqlite3.connect(db_location)
        self.cur = self.con.cursor()
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS articles ("
            "title text PRIMARY KEY,"
            "extract text,"
            "created text)"
        )

    def __del__(self) -> None:
        self.con.close()

    def get_article_cache(self, title: str) -> Optional[str]:
        self.cur.execute(
            "SELECT extract FROM articles "
            "WHERE title = ? AND created > datetime('now', 'utc', '-1 day')",
            [title],
        )
        row = self.cur.fetchone()
        if row:
            return row[0]
        return None

    def add_article_cache(self, title: str, extract: str) -> None:
        self.cur.execute("DELETE FROM articles WHERE title = ?", [title])
        self.cur.execute(
            "INSERT INTO articles (title, extract, created) VALUES (?, ?, datetime('now', 'utc'))",
            [title, extract],
        )
        self.con.commit()


def get_wiki_article_text(
    title: str, endpoint: str = "https://en.wikipedia.org/w/api.php"
) -> str:
    wiki_cache = WikiCache()

    text = wiki_cache.get_article_cache(title)
    if text:
        return text

    logger.info("Querying wiki for article text '%s'", title)

    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts",
        "exsectionformat": "wiki",
        "redirects": "1",
        "titles": title,
    }
    resp = requests.get(endpoint, params)

    # TODO: this may require several requests for large articles
    # params["excontinue"] =  content['continue']

    if resp.status_code == 200:
        data = resp.json()["query"]["pages"]
        page = data[0]
        title = page["title"]
        text = page["extract"]
        wiki_cache.add_article_cache(title, text)
        logger.info("Cached article text as '%s'", title)
        return text
    else:
        logger.error(
            "Failed to obtain article text from wiki (status: %s, text: %s)",
            resp.status_code,
            resp.text,
        )
        raise Exception(
            f"Failed to obtain article text from wiki (status: {resp.status_code})"
        )
