import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = "wsb.sqlite"


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY,
            platform TEXT NOT NULL,
            platform_post_id TEXT NOT NULL,
            title TEXT,
            body TEXT,
            author TEXT,
            subreddit_channel TEXT,
            created_utc TIMESTAMP NOT NULL,
            score_likes INTEGER,
            upvote_ratio REAL,
            num_comments INTEGER,
            url TEXT,
            collected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            media TEXT,
            UNIQUE(platform, platform_post_id)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_posts_platform_post_id
        ON posts(platform, platform_post_id)
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            ts TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            source TEXT,
            UNIQUE(ticker, ts)
        )
        """
    )
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_prices_ticker_ts
        ON prices(ticker, ts)
        """
    )
    conn.commit()
    if close_after:
        conn.close()

def insert_post(
    platform: str,
    platform_post_id: str,
    title: Optional[str] = None,
    body: Optional[str] = None,
    author: Optional[str] = None,
    subreddit_channel: Optional[str] = None,
    created_utc: datetime = None,
    score_likes: Optional[int] = None,
    upvote_ratio: Optional[float] = None,
    num_comments: Optional[int] = None,
    url: Optional[str] = None,
    collected_at: Optional[datetime] = None,
    media: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO posts (
            platform, platform_post_id, title, body, author, subreddit_channel,
            created_utc, score_likes, upvote_ratio, num_comments, url, collected_at, media
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP), ?)
        """,
        (
            platform,
            platform_post_id,
            title,
            body,
            author,
            subreddit_channel,
            (
                created_utc.isoformat()
                if isinstance(created_utc, datetime)
                else created_utc
            ),
            score_likes,
            upvote_ratio,
            num_comments,
            url,
            (
                collected_at.isoformat()
                if isinstance(collected_at, datetime)
                else collected_at
            ),
            media,
        ),
    )
    conn.commit()
    rowid = cur.lastrowid
    if close_after:
        conn.close()
    return rowid


def upsert_post_by_platform_id(
    platform: str,
    platform_post_id: str,
    values: Dict[str, Any],
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    columns = [
        "title",
        "body",
        "author",
        "subreddit_channel",
        "created_utc",
        "score_likes",
        "upvote_ratio",
        "num_comments",
        "url",
        "collected_at",
        "media",
    ]
    payload = {k: values.get(k) for k in columns}
    if isinstance(payload.get("created_utc"), datetime):
        payload["created_utc"] = payload["created_utc"].isoformat()
    if isinstance(payload.get("collected_at"), datetime):
        payload["collected_at"] = payload["collected_at"].isoformat()
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO posts (platform, platform_post_id, title, body, author, subreddit_channel,
            created_utc, score_likes, upvote_ratio, num_comments, url, collected_at, media)
        VALUES (:platform, :platform_post_id, :title, :body, :author, :subreddit_channel,
            :created_utc, :score_likes, :upvote_ratio, :num_comments, :url, COALESCE(:collected_at, CURRENT_TIMESTAMP), :media)
        ON CONFLICT(platform, platform_post_id) DO UPDATE SET
            title=excluded.title,
            body=excluded.body,
            author=excluded.author,
            subreddit_channel=excluded.subreddit_channel,
            created_utc=excluded.created_utc,
            score_likes=excluded.score_likes,
            upvote_ratio=excluded.upvote_ratio,
            num_comments=excluded.num_comments,
            url=excluded.url,
            collected_at=COALESCE(excluded.collected_at, posts.collected_at),
            media=excluded.media
        """,
        {
            "platform": platform,
            "platform_post_id": platform_post_id,
            **payload,
        },
    )
    conn.commit()
    if close_after:
        conn.close()


def get_post_by_platform_id(
    platform: str, platform_post_id: str, conn: Optional[sqlite3.Connection] = None
) -> Optional[sqlite3.Row]:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM posts WHERE platform = ? AND platform_post_id = ?",
        (platform, platform_post_id),
    )
    row = cur.fetchone()
    if close_after:
        conn.close()
    return row


def list_posts(
    platform: Optional[str] = None,
    subreddit_channel: Optional[str] = None,
    ticker: Optional[str] = None,
    date_range: Optional[Tuple[str, str]] = None,
    limit: int = 100,
    offset: int = 0,
    conn: Optional[sqlite3.Connection] = None,
) -> List[sqlite3.Row]:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    where = []
    params: List[Any] = []
    if platform:
        where.append("platform = ?")
        params.append(platform)
    if subreddit_channel:
        where.append("subreddit_channel = ?")
        params.append(subreddit_channel)
    if ticker:
        where.append("title LIKE ? OR body LIKE ?")
        params.extend([f"%{ticker}%", f"%{ticker}%"])
    if date_range:
        where.append("created_utc BETWEEN ? AND ?")
        params.extend(list(date_range))
    sql = "SELECT * FROM posts"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY created_utc DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    if close_after:
        conn.close()
    return rows


def upsert_price(
    ticker: str,
    ts: datetime,
    open_price: Optional[float] = None,
    high: Optional[float] = None,
    low: Optional[float] = None,
    close: Optional[float] = None,
    volume: Optional[float] = None,
    source: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO prices (ticker, ts, open, high, low, close, volume, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, ts) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            source=excluded.source
        """,
        (
            ticker,
            ts.isoformat() if isinstance(ts, datetime) else ts,
            open_price,
            high,
            low,
            close,
            volume,
            source,
        ),
    )
    conn.commit()
    if close_after:
        conn.close()


def get_prices(
    ticker: str,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    limit: int = 1000,
    conn: Optional[sqlite3.Connection] = None,
) -> List[sqlite3.Row]:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    where = ["ticker = ?"]
    params: List[Any] = [ticker]
    if start_ts:
        where.append("ts >= ?")
        params.append(start_ts)
    if end_ts:
        where.append("ts <= ?")
        params.append(end_ts)
    sql = (
        "SELECT * FROM prices WHERE " + " AND ".join(where) + " ORDER BY ts ASC LIMIT ?"
    )
    params.append(limit)
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    if close_after:
        conn.close()
    return rows


def find_nearest_price(
    ticker: str,
    post_ts: str,
    window_minutes: int = 60,
    conn: Optional[sqlite3.Connection] = None,
) -> Optional[sqlite3.Row]:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM prices
        WHERE ticker = ? AND ts BETWEEN datetime(?, '-' || ? || ' minutes') AND datetime(?, '+' || ? || ' minutes')
        ORDER BY ABS(strftime('%s', ts) - strftime('%s', ?)) ASC
        LIMIT 1
        """,
        (ticker, post_ts, window_minutes, post_ts, window_minutes, post_ts),
    )
    row = cur.fetchone()
    if close_after:
        conn.close()
    return row


def migrate_reddit_raw_to_posts(
    default_ticker: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    close_after = False
    if conn is None:
        conn = get_connection()
        close_after = True
    cur = conn.cursor()
    cur.execute(
        """
        SELECT post_id, subreddit, ticker, title, url, author, score, num_comments, post_time
        FROM reddit_raw_posts
        ORDER BY post_time ASC
        """
    )
    rows = cur.fetchall()
    inserted = 0
    for r in rows:
        platform_post_id = r["post_id"]
        subreddit = r["subreddit"]
        title = r["title"]
        url = r["url"]
        author = r["author"]
        score = r["score"]
        num_comments = r["num_comments"]
        created_utc = r["post_time"]
        upsert_post_by_platform_id(
            platform="reddit",
            platform_post_id=platform_post_id,
            values={
                "title": title,
                "body": None,
                "author": author,
                "subreddit_channel": subreddit,
                "created_utc": created_utc,
                "score_likes": score,
                "upvote_ratio": None,
                "num_comments": num_comments,
                "url": url,
                "collected_at": None,
                "media": None,
            },
            conn=conn,
        )
        inserted += 1
    conn.commit()
    if close_after:
        conn.close()
    return inserted



if __name__ == "__main__":
    c = get_connection()
    init_db(c)
    migrate_reddit_raw_to_posts(c)
    c.close()
