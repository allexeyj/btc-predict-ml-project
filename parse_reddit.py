import praw
from prawcore import NotFound
import datetime
import json
import pandas as pd
import time
from typing import Literal
from secrets import client_id, client_secret

reddit: praw.Reddit = praw.Reddit(client_id=client_id,
                                  client_secret=client_secret,
                                  user_agent='Mozilla/5.0 (platform; rv:gecko-version) Gecko/gecko-trail Firefox/firefox-version',
                                  check_for_async=False)


def open_subreddit(name: str) -> praw.models.Subreddit | None:
    global reddit
    try:
        reddit.subreddits.search_by_name(name, exact=True)
    except NotFound:
        return
    return reddit.subreddit(name)


def get_date(submission: praw.models.Subreddit) -> datetime:
    time = submission.created
    return datetime.datetime.fromtimestamp(time)


def get_comments(submission: praw.models.Subreddit) -> list[str]:
    submission.comments.replace_more(limit=None)
    comments: list[str] = []
    for i in range(min(10, len(submission.comments.list()))):
        comments.append(submission.comments.list()[i].body)
    return comments


def get_post_properties(submission: praw.models.Subreddit) -> dict:
    if submission.stickied or not submission.is_self:
        return {}
    post_properties: dict = {'title': submission.title, 'text': submission.selftext,
                             'upvotes': submission.score, 'upvote_ratio': submission.upvote_ratio,
                             'comments': get_comments(submission), 'date': get_date(submission)}
    return post_properties


def add_post(posts: pd.DataFrame, submission: praw.models.Subreddit) -> pd.DataFrame:
    post_properties: dict = get_post_properties(submission)
    if not post_properties:
        return posts
    if posts.empty:
        return pd.DataFrame([post_properties])
    return pd.concat([posts, pd.DataFrame([post_properties])], ignore_index=True)


def get_limit(time_filter: Literal["day", "week", "month", "year"]) -> int:
    limit: int = 10
    if time_filter == "week":
        limit *= 7
    if time_filter == "month":
        limit *= 30
    if time_filter == "year":
        limit *= 365
    return limit


def get_posts_dataframe(subreddit_name: str, time_filter: Literal["day", "week", "month", "year"],
                        limit: int | None = None) -> pd.DataFrame:
    subreddit: praw.models.Subreddit | None = open_subreddit(subreddit_name)
    if not subreddit:
        return pd.DataFrame()
    filtered_posts: praw.models.ListingGenerator = subreddit.top(
        limit=limit, time_filter=time_filter)
    to_return: pd.DataFrame = pd.DataFrame()
    if not limit:
        limit: int = get_limit(time_filter)
    for post in filtered_posts:
        to_return = add_post(to_return, post)
        if to_return.shape[0] == limit:
            break
    return to_return


def posts_to_json(posts: pd.DataFrame, json_name: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) -> None:
    posts['date'] = posts['date'].dt.strftime('%Y-%m-%d-%H-%M-%S')
    posts.to_json('{}.json'.format(json_name), orient='records', lines=True)


bitcoin = get_posts_dataframe("Bitcoin", "month")
posts_to_json(bitcoin)

crypto_currency = get_posts_dataframe("CryptoCurrency", "week")
posts_to_json(crypto_currency)
