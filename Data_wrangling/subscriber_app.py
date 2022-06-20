from typing import NamedTuple, Deque, DefaultDict, Set, List, Optional
from collections import deque, defaultdict
from itertools import islice
from heapq import merge
from sys import intern  # so that strings are stored just once in memory
import time


User = str
Post = NamedTuple('Post', [('timestamp', float), ('user', str), ('text', str)])
posts: Deque[Post] = deque()  # Posts from newest to oldest

user_posts: DefaultDict[User, deque] = defaultdict(deque)
following: DefaultDict[User, Set[User]] = defaultdict(set)
followers: DefaultDict[User, Set[User]] = defaultdict(set)


def post_message(user: User, text: str, timestamp: float = None) -> None:
    user = intern(user)
    timestamp = timestamp or time.time()
    post = Post(timestamp, user, text)
    posts.appendleft(post)
    user_posts[user].appendleft(post)


def follow(user: User, followed_user: User) -> None:
    user, followed_user = intern(user), intern(followed_user)
    following[user].add(followed_user)
    followers[followed_user].add(user)


def post_by_user(user: User, limit: Optional[int] = None) -> List[Post]:
    return list(islice(user_posts[user], limit))


def post_for_user(user: User, limit: Optional[int] = None) -> List[Post]:
    relevant = merge(*[user_posts[followed]
                     for followed in following[user]], reverse=True)
    return list(islice(relevant, limit))


def search(phrase: str, limit: Optional[str] = None) -> List[Post]:
    return list(islice((post for post in posts if phrase in post.text), limit))
