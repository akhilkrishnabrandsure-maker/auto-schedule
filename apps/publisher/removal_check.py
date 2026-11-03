"""Detect published posts that were deleted directly on the platform.

A post deleted on Facebook, Instagram, YouTube or LinkedIn keeps showing as
"Published" in Studio, because nothing ever looks back after publishing. This
module asks each platform whether the post still exists and stamps
``PlatformPost.removed_on_platform_at`` when it is gone for certain.

Only a definite "does not exist" answer marks a post removed. Permission
errors, expired tokens, rate limits and network failures leave the post as it
is, so a flaky API never produces a false "Removed" label.
"""

import logging
from datetime import timedelta

import httpx
from django.db.models import F, Q
from django.utils import timezone

logger = logging.getLogger(__name__)

EXISTS = "exists"
REMOVED = "removed"
UNKNOWN = "unknown"

SUPPORTED_PLATFORMS = (
    "facebook",
    "instagram",
    "instagram_login",
    "threads",
    "youtube",
    "linkedin_personal",
    "linkedin_company",
)

CHECK_INTERVAL_SECONDS = 6 * 3600
# Only recent posts are re-checked automatically; older ones can still be
# checked from the post page.
MAX_POST_AGE = timedelta(days=90)
BATCH_SIZE = 300
TIMEOUT = 20.0

_META_HOSTS = {
    "facebook": "https://graph.facebook.com/v25.0",
    "instagram": "https://graph.facebook.com/v25.0",
    "instagram_login": "https://graph.instagram.com/v25.0",
    "threads": "https://graph.threads.net/v1.0",
}


def check_platform_post(platform: str, post_id: str, access_token: str, account_id: str = "") -> str:
    """Return EXISTS, REMOVED or UNKNOWN for one published post.

    ``account_id`` is the Page ID for Facebook, used to tell "deleted" apart
    from "no permission" when Facebook answers with its ambiguous error #10.
    """
    if not post_id or not access_token or platform not in SUPPORTED_PLATFORMS:
        return UNKNOWN
    try:
        if platform == "facebook":
            return _check_facebook(post_id, access_token, account_id)
        if platform in _META_HOSTS:
            return _check_meta(_META_HOSTS[platform], post_id, access_token)
        if platform == "youtube":
            return _check_youtube(post_id, access_token)
        return _check_linkedin(post_id, access_token)
    except httpx.HTTPError as exc:
        logger.info("removal check: %s %s unreachable: %s", platform, post_id, exc)
        return UNKNOWN


def _check_meta(base: str, post_id: str, token: str) -> str:
    resp = httpx.get(f"{base}/{post_id}", params={"fields": "id", "access_token": token}, timeout=TIMEOUT)
    if resp.status_code == 200:
        return EXISTS
    try:
        error = resp.json().get("error", {})
    except ValueError:
        return UNKNOWN
    # Code 100 / subcode 33: "Object with ID ... does not exist, cannot be
    # loaded due to missing permissions, or does not support this operation".
    # With the token that published the post, that means it was deleted.
    if error.get("code") == 100 and error.get("error_subcode") == 33:
        return REMOVED
    logger.info("removal check: Meta error for %s: %s", post_id, error.get("message", resp.status_code))
    return UNKNOWN


def _check_facebook(post_id: str, token: str, page_id: str) -> str:
    base = _META_HOSTS["facebook"]
    resp = httpx.get(f"{base}/{post_id}", params={"fields": "id", "access_token": token}, timeout=TIMEOUT)
    if resp.status_code == 200:
        return EXISTS
    try:
        error = resp.json().get("error", {})
    except ValueError:
        return UNKNOWN
    if error.get("code") == 100 and error.get("error_subcode") == 33:
        return REMOVED
    # Error #10 reads "Object does not exist, cannot be loaded due to missing
    # permission ...". It means deleted only if this token can still read the
    # Page's own posts, so confirm with a call that needs the same permission.
    if error.get("code") == 10 and page_id:
        control = httpx.get(
            f"{base}/{page_id}/published_posts",
            params={"fields": "id", "limit": 1, "access_token": token},
            timeout=TIMEOUT,
        )
        if control.status_code == 200:
            return REMOVED
    logger.info("removal check: Facebook error for %s: %s", post_id, error.get("message", resp.status_code))
    return UNKNOWN


def _check_youtube(video_id: str, token: str) -> str:
    resp = httpx.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"part": "id", "id": video_id},
        headers={"Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    if resp.status_code != 200:
        return UNKNOWN
    return EXISTS if resp.json().get("items") else REMOVED


def _check_linkedin(post_urn: str, token: str) -> str:
    from providers.linkedin import LINKEDIN_HEADERS, _encode_urn

    resp = httpx.get(
        f"https://api.linkedin.com/rest/posts/{_encode_urn(post_urn)}",
        headers={"Authorization": f"Bearer {token}", **LINKEDIN_HEADERS},
        timeout=TIMEOUT,
    )
    if resp.status_code == 200:
        return EXISTS
    if resp.status_code in (404, 410):
        return REMOVED
    return UNKNOWN


def _lookup_id(pp) -> str:
    """The ID to look the post up by.

    Facebook photo and video posts store the media ID as ``platform_post_id``,
    which the Graph API no longer serves on its own ("singular statuses API is
    deprecated"); the Page post ID from the publish response is what works.
    """
    if pp.social_account.platform == "facebook":
        return (pp.platform_extra or {}).get("post_id") or pp.platform_post_id
    return pp.platform_post_id


def _token_for(account) -> str:
    from apps.analytics.tasks import _analytics_provider_and_token

    _provider, token = _analytics_provider_and_token(account)
    return token


def check_posts(platform_posts) -> dict:
    """Check the given PlatformPosts, update them, and return counts by result."""
    counts = {EXISTS: 0, REMOVED: 0, UNKNOWN: 0}
    tokens = {}
    now = timezone.now()
    for pp in platform_posts:
        account = pp.social_account
        if account.platform not in SUPPORTED_PLATFORMS or not pp.platform_post_id:
            continue
        if account.pk not in tokens:
            try:
                tokens[account.pk] = _token_for(account)
            except Exception:
                logger.warning("removal check: no usable token for %s", account, exc_info=True)
                tokens[account.pk] = ""
        result = check_platform_post(
            account.platform, _lookup_id(pp), tokens[account.pk], account_id=account.account_platform_id
        )
        counts[result] += 1

        updates = {"removal_checked_at": now}
        if result == REMOVED and not pp.removed_on_platform_at:
            updates["removed_on_platform_at"] = now
        elif result == EXISTS and pp.removed_on_platform_at:
            updates["removed_on_platform_at"] = None
        type(pp).objects.filter(pk=pp.pk).update(**updates)
        for field, value in updates.items():
            setattr(pp, field, value)
    return counts


def posts_due_for_check(now=None):
    from apps.composer.models import PlatformPost

    now = now or timezone.now()
    # Skip posts already checked within the last interval.
    recently_checked = Q(removal_checked_at__gt=now - timedelta(seconds=CHECK_INTERVAL_SECONDS))
    return (
        PlatformPost.objects.filter(
            status=PlatformPost.Status.PUBLISHED,
            removed_on_platform_at__isnull=True,
            published_at__gte=now - MAX_POST_AGE,
            social_account__platform__in=SUPPORTED_PLATFORMS,
        )
        .exclude(platform_post_id="")
        .exclude(recently_checked)
        .select_related("social_account")
        .order_by(F("removal_checked_at").asc(nulls_first=True))
    )


def run_removal_check() -> dict:
    """One background pass over recently published posts, oldest check first."""
    counts = check_posts(list(posts_due_for_check()[:BATCH_SIZE]))
    if counts[REMOVED]:
        logger.info("removal check: %s post(s) found removed on their platform", counts[REMOVED])
    return counts
