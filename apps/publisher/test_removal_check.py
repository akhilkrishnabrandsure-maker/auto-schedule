"""Tests for detecting posts deleted directly on their platform."""

from datetime import timedelta
from unittest import mock

from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.accounts.models import User
from apps.composer.models import PlatformPost, Post
from apps.members.models import OrgMembership, WorkspaceMembership
from apps.organizations.models import Organization
from apps.publisher import removal_check as rc
from apps.social_accounts.models import SocialAccount
from apps.workspaces.models import Workspace


def _resp(status, payload):
    resp = mock.Mock(status_code=status)
    resp.json.return_value = payload
    return resp


class CheckPlatformPostTests(TestCase):
    def test_meta_object_missing_is_removed(self):
        err = {"error": {"code": 100, "error_subcode": 33, "message": "Object does not exist"}}
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(400, err)):
            self.assertEqual(rc.check_platform_post("facebook", "123", "tok"), rc.REMOVED)

    def test_facebook_error_10_with_readable_page_is_removed(self):
        err = {"error": {"code": 10, "message": "Object does not exist, cannot be loaded due to missing permission"}}
        with mock.patch(
            "apps.publisher.removal_check.httpx.get", side_effect=[_resp(400, err), _resp(200, {"data": []})]
        ) as get:
            self.assertEqual(rc.check_platform_post("facebook", "page_post", "tok", account_id="page"), rc.REMOVED)
        self.assertIn("/page/published_posts", get.call_args.args[0])

    def test_facebook_error_10_without_page_access_is_unknown(self):
        err = {"error": {"code": 10, "message": "missing permission"}}
        with mock.patch(
            "apps.publisher.removal_check.httpx.get", side_effect=[_resp(400, err), _resp(403, {"error": {}})]
        ):
            self.assertEqual(rc.check_platform_post("facebook", "page_post", "tok", account_id="page"), rc.UNKNOWN)

    def test_meta_live_post_exists(self):
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(200, {"id": "123"})):
            self.assertEqual(rc.check_platform_post("instagram", "123", "tok"), rc.EXISTS)

    def test_meta_expired_token_is_unknown_not_removed(self):
        err = {"error": {"code": 190, "message": "Error validating access token"}}
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(400, err)):
            self.assertEqual(rc.check_platform_post("facebook", "123", "tok"), rc.UNKNOWN)

    def test_youtube_empty_items_is_removed(self):
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(200, {"items": []})):
            self.assertEqual(rc.check_platform_post("youtube", "vid", "tok"), rc.REMOVED)

    def test_youtube_quota_error_is_unknown(self):
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(403, {})):
            self.assertEqual(rc.check_platform_post("youtube", "vid", "tok"), rc.UNKNOWN)

    def test_linkedin_404_is_removed(self):
        with mock.patch("apps.publisher.removal_check.httpx.get", return_value=_resp(404, {})) as get:
            self.assertEqual(rc.check_platform_post("linkedin_company", "urn:li:share:1", "tok"), rc.REMOVED)
        self.assertIn("urn%3Ali%3Ashare%3A1", get.call_args.args[0])

    def test_network_error_is_unknown(self):
        import httpx

        with mock.patch("apps.publisher.removal_check.httpx.get", side_effect=httpx.ConnectError("down")):
            self.assertEqual(rc.check_platform_post("facebook", "123", "tok"), rc.UNKNOWN)

    def test_unsupported_platform_is_unknown(self):
        self.assertEqual(rc.check_platform_post("bluesky", "at://x", "tok"), rc.UNKNOWN)


class RemovalFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="owner@example.com", password="testpass123", tos_accepted_at=timezone.now()
        )
        self.org = Organization.objects.create(name="Org")
        self.workspace = Workspace.objects.create(organization=self.org, name="WS")
        OrgMembership.objects.create(user=self.user, organization=self.org, org_role=OrgMembership.OrgRole.OWNER)
        WorkspaceMembership.objects.create(
            user=self.user, workspace=self.workspace, workspace_role=WorkspaceMembership.WorkspaceRole.OWNER
        )
        self.client.force_login(self.user)
        self.account = SocialAccount.objects.create(
            workspace=self.workspace,
            platform="facebook",
            account_platform_id="page-1",
            account_name="Client Page",
            connection_status=SocialAccount.ConnectionStatus.CONNECTED,
        )
        self.post = Post.objects.create(workspace=self.workspace, author=self.user, caption="hi")
        self.pp = PlatformPost.objects.create(
            post=self.post,
            social_account=self.account,
            status=PlatformPost.Status.PUBLISHED,
            platform_post_id="fb-1",
            published_at=timezone.now() - timedelta(days=1),
        )
        patcher = mock.patch("apps.publisher.removal_check._token_for", return_value="tok")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_background_pass_marks_removed_and_chip_shows_it(self):
        with mock.patch("apps.publisher.removal_check.check_platform_post", return_value=rc.REMOVED):
            counts = rc.run_removal_check()

        self.assertEqual(counts[rc.REMOVED], 1)
        self.pp.refresh_from_db()
        self.assertIsNotNone(self.pp.removed_on_platform_at)
        self.assertEqual(self.pp.status, PlatformPost.Status.PUBLISHED)
        self.assertEqual(self.pp.chip_status, "removed")
        self.assertEqual(self.pp.chip_status_label, "Removed on platform")

    def test_facebook_is_looked_up_by_page_post_id(self):
        PlatformPost.objects.filter(pk=self.pp.pk).update(platform_extra={"id": "fb-1", "post_id": "page-1_fb-1"})
        with mock.patch("apps.publisher.removal_check.check_platform_post", return_value=rc.EXISTS) as check:
            rc.run_removal_check()
        self.assertEqual(check.call_args.args[1], "page-1_fb-1")
        self.assertEqual(check.call_args.kwargs["account_id"], "page-1")

    def test_recently_checked_posts_are_skipped(self):
        PlatformPost.objects.filter(pk=self.pp.pk).update(removal_checked_at=timezone.now())
        with mock.patch("apps.publisher.removal_check.check_platform_post") as check:
            rc.run_removal_check()
        check.assert_not_called()

    def test_unknown_result_never_marks_removed(self):
        with mock.patch("apps.publisher.removal_check.check_platform_post", return_value=rc.UNKNOWN):
            rc.run_removal_check()
        self.pp.refresh_from_db()
        self.assertIsNone(self.pp.removed_on_platform_at)
        self.assertIsNotNone(self.pp.removal_checked_at)

    def test_post_found_again_clears_the_mark(self):
        PlatformPost.objects.filter(pk=self.pp.pk).update(removed_on_platform_at=timezone.now())
        self.pp.refresh_from_db()
        with mock.patch("apps.publisher.removal_check.check_platform_post", return_value=rc.EXISTS):
            rc.check_posts([self.pp])
        self.pp.refresh_from_db()
        self.assertIsNone(self.pp.removed_on_platform_at)

    def test_check_now_button_and_banner(self):
        url = reverse("composer:compose_edit", kwargs={"workspace_id": self.workspace.id, "post_id": self.post.id})
        self.assertContains(self.client.get(url), "Check if still live on platforms")

        check_url = reverse("composer:check_removed", kwargs={"workspace_id": self.workspace.id, "post_id": self.post.id})
        with mock.patch("apps.publisher.removal_check.check_platform_post", return_value=rc.REMOVED):
            resp = self.client.post(check_url)
        self.assertEqual(resp.status_code, 204)
        self.assertEqual(resp["HX-Refresh"], "true")

        page = self.client.get(url)
        self.assertContains(page, "Removed on platform")
        self.assertContains(page, "this post was deleted on")
        # The time is rendered in the viewer's own timezone by the browser.
        self.assertContains(page, '<time data-local-time datetime="')
        self.assertContains(page, "+00:00")
