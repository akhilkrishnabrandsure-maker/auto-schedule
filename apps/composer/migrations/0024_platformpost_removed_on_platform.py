from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("composer", "0023_platformpost_youtube_video_checked_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="platformpost",
            name="removed_on_platform_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="platformpost",
            name="removal_checked_at",
            field=models.DateTimeField(blank=True, db_index=True, null=True),
        ),
    ]
