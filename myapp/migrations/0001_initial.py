# Generated by Django 4.1.2 on 2023-05-31 00:25

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Destination',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=200)),
                ('image', models.ImageField(blank=True, null=True, upload_to='destination_pics')),
                ('description', models.TextField()),
                ('category', models.CharField(choices=[('WR', 'Wisata Religi'), ('TW', 'Taman Wisata'), ('WA', 'Wisata Air'), ('WK', 'Wisata Kuliner')], default='WR', max_length=2)),
                ('budget', models.CharField(choices=[('0-20000', '0-20000'), ('20000-50000', '20000-50000'), ('50000-100000', '50000-100000'), ('100000-200000', '100000-200000')], default='0-20000', max_length=13)),
                ('latitude', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
                ('longitude', models.DecimalField(blank=True, decimal_places=6, max_digits=9, null=True)),
            ],
        ),
    ]