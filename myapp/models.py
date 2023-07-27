from django.db import models
from django.contrib.auth.models import User


class Destination(models.Model):
    WISATA_REGLIGI = 'WR'
    TAMAN_WISATA = 'TW'
    WISATA_AIR = 'WA'
    WISATA_KULINER = 'WK'

    CATEGORY_CHOICES = [
        (WISATA_REGLIGI, 'Wisata Religi'),
        (TAMAN_WISATA, 'Taman Wisata'),
        (WISATA_AIR, 'Wisata Air'),
        (WISATA_KULINER, 'Wisata Kuliner'),
    ]

    BUDGET_0_20000 = '0-20000'
    BUDGET_20000_50000 = '20000-50000'
    BUDGET_50000_100000 = '50000-100000'
    BUDGET_100000_200000 = '100000-200000'

    BUDGET_CHOICES = [
        (BUDGET_0_20000, '0-20000'),
        (BUDGET_20000_50000, '20000-50000'),
        (BUDGET_50000_100000, '50000-100000'),
        (BUDGET_100000_200000, '100000-200000'),
    ]

    title = models.CharField(max_length=200)
    image = models.ImageField(upload_to='destination_pics', null=True, blank=True)
    description = models.TextField()
    category = models.CharField(max_length=2, choices=CATEGORY_CHOICES,default=WISATA_REGLIGI,)
    budget = models.CharField(max_length=13, choices=BUDGET_CHOICES,default=BUDGET_0_20000,)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    youtube_url = models.URLField(max_length=200, null=True, blank=True)

    def __str__(self):
        return self.title


class Review(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    destination = models.ForeignKey(Destination, on_delete=models.CASCADE)
    review_text = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Review by {self.user.username} on {self.destination.title}'
