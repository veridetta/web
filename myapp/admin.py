from django.contrib import admin
from .models import Destination, Review

class DestinationAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'budget')
    list_filter = ('category', 'budget')
    search_fields = ('title',)

admin.site.register(Destination, DestinationAdmin)
admin.site.register(Review)
