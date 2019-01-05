from django.contrib import admin
from .models import ImageUploadModel

# Register your models here.

class upload_image_Admin(admin.ModelAdmin):
    list_display = ('description', 'document')

admin.site.register(ImageUploadModel, upload_image_Admin)