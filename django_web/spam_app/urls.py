from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('api/check-spam/', views.check_spam, name='check_spam'),
    path('api/history/', views.prediction_history, name='prediction_history'),
    path('api/status/', views.service_status, name='service_status'),
]
