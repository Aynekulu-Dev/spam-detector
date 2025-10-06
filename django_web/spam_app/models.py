from django.db import models

class Prediction(models.Model):
    text = models.TextField()
    prediction = models.CharField(max_length=10)  # 'spam' or 'ham'
    confidence = models.FloatField()
    is_spam = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.text[:50]} - {self.prediction} ({(self.confidence * 100):.1f}%)"
    
    class Meta:
        ordering = ['-created_at']