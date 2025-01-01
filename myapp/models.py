from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    balance = models.FloatField(default=0)

    def __str__(self):
        return self.user.username

class Stock(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10)
    quantity = models.IntegerField()
    average_price = models.FloatField()

    def __str__(self):
        return f"{self.ticker} - {self.quantity} shares"

class Transaction(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10)
    quantity = models.IntegerField()
    price_per_stock = models.FloatField()
    transaction_type = models.CharField(max_length=10, choices=[('BUY', 'Buy'), ('SELL', 'Sell')])
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transaction_type} - {self.ticker} - {self.quantity} shares"

class Watchlist(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    ticker = models.CharField(max_length=10)
    added_date = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True, null=True)  # Optional notes for each watched stock

    class Meta:
        unique_together = ('user', 'ticker')  # Prevents duplicate entries
        ordering = ['-added_date']  # Most recent first

    def __str__(self):
        return f"{self.user.user.username} - {self.ticker}"


