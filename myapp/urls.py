from django.urls import path
from . import views

urlpatterns = [
    path('portfolio', views.portfolio, name='portfolio'),
    path('search/', views.search_stock, name='search_stock'),
    path('buy/', views.buy_stock, name='buy_stock'),
    path('sell/', views.sell_stock, name='sell_stock'),
    path('', views.signup, name='signup'),  # Signup page
    path('login/', views.login_view, name='login'),  # Login page
    path('logout/', views.logout_view, name='logout'),  # Logout functionality
    path('get_stock_price/<str:ticker>/', views.get_stock_price, name='get_stock_price'),
    path('watchlist/', views.watchlist_view, name='watchlist'),
    path('stock/<str:ticker>/', views.stock_detail, name='stock_detail'),
    path('get_stock_history/<str:ticker>/<str:timeframe>/', views.get_stock_history, name='get_stock_history'),
    path('get_stock_news/<str:ticker>/', views.get_stock_news, name='get_stock_news'),
    path('add_balance/', views.add_balance, name='add_balance'),
    path('remove_funds/', views.remove_funds, name='remove_funds'),
]