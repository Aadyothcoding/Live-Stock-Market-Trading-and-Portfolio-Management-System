from django.shortcuts import render, redirect
from .models import UserProfile, Stock, Transaction
import yfinance as yf
from .forms import UserForm
from django.contrib.auth.decorators import login_required
from decimal import Decimal
from django.contrib.auth import login
from django.http import JsonResponse
import pandas as pd
from django.contrib import messages
import requests
from nsetools import Nse
from concurrent.futures import ThreadPoolExecutor

@login_required
def portfolio(request):
    try:
        user_profile, created = UserProfile.objects.get_or_create(
            user=request.user,
            defaults={'balance': 10000.00}
        )
        
        stocks = Stock.objects.filter(user=user_profile)
        stock_data = []
        
        for stock in stocks:
            try:
                ticker_data = yf.Ticker(stock.ticker)
                current_price = ticker_data.info.get('regularMarketPrice', stock.average_price)
                
                total_value = stock.quantity * current_price
                profit_loss = ((current_price - stock.average_price) / stock.average_price) * 100 if stock.average_price > 0 else 0
                
                stock_data.append({
                    'ticker': stock.ticker,
                    'quantity': stock.quantity,
                    'average_price': stock.average_price,
                    'current_price': current_price,
                    'total_value': total_value,
                    'profit_loss': profit_loss
                })
            except Exception as e:
                print(f"Error fetching data for {stock.ticker}: {e}")
                stock_data.append({
                    'ticker': stock.ticker,
                    'quantity': stock.quantity,
                    'average_price': stock.average_price,
                    'current_price': stock.average_price,
                    'total_value': stock.quantity * stock.average_price,
                    'profit_loss': 0
                })
        
        # List of major Indian stocks to track
        stock_symbols = [
            'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
            'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS', 'KOTAKBANK.NS'
        ]

        live_stocks = []
        
        def fetch_stock_data(symbol):
            try:
                stock = yf.Ticker(symbol)
                info = stock.info
                current_price = info.get('regularMarketPrice', 0)
                previous_close = info.get('regularMarketPreviousClose', 0)
                
                if previous_close > 0:
                    change = ((current_price - previous_close) / previous_close) * 100
                else:
                    change = 0

                print(f"Fetched data for {symbol}: Price={current_price}, Change={change}%")  # Debug print

                return {
                    'symbol': symbol.replace('.NS', ''),
                    'price': current_price,
                    'change': abs(change) if change < 0 else change,
                    'type': 'gainer' if change >= 0 else 'loser'
                }
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return None

        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(fetch_stock_data, stock_symbols)
            
        # Filter out None results and sort by change percentage
        live_stocks = [stock for stock in results if stock is not None]
        live_stocks.sort(key=lambda x: x['change'], reverse=True)

        print(f"Total stocks fetched: {len(live_stocks)}")  # Debug print
        print("Live stocks data:", live_stocks)  # Debug print

        context = {
            'stocks': stock_data,
            'balance': user_profile.balance,
            'live_stocks': live_stocks,
        }
        return render(request, 'portfolio.html', context)
    except Exception as e:
        print(f"Portfolio view error: {e}")  # Debug print
        messages.error(request, f'Error: {str(e)}')
        return redirect('login')


@login_required
def search_stock(request):
    if request.method == 'GET' and 'ticker' in request.GET:
        ticker = request.GET['ticker'].upper()
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Print debug information
            print(f"Fetching data for ticker: {ticker}")
            print(f"Raw info: {info}")
            
            # Get current price with fallback options
            current_price = (
                info.get('regularMarketPrice') or 
                info.get('currentPrice') or 
                info.get('price', 0)
            )
            
            stock_data = {
                'ticker': ticker,
                'name': info.get('shortName', info.get('longName', 'N/A')),
                'current_price': current_price,
                'currency': info.get('currency', 'USD'),
                'market_cap': info.get('marketCap', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'pe_ratio': info.get('trailingPE', info.get('forwardPE', 'N/A')),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 'N/A'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 'N/A'),
            }
            
            print(f"Processed stock data: {stock_data}")  # Debug print
            
            if current_price == 0:
                raise ValueError(f"Could not fetch price for {ticker}")
                
            return render(request, 'buy_stock.html', {'stock_data': stock_data})
            
        except Exception as e:
            print(f"Error fetching stock data: {str(e)}")
            import traceback
            traceback.print_exc()
            return render(request, 'portfolio.html', {
                'error': f"Could not find stock with ticker symbol {ticker}. Error: {str(e)}"
            })
    
    elif request.method == 'POST':
        try:
            # Get data from POST request
            ticker = request.POST.get('ticker')  # Changed from GET to POST
            quantity = int(request.POST.get('quantity'))
            current_price = float(request.POST.get('current_price'))
            
            # Print debug information
            print(f"Received purchase request:")
            print(f"Ticker: {ticker}")
            print(f"Quantity: {quantity}")
            print(f"Price: {current_price}")
            
            user_profile = UserProfile.objects.get(user=request.user)
            total_cost = quantity * current_price

            print(f"User balance: {user_profile.balance}")
            print(f"Total cost: {total_cost}")

            if user_profile.balance >= total_cost:
                # Create or update stock holding
                stock, created = Stock.objects.get_or_create(
                    user=user_profile,
                    ticker=ticker,
                    defaults={
                        'quantity': quantity,
                        'average_price': current_price
                    }
                )

                if not created:
                    # Update existing stock holding
                    total_value = (stock.quantity * stock.average_price) + (quantity * current_price)
                    total_quantity = stock.quantity + quantity
                    stock.quantity = total_quantity
                    stock.average_price = total_value / total_quantity
                    stock.save()

                # Update user's balance
                user_profile.balance -= total_cost
                user_profile.save()

                # Record the transaction
                Transaction.objects.create(
                    user=user_profile,
                    ticker=ticker,
                    quantity=quantity,
                    price_per_stock=current_price,
                    transaction_type='BUY'
                )

                print(f"Purchase successful. New balance: {user_profile.balance}")
                return redirect('portfolio')
            else:
                print(f"Insufficient funds. Required: {total_cost}, Available: {user_profile.balance}")
                return render(request, 'buy_stock.html', {
                    'error': 'Insufficient funds',
                    'stock_data': {
                        'ticker': ticker,
                        'current_price': current_price
                    }
                })
        except Exception as e:
            print(f"Error in buy_stock: {str(e)}")
            import traceback
            traceback.print_exc()  # This will print the full error traceback
            return render(request, 'buy_stock.html', {
                'error': f'An error occurred while processing your purchase: {str(e)}'
            })
    
    return redirect('portfolio')



@login_required
def buy_stock(request):
    if request.method == 'POST':
        try:
            ticker = request.POST.get('ticker')
            quantity = int(request.POST.get('quantity'))
            
            # Get stock price with more detailed error handling
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('regularMarketPrice')
                
                if current_price is None:
                    current_price = info.get('currentPrice')  # Try alternative price field
                
                if current_price is None or current_price <= 0:
                    print(f"Debug - Stock Info: {info}")  # Debug print
                    messages.error(request, f'Could not fetch valid stock price for {ticker}. Please try again.')
                    return redirect('portfolio')
                
            except Exception as e:
                print(f"Error fetching stock price: {e}")  # Debug print
                messages.error(request, 'Error fetching stock price. Please try again.')
                return redirect('portfolio')
                
            total_cost = quantity * current_price
            
            user_profile = UserProfile.objects.get(user=request.user)
            
            if user_profile.balance >= total_cost:
                # Process purchase
                user_profile.balance -= total_cost
                user_profile.save()
                
                stock_record, created = Stock.objects.get_or_create(
                    user=user_profile,
                    ticker=ticker,
                    defaults={
                        'quantity': quantity,
                        'average_price': current_price
                    }
                )
                
                if not created:
                    # Calculate new average price
                    total_cost_basis = (stock_record.quantity * stock_record.average_price) + (quantity * current_price)
                    new_total_quantity = stock_record.quantity + quantity
                    stock_record.average_price = total_cost_basis / new_total_quantity
                    stock_record.quantity = new_total_quantity
                    stock_record.save()
                
                messages.success(
                    request, 
                    f'Successfully bought {quantity} shares of {ticker} at ₹{current_price:.2f} per share'
                )
            else:
                messages.error(request, f'Insufficient balance. Required: ₹{total_cost:.2f}, Available: ₹{user_profile.balance:.2f}')
            
            return redirect('portfolio')
        except Exception as e:
            print(f"General error: {e}")  # Debug print
            messages.error(request, f'Error: {str(e)}')
            return redirect('portfolio')

    return redirect('portfolio')


@login_required
def sell_stock(request):
    if request.method == 'POST':
        ticker = request.POST['ticker']
        quantity = int(request.POST['quantity'])
        current_price = float(request.POST['current_price'])

        user_profile = UserProfile.objects.get(user=request.user)
        stock = Stock.objects.get(user=user_profile, ticker=ticker)

        if stock and stock.quantity >= quantity:
            # Update stock quantity or delete if fully sold
            stock.quantity -= quantity
            if stock.quantity == 0:
                stock.delete()
            else:
                stock.save()

            # Add funds to user balance
            total_revenue = quantity * current_price
            user_profile.balance += total_revenue
            user_profile.save()

            # Record transaction
            Transaction.objects.create(
                user=user_profile,
                ticker=ticker,
                quantity=quantity,
                price_per_stock=current_price,
                transaction_type='SELL'
            )

        return redirect('portfolio')
    return redirect('portfolio')

from django.shortcuts import render, redirect
from django.contrib.auth import logout, authenticate
from django.contrib.auth.decorators import login_required

def signup(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Create UserProfile with initial balance (e.g., $10000)
            UserProfile.objects.create(
                user=user,
                balance=10000.00  # Set initial balance
            )
            login(request, user)
            return redirect('portfolio')
    else:
        form = UserForm()
    return render(request, 'signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)  # Log the user in
            return redirect('portfolio')  # Redirect to the home page
    return render(request, 'login.html')

def logout_view(request):
    logout(request)  # Logs out the user
    return redirect('login')  # Redirects to the login page

@login_required
def get_stock_price(request, ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = (
            info.get('regularMarketPrice') or 
            info.get('currentPrice') or 
            info.get('price', 0)
        )
        
        print(f"Fetching price for {ticker}: {current_price}")  # Debug log
        
        return JsonResponse({
            'price': current_price,
            'previousClose': info.get('previousClose', 0),
            'currency': info.get('currency', 'USD'),
            'name': info.get('shortName', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume', 'N/A'),
            'change': info.get('regularMarketChangePercent', 0)
        })
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")  # Debug log
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def watchlist_view(request):
    return render(request, 'watchlist.html')

@login_required
def stock_detail(request, ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Print debug information
        print(f"Fetching details for {ticker}")
        print(f"Info received: {info.keys()}")
        
        stock_data = {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'current_price': info.get('regularMarketPrice', 0),
            'price_change': info.get('regularMarketChangePercent', 0),
            'market_cap': format_market_cap(info.get('marketCap', 0)),
            'volume': format_volume(info.get('volume', 0)),
            'year_high': info.get('fiftyTwoWeekHigh', 0),
            'year_low': info.get('fiftyTwoWeekLow', 0),
        }
        
        print(f"Processed stock data: {stock_data}")  # Debug print
        return render(request, 'stock_detail.html', {'stock': stock_data})
    except Exception as e:
        print(f"Error in stock_detail: {str(e)}")  # Debug print
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def get_stock_history(request, ticker, timeframe):
    try:
        print(f"Fetching {timeframe} history for {ticker}")  # Debug print
        stock = yf.Ticker(ticker)
        
        timeframe_map = {
            '1d': ('1d', '5m'),
            '5d': ('5d', '15m'),
            '1mo': ('1mo', '1d'),
            '6mo': ('6mo', '1d'),
            '1y': ('1y', '1d')
        }
        
        period, interval = timeframe_map.get(timeframe, ('1mo', '1d'))
        history = stock.history(period=period, interval=interval)
        
        dates = history.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        prices = history['Close'].tolist()
        
        print(f"Retrieved {len(dates)} data points")  # Debug print
        
        return JsonResponse({
            'dates': dates,
            'prices': prices
        })
    except Exception as e:
        print(f"Error in get_stock_history: {str(e)}")  # Debug print
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def get_stock_news(request, ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        articles = []
        for item in news[:10]:  # Get latest 10 news items
            articles.append({
                'title': item.get('title'),
                'summary': item.get('summary'),
                'url': item.get('link'),
                'published': item.get('published')
            })
            
        return JsonResponse({'articles': articles})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

def format_market_cap(market_cap):
    if market_cap >= 1e12:
        return f"₹{market_cap/1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"₹{market_cap/1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"₹{market_cap/1e6:.2f}M"
    else:
        return f"₹{market_cap:.2f}"

def format_volume(volume):
    if volume >= 1e9:
        return f"{volume/1e9:.2f}B"
    elif volume >= 1e6:
        return f"{volume/1e6:.2f}M"
    elif volume >= 1e3:
        return f"{volume/1e3:.2f}K"
    else:
        return str(volume)

@login_required
def add_balance(request):
    if request.method == 'POST':
        try:
            amount = float(request.POST.get('amount', 0))
            if amount <= 0:
                messages.error(request, 'Please enter a valid amount greater than 0.')
                return redirect('portfolio')

            user_profile = UserProfile.objects.get(user=request.user)
            user_profile.balance += amount
            user_profile.save()

            messages.success(request, f'Successfully added ₹{amount:,.2f} to your balance.')
        except ValueError:
            messages.error(request, 'Please enter a valid number.')
        except Exception as e:
            messages.error(request, f'An error occurred: {str(e)}')
    
    return redirect('portfolio')

@login_required
def remove_funds(request):
    if request.method == 'POST':
        try:
            amount = float(request.POST.get('amount'))
            user_profile = UserProfile.objects.get(user=request.user)
            
            if amount <= 0:
                messages.error(request, 'Please enter a valid amount')
            elif amount > user_profile.balance:
                messages.error(request, 'Insufficient balance')
            else:
                user_profile.balance -= amount
                user_profile.save()
                messages.success(request, f'Successfully removed ₹{amount:.2f} from your balance')
            
            return redirect('portfolio')
        except ValueError:
            messages.error(request, 'Please enter a valid amount')
            return redirect('portfolio')
        except Exception as e:
            messages.error(request, str(e))
            return redirect('portfolio')
    return redirect('portfolio')
