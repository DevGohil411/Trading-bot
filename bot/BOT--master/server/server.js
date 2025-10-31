const express = require('express');
const cors = require('cors');
const TelegramBot = require('node-telegram-bot-api');
require('dotenv').config();
const session = require('express-session');
const passport = require('passport');
const GoogleStrategy = require('passport-google-oauth20').Strategy;

const app = express();
const PORT = process.env.PORT || 8000;

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true,
}));
app.use(express.json());

// Session and Passport setup
app.use(
  session({
    secret: process.env.SESSION_SECRET || 'dev_secret_change_me',
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      sameSite: 'lax',
      secure: false,
    },
  })
);
app.use(passport.initialize());
app.use(passport.session());

passport.serializeUser((user, done) => {
  done(null, user);
});

passport.deserializeUser((obj, done) => {
  done(null, obj);
});

const hasGoogleCreds = !!(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET);

if (hasGoogleCreds) {
  passport.use(
    new GoogleStrategy(
      {
        clientID: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
        callbackURL: '/auth/google/callback',
      },
      async (accessToken, refreshToken, profile, done) => {
        try {
          const user = {
            id: profile.id,
            email: profile.emails && profile.emails[0] ? profile.emails[0].value : null,
            name: profile.displayName,
          };
          return done(null, user);
        } catch (error) {
          return done(error);
        }
      }
    )
  );

  // Google OAuth routes
  app.get('/auth/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

  app.get(
    '/auth/google/callback',
    passport.authenticate('google', { failureRedirect: 'http://localhost:3000/login' }),
    (req, res) => {
      res.redirect('http://localhost:3000/dashboard');
    }
  );
} else {
  // Fallback handlers when creds missing
  app.get('/auth/google', (req, res) => {
    res.status(503).json({ message: 'Google OAuth not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in server/.env' });
  });
  app.get('/auth/google/callback', (req, res) => {
    res.status(503).json({ message: 'Google OAuth not configured' });
  });
}

// Auth status and logout
app.get('/auth/status', (req, res) => {
  res.json({ authenticated: !!req.user, user: req.user || null });
});

app.get('/logout', (req, res) => {
  req.logout(() => {
    res.json({ success: true });
  });
});

// Initialize Telegram Bot
let bot = null;
if (process.env.TELEGRAM_BOT_TOKEN) {
  try {
    bot = new TelegramBot(process.env.TELEGRAM_BOT_TOKEN, { polling: true });
    console.log('✅ Telegram bot initialized successfully');
  } catch (error) {
    console.error('❌ Failed to initialize Telegram bot:', error);
  }
}

// Store authorized users (in production, use a database)
const authorizedUsers = new Set();

// Mock data for development
const mockData = {
  accounts: [
    {
      login: "12345678",
      server: "ICMarkets-Live01",
      name: "Live Trading Account",
      balance: 10000.00,
      equity: 10250.00,
      margin: 500.00,
      freeMargin: 9750.00,
      marginLevel: 2050.0,
      connected: true,
      winRate: 73.5,
      totalTrades: 156,
      avgWin: 45.20,
      avgLoss: -28.50
    }
  ],
  trades: {
    open_trades: [
      {
        ticket: 123456,
        symbol: "EURUSD",
        type: "BUY",
        volume: 0.1,
        openPrice: 1.0850,
        currentPrice: 1.0875,
        profit: 25.00,
        swap: 0.00,
        openTime: "2024-12-19T10:30:00Z"
      },
      {
        ticket: 123457,
        symbol: "GBPUSD",
        type: "SELL",
        volume: 0.1,
        openPrice: 1.2650,
        currentPrice: 1.2635,
        profit: 15.00,
        swap: 0.00,
        openTime: "2024-12-19T11:15:00Z"
      }
    ],
    closed_trades: [
      {
        ticket: 123450,
        symbol: "EURUSD",
        type: "BUY",
        volume: 0.1,
        openPrice: 1.0820,
        closePrice: 1.0865,
        profit: 45.00,
        swap: 2.50,
        openTime: "2024-12-19T09:00:00Z",
        closeTime: "2024-12-19T10:00:00Z"
      },
      {
        ticket: 123451,
        symbol: "USDJPY",
        type: "SELL",
        volume: 0.1,
        openPrice: 148.50,
        closePrice: 148.70,
        profit: -20.00,
        swap: 1.50,
        openTime: "2024-12-19T08:30:00Z",
        closeTime: "2024-12-19T09:30:00Z"
      }
    ]
  },
  botSettings: {
    bot_active: false,
    killzone: true,
    strategy: "mmxm",
    all_strategies: false,
    selected_strategies: ["mmxm"],
    killzone_map: { "mmxm": true },
    risk_settings: {
      max_daily_loss: 500,
      max_daily_profit: 1000,
      risk_per_trade: 1.5,
      default_lot_size: 0.05,
      max_open_trades: 20,
      max_daily_trades: 50,
      max_slippage: 3,
      max_spread: 20,
    },
    trading_sessions: {
      enabled: true,
      london_session: true,
      new_york_session: true,
      tokyo_session: false,
      sydney_session: false,
    },
    notifications: {
      email_notifications: false,
      telegram_notifications: true,
      trade_alerts: true,
      error_alerts: true,
      push_notifications: false,
    },
    news_filter: false,
    volatility_filter: true,
    trend_filter: true,
    advanced_settings: {
      max_slippage: 3,
      max_spread: 20,
      auto_reconnect: true,
      emergency_stop: false,
    }
  },
  botStatus: {
    running: false,
    lastUpdate: new Date().toISOString()
  },
  activityLog: []
};

// Helper to push to activity log
function pushLog(entry) {
  mockData.activityLog.unshift({
    type: entry.type || 'bot',
    title: entry.title || 'Update',
    details: entry.details || '',
    timestamp: new Date().toISOString(),
    tag: entry.tag || 'system'
  });
  mockData.activityLog = mockData.activityLog.slice(0, 200);
}

// MT5 login/connect endpoint (mock)
app.post('/api/mt5/login', (req, res) => {
  const { login, password, server, name } = req.body || {};
  if (!login || !password || !server) {
    return res.status(400).json({ success: false, message: 'Missing MT5 credentials' });
  }
  const exists = mockData.accounts.find(a => a.login === login);
  const newAcc = {
    login: String(login),
    server: server,
    name: name || 'MT5 Account',
    balance: 10000.00,
    equity: 10000.00,
    margin: 0.00,
    freeMargin: 10000.00,
    marginLevel: 0,
    connected: true,
    winRate: 0,
    totalTrades: 0,
    avgWin: 0,
    avgLoss: 0
  };
  if (exists) {
    Object.assign(exists, newAcc);
  } else {
    mockData.accounts.push(newAcc);
  }
  pushLog({ type: 'connection', title: 'MT5 Connected', details: `${login} @ ${server}`, tag: 'mt5' });
  return res.json({ success: true });
});

// Activity log endpoint
app.get('/api/activity-log', (req, res) => {
  res.json(mockData.activityLog);
});

// Endpoint to receive user info from NextAuth (Snipr-X app)
app.post('/api/user/google', (req, res) => {
  const { email, name } = req.body || {};
  console.log('Received user from NextAuth:', { email, name });
  res.json({ success: true });
});

// API Routes

// Account data
app.get('/api/account', (req, res) => {
  res.json(mockData.accounts);
});

// MT5 Accounts
app.get('/api/mt5/accounts', (req, res) => {
  res.json(mockData.accounts);
});

// MT5 Trades
app.get('/api/mt5/trades', (req, res) => {
  res.json(mockData.trades);
});

// Bot Settings
app.get('/api/bot/settings', (req, res) => {
  res.json(mockData.botSettings);
});

app.post('/api/bot/settings', (req, res) => {
  const updates = req.body;
  Object.assign(mockData.botSettings, updates);
  
  if (updates.bot_active !== undefined && bot) {
    const status = updates.bot_active ? 'Bot Started' : 'Bot Stopped';
    const details = `Strategy: ${updates.strategy || mockData.botSettings.strategy}`;
    authorizedUsers.forEach(chatId => {
      bot.sendMessage(chatId, `🤖 ${status}\n📝 ${details}\n⏰ ${new Date().toLocaleString()}`);
    });
  }
  
  res.json({ success: true, settings: mockData.botSettings });
});

// Bot Status
app.get('/api/bot/status', (req, res) => {
  res.json(mockData.botStatus);
});

// Bot Start
app.post('/api/bot/start', (req, res) => {
  mockData.botStatus.running = true;
  mockData.botStatus.lastUpdate = new Date().toISOString();
  
  if (bot) {
    authorizedUsers.forEach(chatId => {
      bot.sendMessage(chatId, `🟢 Bot Started\n📝 Strategy: ${mockData.botSettings.strategy}\n⏰ ${new Date().toLocaleString()}`);
    });
  }
  pushLog({ type: 'bot', title: 'Bot Started', details: `Strategy: ${mockData.botSettings.strategy}`, tag: 'bot' });
  
  res.json({ success: true, status: 'Bot started' });
});

// Bot Stop
app.post('/api/bot/stop', (req, res) => {
  mockData.botStatus.running = false;
  mockData.botStatus.lastUpdate = new Date().toISOString();
  
  if (bot) {
    authorizedUsers.forEach(chatId => {
      bot.sendMessage(chatId, `🔴 Bot Stopped\n⏰ ${new Date().toLocaleString()}`);
    });
  }
  pushLog({ type: 'bot', title: 'Bot Stopped', details: '', tag: 'bot' });
  
  res.json({ success: true, status: 'Bot stopped' });
});

// Telegram Bot Info
app.get('/api/telegram/bot-info', (req, res) => {
  const botUsername = process.env.TELEGRAM_BOT_USERNAME || 'SniprXBot';
  res.json({
    success: true,
    botInfo: {
      username: botUsername.startsWith('@') ? botUsername : `@${botUsername}`
    }
  });
});

// Telegram authorization endpoints
app.post('/api/telegram/authorize', (req, res) => {
  const { chatId } = req.body || {};
  if (!chatId) return res.status(400).json({ success: false, message: 'chatId required' });
  authorizedUsers.add(Number(chatId));
  res.json({ success: true });
});

app.post('/api/telegram/deauthorize', (req, res) => {
  const { chatId } = req.body || {};
  if (!chatId) return res.status(400).json({ success: false, message: 'chatId required' });
  authorizedUsers.delete(Number(chatId));
  res.json({ success: true });
});

// Telegram Notifications
app.post('/api/telegram/notify', (req, res) => {
  const { type, data } = req.body;
  
  if (!bot) {
    return res.status(500).json({ message: 'Telegram bot not initialized' });
  }
  
  let message = '';
  
  switch (type) {
    case 'login':
      message = `🔐 User Login\n👤 User: ${data.userEmail}\n🆔 ID: ${data.userId}\n⏰ ${new Date().toLocaleString()}`;
      break;
    case 'logout':
      message = `🚪 User Logout\n👤 User: ${data.userEmail}\n🆔 ID: ${data.userId}\n⏰ ${new Date().toLocaleString()}`;
      break;
    case 'trade':
      const profit = data.trade.profit >= 0 ? `+$${data.trade.profit.toFixed(2)}` : `-$${Math.abs(data.trade.profit).toFixed(2)}`;
      const emoji = data.trade.profit >= 0 ? '🟢' : '🔴';
      message = `${emoji} TRADE\n📊 Symbol: ${data.trade.symbol}\n📈 Type: ${data.trade.type}\n📊 Volume: ${data.trade.volume}L\n💰 Profit: ${profit}\n⏰ ${new Date().toLocaleString()}`;
      break;
    case 'bot_status':
      const statusEmoji = data.status.toLowerCase().includes('start') ? '🟢' : '🔴';
      message = `${statusEmoji} Bot Status Update\n🔄 Status: ${data.status}\n📝 Details: ${data.details || ''}\n⏰ ${new Date().toLocaleString()}`;
      break;
    case 'error':
      message = `⚠️ Error Alert\n❌ Error: ${data.error}\n⏰ ${new Date().toLocaleString()}`;
      break;
    default:
      return res.status(400).json({ message: 'Invalid notification type' });
  }
  
  let sentCount = 0;
  authorizedUsers.forEach(chatId => {
    bot.sendMessage(chatId, message, { parse_mode: 'HTML' })
      .then(() => sentCount++)
      .catch(error => console.error(`Failed to send message to ${chatId}:`, error));
  });
  
  res.json({ success: true, message: `Notification sent to ${sentCount} users` });
});

// Telegram Bot Commands (if bot is initialized)
if (bot) {
  bot.onText(/\/start/, (msg) => {
    const chatId = msg.chat.id;
    const welcomeMessage = `
🤖 Welcome to SniprX Trading Bot!

I'm here to keep you updated on your trading activities. Here are the available commands:

📊 /status - Get current trading status
💰 /balance - Check account balance
📈 /trades - View recent trades
🔔 /notifications - Toggle notifications
❓ /help - Show this help message

To receive notifications, please contact the admin to authorize your chat ID: ${chatId}
    `;
    bot.sendMessage(chatId, welcomeMessage);
  });

  bot.onText(/\/help/, (msg) => {
    const chatId = msg.chat.id;
    const helpMessage = `
🤖 SniprX Trading Bot Commands:

📊 /status - Get current trading status and bot status
💰 /balance - Check your MT5 account balance and equity
📈 /trades - View recent open and closed trades
🔔 /notifications - Toggle trading notifications on/off
📋 /summary - Get trading summary and performance stats
❓ /help - Show this help message

For support, contact your trading administrator.
    `;
    bot.sendMessage(chatId, helpMessage);
  });

  bot.onText(/\/status/, async (msg) => {
    const chatId = msg.chat.id;
    if (!authorizedUsers.has(chatId)) {
      bot.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
      return;
    }

    const account = mockData.accounts[0];
    const status = `
📊 Trading Status

🔄 Connection Status: ${account.connected ? '🟢 Connected' : '🔴 Disconnected'}
💰 Balance: $${account.balance.toFixed(2)}
📈 Equity: $${account.equity.toFixed(2)}
📊 Open Trades: ${mockData.trades.open_trades.length}
✅ Closed Trades: ${mockData.trades.closed_trades.length}
🎯 Win Rate: ${account.winRate.toFixed(1)}%
    `;
    bot.sendMessage(chatId, status, { parse_mode: 'HTML' });
  });

  bot.onText(/\/balance/, async (msg) => {
    const chatId = msg.chat.id;
    if (!authorizedUsers.has(chatId)) {
      bot.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
      return;
    }

    const account = mockData.accounts[0];
    const balance = `
💰 Account Balance

🏦 Balance: $${account.balance.toFixed(2)}
📊 Equity: $${account.equity.toFixed(2)}
💼 Margin: $${account.margin.toFixed(2)}
🆓 Free Margin: $${account.freeMargin.toFixed(2)}
📈 Margin Level: ${account.marginLevel.toFixed(1)}%
    `;
    bot.sendMessage(chatId, balance, { parse_mode: 'HTML' });
  });

  bot.onText(/\/trades/, async (msg) => {
    const chatId = msg.chat.id;
    if (!authorizedUsers.has(chatId)) {
      bot.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
      return;
    }

    let message = `📈 Recent Trades\n\n`;

    if (mockData.trades.open_trades.length > 0) {
      message += `🔄 Open Trades (${mockData.trades.open_trades.length}):\n`;
      mockData.trades.open_trades.slice(0, 5).forEach(trade => {
        const profit = trade.profit >= 0 ? `+$${trade.profit.toFixed(2)}` : `-$${Math.abs(trade.profit).toFixed(2)}`;
        const emoji = trade.profit >= 0 ? '🟢' : '🔴';
        message += `${emoji} ${trade.symbol} ${trade.type} ${trade.volume}L - ${profit}\n`;
      });
    }

    if (mockData.trades.closed_trades.length > 0) {
      message += `\n✅ Recent Closed Trades (${mockData.trades.closed_trades.length}):\n`;
      mockData.trades.closed_trades.slice(0, 5).forEach(trade => {
        const profit = trade.profit >= 0 ? `+$${trade.profit.toFixed(2)}` : `-$${Math.abs(trade.profit).toFixed(2)}`;
        const emoji = trade.profit >= 0 ? '🟢' : '🔴';
        message += `${emoji} ${trade.symbol} ${trade.type} ${trade.volume}L - ${profit}\n`;
      });
    }

    bot.sendMessage(chatId, message, { parse_mode: 'HTML' });
  });

  bot.onText(/\/summary/, async (msg) => {
    const chatId = msg.chat.id;
    if (!authorizedUsers.has(chatId)) {
      bot.sendMessage(chatId, "❌ You are not authorized to use this bot. Please contact the administrator.");
      return;
    }

    const closedTrades = mockData.trades.closed_trades;
    const wins = closedTrades.filter(t => t.profit > 0);
    const losses = closedTrades.filter(t => t.profit < 0);
    const totalProfit = closedTrades.reduce((sum, t) => sum + t.profit, 0);
    const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + t.profit, 0) / wins.length : 0;
    const avgLoss = losses.length > 0 ? losses.reduce((sum, t) => sum + t.profit, 0) / losses.length : 0;

    const summary = `
📋 Trading Summary

💰 Total Profit: ${totalProfit >= 0 ? '+' : ''}$${totalProfit.toFixed(2)}
📊 Total Trades: ${closedTrades.length}
✅ Winning Trades: ${wins.length}
❌ Losing Trades: ${losses.length}
🎯 Win Rate: ${closedTrades.length > 0 ? ((wins.length / closedTrades.length) * 100).toFixed(1) : '0'}%
📈 Average Win: $${avgWin.toFixed(2)}
📉 Average Loss: $${avgLoss.toFixed(2)}
🔄 Open Trades: ${mockData.trades.open_trades.length}
    `;
    bot.sendMessage(chatId, summary, { parse_mode: 'HTML' });
  });

  // Handle polling errors
  bot.on('polling_error', (error) => {
    console.error('Telegram bot polling error:', error.message);
  });
}

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'OK', 
    timestamp: new Date().toISOString(),
    telegramBot: bot ? 'Connected' : 'Not configured',
    googleAuth: hasGoogleCreds ? 'Configured' : 'Not configured'
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running on http://localhost:${PORT}`);
}); 